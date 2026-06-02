#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MobileExplorer 20-task experimental runner.

- slot-complete best-first expansion
- aggressive skip on high-confidence slot-complete branches
- evidence trace + branch/step logging
- safety gates for risky actions
- output JSON/CSV/Markdown report
- default dry-run uses mock planner/explorer
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
import heapq
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


class EvidenceType(str, Enum):
    ACTION_HINT = "ACTION_HINT"
    ANSWER_HINT = "ANSWER_HINT"
    SCHEMA_HINT = "SCHEMA_HINT"
    AVOID_HINT = "AVOID_HINT"
    RISK_HINT = "RISK_HINT"


class BranchStatus(str, Enum):
    ACTIVE = "active"
    DONE = "done"
    BLOCKED = "blocked"


@dataclass
class Evidence:
    type: EvidenceType
    text: str
    confidence: float
    slots: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionCandidate:
    action: Dict[str, Any]
    target_page: Optional[str]
    evidences: List[Evidence] = field(default_factory=list)
    confidence: float = 0.0
    risk_score: float = 0.0
    slot_filled: List[str] = field(default_factory=list)
    slot_complete: bool = False


@dataclass
class PlanBundle:
    step: int
    candidates: List[ActionCandidate] = field(default_factory=list)
    requires_wait: bool = False
    rationale: str = ""
    cache_hit: bool = False


@dataclass
class TaskDefinition:
    task_id: str
    app: str
    mode: str
    prompt: str
    expected_slots: List[str]
    baseline_status: str = "success"


@dataclass
class StepRecord:
    task_step: int
    branch_id: str
    from_page: str
    to_page: str
    planned_page_candidates: List[str]
    chosen_page: Optional[str]
    planned_action: Dict[str, Any]
    actual_action: Dict[str, Any]
    evidences: List[Evidence]
    slot_complete: bool
    slot_coverage: float
    confidence: float
    risk_score: float
    planned_wait: bool = False
    rollback: bool = False
    rollback_success: Optional[bool] = None
    suppress: bool = False
    cache_hit: bool = False
    aggressive_skip: bool = False
    skipped_to_step: Optional[int] = None
    executed_after_skip: bool = False
    action_executed: bool = True
    reason: Optional[str] = None
    cost_ms: float = 0.0
    aligned_injected: int = 0
    applied_slots: List[str] = field(default_factory=list)


@dataclass
class BranchState:
    task_id: str
    app: str
    branch_id: str
    parent_id: Optional[str]
    depth: int
    next_step: int
    current_page: str
    required_slots: List[str]
    filled_slots: List[str] = field(default_factory=list)
    path_actions: List[Dict[str, Any]] = field(default_factory=list)
    candidate_page_options: List[str] = field(default_factory=list)
    pending_candidate: Optional[ActionCandidate] = None
    pending_cache_hit: bool = False
    status: BranchStatus = BranchStatus.ACTIVE
    evidence_context: List[Evidence] = field(default_factory=list)
    selected_pages: List[str] = field(default_factory=list)
    slot_hits: List[bool] = field(default_factory=list)
    score: float = 0.0


@dataclass
class TaskTrace:
    task_id: str
    app: str
    mode: str
    baseline_status: str
    start_ts: float
    end_ts: float = 0.0
    required_slots: List[str] = field(default_factory=list)
    final_filled_slots: List[str] = field(default_factory=list)
    success: bool = False
    rescued: bool = False
    broken: bool = False
    steps: List[StepRecord] = field(default_factory=list)
    branch_records: List[Dict[str, Any]] = field(default_factory=list)
    aggressive_skip_events: List[Dict[str, Any]] = field(default_factory=list)
    rollback_count: int = 0
    wait_count: int = 0
    suppress_count: int = 0
    explore_pages_count: int = 0

    @property
    def step_count(self) -> int:
        return len(self.steps)


@dataclass
class ExperimentConfig:
    max_depth: int = 1
    max_steps: int = 12
    max_frontier: int = 12
    top_k: int = 3
    high_confidence: float = 0.90
    risk_boundary: float = 0.75
    injection_confidence: float = 0.70
    dangerous_actions: Tuple[str, ...] = ("DELETE", "SAVE", "CONFIRM", "TOGGLE", "LONG_PRESS", "INPUT_TEXT")
    cache_file: Path = Path("exploration_evidence_cache.json")
    output_dir: Path = Path("mobile_explorer_20task_outputs")


def to_jsonable(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def safe_action_type(action: Dict[str, Any], dangerous: Tuple[str, ...]) -> bool:
    t = str(action.get("type", "")).strip().upper().replace("-", "_")
    return t in dangerous


def evidence_gain(prev_filled: List[str], add_slots: List[str], required: List[str]) -> float:
    before = len(set(prev_filled))
    after = len(set(prev_filled) | set(add_slots))
    req = max(1, len(required))
    return (after - before) / req


def clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


class EvidenceCache:
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Any] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}

    def get(self, key: str):
        return self.data.get(key)

    def set(self, key: str, value: Any):
        self.data[key] = value

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")


class Planner(ABC):
    @abstractmethod
    def parse_task_slots(self, task: TaskDefinition, state: Dict[str, Any]) -> List[str]:
        ...

    @abstractmethod
    def propose_plan(
        self,
        task: TaskDefinition,
        state: Dict[str, Any],
        required_slots: List[str],
        filled_slots: List[str],
        candidate_pages: List[str],
        target_step: int,
        injected_evidences: List[Evidence]
    ) -> PlanBundle:
        ...


class Explorer(ABC):
    @abstractmethod
    def reset_to_root(self, app: str) -> bool:
        ...

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def list_candidate_pages(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return [{'page_id': str, 'evidence_score': 0..1}, ...]"""

    @abstractmethod
    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @abstractmethod
    def rollback(self) -> bool:
        ...

    @abstractmethod
    def get_current_page(self) -> str:
        ...

    @abstractmethod
    def wait(self, seconds: float = 0.5) -> bool:
        ...


class MockPlanner(Planner):
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def parse_task_slots(self, task: TaskDefinition, state: Dict[str, Any]) -> List[str]:
        return task.expected_slots[:]

    def propose_plan(
        self,
        task: TaskDefinition,
        state: Dict[str, Any],
        required_slots: List[str],
        filled_slots: List[str],
        candidate_pages: List[str],
        target_step: int,
        injected_evidences: List[Evidence]
    ) -> PlanBundle:
        remain = [s for s in required_slots if s not in set(filled_slots)]
        candidates: List[ActionCandidate] = []

        if not remain:
            act = {"type": "WAIT", "target": "task_done"}
            ev = [Evidence(
                EvidenceType.ANSWER_HINT,
                f"任务{task.task_id}在当前状态下已满足所有 slot",
                0.95,
                [],
                {"step": target_step}
            )]
            candidates.append(ActionCandidate(
                action=act,
                target_page=state.get("page_id", ""),
                evidences=ev,
                confidence=0.95,
                risk_score=0.0,
                slot_filled=[],
                slot_complete=True
            ))
            return PlanBundle(step=target_step, candidates=candidates, requires_wait=True, rationale="done")

        for idx, slot in enumerate(remain[:max(1, len(remain))]):
            if candidate_pages:
                page = self.rng.choice(candidate_pages)
            else:
                page = state.get("page_id", "home")

            if len(remain) == 1:
                conf = 0.95
            else:
                conf = 0.82 - idx * 0.08 + self.rng.uniform(-0.03, 0.03)
            conf = clamp(conf)
            risk = 0.12 + self.rng.uniform(0, 0.20)

            ev = [
                Evidence(EvidenceType.ACTION_HINT, f"填写槽位 {slot}", conf, [slot], {"step": target_step}),
                Evidence(EvidenceType.SCHEMA_HINT, f"{task.app} 页面含有字段 {slot}", min(1.0, conf + 0.02), [slot], {})
            ]
            if risk > 0.55:
                ev.append(Evidence(EvidenceType.RISK_HINT, "可能存在非高置信操作，建议谨慎", risk, [slot], {}))

            action = {"type": "OPEN_PAGE", "target_page": page, "slot": slot, "task": task.task_id}
            candidates.append(ActionCandidate(
                action=action,
                target_page=page,
                evidences=ev,
                confidence=conf,
                risk_score=risk,
                slot_filled=[slot],
                slot_complete=(len(remain) == 1)
            ))

        return PlanBundle(step=target_step, candidates=candidates, rationale="mock", cache_hit=False)


class MockExplorer(Explorer):
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.current_app = ""
        self.current_page = ""
        self.history: List[str] = []
        self.app_pages = {
            "OpenTracks": ["home", "recent", "new_activity", "details", "settings"],
            "Calendar": ["home", "day", "week", "event_edit", "event_detail"],
            "Joplin": ["home", "notes", "editor", "search", "settings"],
            "Tasks": ["home", "task_list", "task_detail", "new_task", "tag"],
            "Markor": ["home", "note_list", "editor", "search", "recent"],
        }

    def reset_to_root(self, app: str) -> bool:
        self.current_app = app
        self.current_page = f"{app}:home"
        self.history = []
        return True

    def get_state(self) -> Dict[str, Any]:
        return {"app": self.current_app, "page_id": self.current_page, "ts": time.time()}

    def list_candidate_pages(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        pages = self.app_pages.get(state.get("app", ""), [state.get("page_id", "home")])
        current = state.get("page_id", pages[0])
        cands = []
        for p in pages:
            if p == current:
                continue
            cands.append({"page_id": f"{state.get('app', '')}:{p}", "evidence_score": self.rng.uniform(0.45, 0.98)})
        if not cands:
            cands.append({"page_id": current, "evidence_score": 1.0})
        return cands

    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        t = str(action.get("type", "")).upper().replace("-", "_")
        if t in {"DELETE", "SAVE", "CONFIRM", "TOGGLE", "LONG_PRESS", "INPUT_TEXT"}:
            return {"success": False, "error": "forbidden_action", "page": self.current_page}

        target = action.get("target_page") or action.get("target") or action.get("page")
        next_page = self.current_page

        if t == "OPEN_PAGE":
            if target:
                next_page = target if ":" in str(target) else f"{self.current_app}:{target}"
                self.history.append(self.current_page)
                self.current_page = next_page
                return {"success": True, "page": next_page}
            return {"success": False, "error": "no_target", "page": self.current_page}

        if t == "CLICK":
            if not target:
                return {"success": False, "error": "no_target", "page": self.current_page}
            next_page = target if ":" in str(target) else f"{self.current_app}:{target}"
            self.history.append(self.current_page)
            self.current_page = next_page
            return {"success": True, "page": next_page}

        if t == "WAIT":
            return {"success": True, "page": self.current_page}

        return {"success": False, "error": f"unsupported:{t}", "page": self.current_page}

    def rollback(self) -> bool:
        if not self.history:
            return False
        self.current_page = self.history.pop()
        return True

    def get_current_page(self) -> str:
        return self.current_page

    def wait(self, seconds: float = 0.5) -> bool:
        time.sleep(seconds)
        return True


def plan_to_raw(bundle: PlanBundle) -> Dict[str, Any]:
    return {
        "step": bundle.step,
        "requires_wait": bundle.requires_wait,
        "rationale": bundle.rationale,
        "candidates": [
            {
                "action": c.action,
                "target_page": c.target_page,
                "confidence": c.confidence,
                "risk_score": c.risk_score,
                "slot_filled": c.slot_filled,
                "slot_complete": c.slot_complete,
                "evidences": [
                    {
                        "type": e.type.value,
                        "text": e.text,
                        "confidence": e.confidence,
                        "slots": e.slots,
                        "meta": e.meta,
                    }
                    for e in c.evidences
                ],
            }
            for c in bundle.candidates
        ],
    }


def plan_from_raw(raw: Dict[str, Any]) -> PlanBundle:
    cands = []
    for c in raw.get("candidates", []):
        evidences = [
            Evidence(
                EvidenceType(c.get("type", EvidenceType.ACTION_HINT)),
                c.get("text", ""),
                float(c.get("confidence", 0.0)),
                c.get("slots", []),
                c.get("meta", {}),
            )
            for c in c.get("evidences", [])
        ]
        cands.append(ActionCandidate(
            action=c.get("action", {}),
            target_page=c.get("target_page"),
            evidences=evidences,
            confidence=float(c.get("confidence", 0.0)),
            risk_score=float(c.get("risk_score", 0.0)),
            slot_filled=c.get("slot_filled", []),
            slot_complete=bool(c.get("slot_complete", False)),
        ))

    return PlanBundle(
        step=int(raw.get("step", 1)),
        candidates=cands,
        requires_wait=bool(raw.get("requires_wait", False)),
        rationale=str(raw.get("rationale", "")),
        cache_hit=True,
    )


def build_default_tasks() -> List[TaskDefinition]:
    return [
        TaskDefinition("OpenTracks_01", "OpenTracks", "create", "记录 5 分钟慢跑活动，保存记录。", ["duration", "distance", "speed", "save"], "failure"),
        TaskDefinition("OpenTracks_02", "OpenTracks", "edit", "为最近一次活动补充备注。", ["activity", "note"], "success"),
        TaskDefinition("OpenTracks_03", "OpenTracks", "search", "在今日活动中按距离筛选最近一次。", ["date", "distance"], "failure"),
        TaskDefinition("OpenTracks_04", "OpenTracks", "create", "新建一次 15 分钟步行记录。", ["duration", "type", "save"], "success"),

        TaskDefinition("Calendar_01", "Calendar", "create", "新建明天 9:00 的会议事件。", ["title", "date", "time", "save"], "success"),
        TaskDefinition("Calendar_02", "Calendar", "edit", "修改该会议为 10:00。", ["title", "new_time", "confirm"], "failure"),
        TaskDefinition("Calendar_03", "Calendar", "search", "查找“项目回顾”相关事件。", ["query"], "success"),
        TaskDefinition("Calendar_04", "Calendar", "remind", "给会议添加提醒。", ["title", "remind_time", "save"], "failure"),

        TaskDefinition("Joplin_01", "Joplin", "create", "新建一条标题为“实验记录”的笔记并保存。", ["title", "content", "save"], "success"),
        TaskDefinition("Joplin_02", "Joplin", "edit", "在“实验记录”笔记中添加标签。", ["note", "tag", "apply"], "success"),
        TaskDefinition("Joplin_03", "Joplin", "search", "在 Joplin 搜索“实验”。", ["query"], "failure"),
        TaskDefinition("Joplin_04", "Joplin", "open", "打开最近编辑的笔记并检查内容。", ["note", "open"], "success"),

        TaskDefinition("Tasks_01", "Tasks", "create", "创建一个高优先级任务：提交实验报告。", ["title", "priority", "save"], "success"),
        TaskDefinition("Tasks_02", "Tasks", "edit", "将该任务截止日期改为明天。", ["title", "due_date", "save"], "failure"),
        TaskDefinition("Tasks_03", "Tasks", "complete", "将“提交实验报告”标记为完成。", ["title", "complete"], "success"),
        TaskDefinition("Tasks_04", "Tasks", "search", "搜索“实验”相关任务。", ["query"], "failure"),

        TaskDefinition("Markor_01", "Markor", "create", "新建 markdown 文件《实验记录.md》。", ["filename", "save"], "success"),
        TaskDefinition("Markor_02", "Markor", "edit", "向文件添加任务清单项。", ["filename", "item", "save"], "failure"),
        TaskDefinition("Markor_03", "Markor", "search", "在 Markor 搜索“实验”。", ["query"], "success"),
        TaskDefinition("Markor_04", "Markor", "open", "打开最近文件并回到编辑状态。", ["filename", "open"], "success"),

    ]


class ExperimentRunner:
    def __init__(self, planner: Planner, explorer: Explorer, cfg: ExperimentConfig, tasks: List[TaskDefinition]):
        self.planner = planner
        self.explorer = explorer
        self.cfg = cfg
        self.tasks = tasks
        self.cache = EvidenceCache(cfg.cache_file)
        self._frontier_counter = 0
        self._global_path: List[Dict[str, Any]] = []

    def _push_frontier(self, heap: List[Tuple[float, int, str, BranchState]], branch: BranchState):
        self._frontier_counter += 1
        heapq.heappush(heap, (-branch.score, self._frontier_counter, branch.branch_id, branch))

    def _pop_frontier(self, heap: List[Tuple[float, int, str, BranchState]]) -> Optional[BranchState]:
        if not heap:
            return None
        _, _, _, b = heapq.heappop(heap)
        return b

    def _stratify(self, candidates: List[ActionCandidate], k: int) -> List[ActionCandidate]:
        if not candidates:
            return []
        cands = sorted(candidates, key=lambda c: (c.confidence, -c.risk_score), reverse=True)
        n = len(cands)
        a = max(1, math.ceil(0.4 * n))
        b = max(1, math.ceil(0.3 * n))
        high = cands[:a]
        mid = cands[a:a + b]
        low = cands[a + b:]
        return (high + mid + low)[:k]

    def _score_branch(self, candidate: ActionCandidate, branch: BranchState, required: List[str], depth: int) -> float:
        cur_cov = len(set(branch.filled_slots)) / max(1, len(required))
        after = len(set(branch.filled_slots) | set(candidate.slot_filled)) / max(1, len(required))
        evidence_term = evidence_gain(branch.filled_slots, candidate.slot_filled, required)
        progress_term = after - cur_cov
        risk_term = -candidate.risk_score
        novelty_term = 0.0
        if branch.current_page:
            novelty_term = 1.0 - min(1.0, len(set(branch.selected_pages) | {branch.current_page}) / max(1, len(branch.selected_pages) + 1))
        depth_term = -0.05 * depth
        return (
            0.40 * candidate.confidence
            + 0.25 * evidence_term
            + 0.20 * progress_term
            + 0.10 * novelty_term
            + depth_term
            + risk_term
        )

    def _cache_key(self, task: TaskDefinition, state: Dict[str, Any], required_slots: List[str], filled_slots: List[str], target_step: int) -> str:
        payload = {
            "task": task.task_id,
            "app": task.app,
            "step": target_step,
            "page": state.get("page_id", ""),
            "required": sorted(required_slots),
            "filled": sorted(filled_slots),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _plan(self, task: TaskDefinition, state: Dict[str, Any], required_slots: List[str], filled_slots: List[str], target_step: int, injected: List[Evidence]) -> PlanBundle:
        cand_pages = [c.get("page_id", "") for c in self.explorer.list_candidate_pages(state)]
        key = self._cache_key(task, state, required_slots, filled_slots, target_step)
        cached = self.cache.get(key)
        if cached is not None:
            bundle = plan_from_raw(cached)
            bundle.cache_hit = True
            return bundle

        bundle = self.planner.propose_plan(task, state, required_slots, filled_slots, cand_pages, target_step, injected)
        self.cache.set(key, plan_to_raw(bundle))
        return bundle

    def _restore_to_path(self, target_path: List[Dict[str, Any]]) -> bool:
        while len(self._global_path) > len(target_path):
            if not self.explorer.rollback():
                return False
            self._global_path.pop()

        i = len(self._global_path)
        while i < len(target_path):
            act = target_path[i]
            ret = self.explorer.execute_action(act)
            if not ret.get("success", False):
                return False
            self._global_path.append(act)
            i += 1
        return True

    def _run_action(
        self,
        task: TaskDefinition,
        branch: BranchState,
        candidate: ActionCandidate,
        task_step: int,
        plan_candidates: List[str],
        aligned_injected: int,
        required_slots: List[str],
        executed_after_skip: bool = False,
    ) -> Tuple[StepRecord, List[str], bool]:
        start = time.perf_counter()
        from_page = self.explorer.get_current_page()
        atype = str(candidate.action.get("type", "")).upper().replace("-", "_")
        risky = candidate.risk_score >= self.cfg.risk_boundary
        suspicious = safe_action_type(candidate.action, self.cfg.dangerous_actions)

        rollback = False
        rollback_success = None
        reason = None
        action_executed = False
        if suspicious:
            suppress = True
            actual_action = {"type": "SUPPRESS", "reason": "forbidden_speculative", "original": candidate.action}
            result_page = from_page
            success = False
            reason = "suppressed_non_speculative"
        elif risky:
            suppress = True
            result_page = from_page
            success = False
            actual_action = {"type": "BLOCK_RISK", "reason": "risk_boundary", "original": candidate.action}
            rollback = True
            rollback_success = self.explorer.rollback()
            reason = "risk_boundary"
        else:
            suppress = False
            result = self.explorer.execute_action(candidate.action)
            success = bool(result.get("success", False))
            action_executed = success or atype == "WAIT"
            result_page = result.get("page", self.explorer.get_current_page())
            if not success and atype != "WAIT":
                rollback = True
                rollback_success = self.explorer.rollback()
                reason = result.get("error", "action_failed")
            actual_action = dict(candidate.action)

        applied = list(branch.filled_slots)
        if action_executed and (not suppress) and (not (not success and atype != "WAIT") and not (successful := False)):
            # placeholder keeps mypy/linters simple for this script
            pass
        if action_executed and not suppress and not risky and success:
            applied = sorted(set(applied) | set(candidate.slot_filled))

        required_set = set(required_slots)
        slot_complete = (len(required_set) == 0) or (required_set.issubset(set(applied)))
        coverage = len(set(applied)) / max(1, len(required_set)) if required_set else 1.0

        rec = StepRecord(
            task_step=task_step,
            branch_id=branch.branch_id,
            from_page=from_page,
            to_page=result_page,
            planned_page_candidates=plan_candidates,
            chosen_page=result_page if isinstance(result_page, str) else from_page,
            planned_action=dict(candidate.action),
            actual_action=actual_action,
            evidences=list(candidate.evidences),
            slot_complete=slot_complete,
            slot_coverage=clamp(coverage),
            confidence=clamp(candidate.confidence),
            risk_score=clamp(candidate.risk_score),
            planned_wait=(atype == "WAIT"),
            rollback=rollback,
            rollback_success=rollback_success,
            suppress=suppress,
            cache_hit=branch.pending_cache_hit,
            aggressive_skip=False,
            skipped_to_step=None,
            executed_after_skip=executed_after_skip,
            action_executed=action_executed,
            reason=reason,
            cost_ms=(time.perf_counter() - start) * 1000.0,
            aligned_injected=aligned_injected,
            applied_slots=applied,
        )
        return rec, applied, action_executed

    def _can_expand(self, branch: BranchState, last_rec: StepRecord) -> bool:
        if branch.depth >= self.cfg.max_depth:
            return False
        if last_rec.rollback or last_rec.suppress:
            return False
        if branch.depth >= 1:
            return (
                last_rec.slot_complete
                and last_rec.confidence >= self.cfg.high_confidence
                and last_rec.risk_score <= self.cfg.risk_boundary
                and not last_rec.planned_wait
            )
        return True

    def _should_aggressive_skip(self, branch: BranchState, last_rec: StepRecord) -> bool:
        return (
            branch.depth >= 1
            and branch.depth < self.cfg.max_depth
            and last_rec.slot_complete
            and last_rec.confidence >= self.cfg.high_confidence
            and last_rec.risk_score <= self.cfg.risk_boundary
            and not last_rec.rollback
            and not last_rec.suppress
            and not last_rec.planned_wait
        )

    def _clone_child(
        self,
        parent: BranchState,
        required_slots: List[str],
        candidate: ActionCandidate,
        candidate_pages: List[str],
        score: float,
        depth: int,
        next_step: int,
        selected_pages: List[str],
    ) -> BranchState:
        return BranchState(
            task_id=parent.task_id,
            app=parent.app,
            branch_id=str(random.randrange(10**18)),
            parent_id=parent.branch_id,
            depth=depth,
            next_step=next_step,
            current_page=parent.current_page,
            required_slots=required_slots,
            filled_slots=list(parent.filled_slots),
            path_actions=list(parent.path_actions),
            candidate_page_options=candidate_pages,
            pending_candidate=candidate,
            pending_cache_hit=False,
            status=BranchStatus.ACTIVE,
            evidence_context=list(parent.evidence_context),
            selected_pages=list(selected_pages),
            slot_hits=list(parent.slot_hits),
            score=score,
        )

    def _collect_branch_records(self, trace: TaskTrace):
        grouped: Dict[str, Dict[str, List[Any]]] = {}
        for st in trace.steps:
            b = grouped.setdefault(st.branch_id, {"branch_id": st.branch_id, "pages": [], "slot_hits": []})
            b["pages"].append(st.from_page)
            b["pages"].append(st.to_page)
            b["slot_hits"].append(int(st.slot_complete))
        trace.branch_records = [
            {"branch_id": k, "pages": v["pages"], "slot_hits": v["slot_hits"]}
            for k, v in grouped.items()
        ]

    def run_task(self, task: TaskDefinition) -> TaskTrace:
        trace = TaskTrace(
            task_id=task.task_id,
            app=task.app,
            mode=task.mode,
            baseline_status=task.baseline_status,
            start_ts=time.time(),
        )

        if not self.explorer.reset_to_root(task.app):
            trace.end_ts = time.time()
            return trace

        root_state = self.explorer.get_state()
        parsed_slots = self.planner.parse_task_slots(task, root_state)
        required = task.expected_slots[:] if not parsed_slots else parsed_slots
        trace.required_slots = required

        frontier: List[Tuple[float, int, str, BranchState]] = []
        self._frontier_counter = 0
        self._global_path = []

        # stratified root frontier initialization
        root_bundle = self._plan(task, root_state, required, [], 1, [])
        if not root_bundle.candidates:
            trace.end_ts = time.time()
            return trace

        init_cands = self._stratify(root_bundle.candidates, self.cfg.top_k)
        init_opts = [c.target_page for c in init_cands]
        for i, cand in enumerate(init_cands):
            b = BranchState(
                task_id=task.task_id,
                app=task.app,
                branch_id=f"{task.task_id}_root_{i}",
                parent_id=None,
                depth=0,
                next_step=1,
                current_page=root_state.get("page_id", ""),
                required_slots=required,
                path_actions=[],
                candidate_page_options=init_opts,
                pending_candidate=cand,
                pending_cache_hit=root_bundle.cache_hit,
                status=BranchStatus.ACTIVE,
                evidence_context=[],
                selected_pages=[root_state.get("page_id", "")],
                slot_hits=[],
            )
            b.score = self._score_branch(cand, b, required, b.depth)
            self._push_frontier(frontier, b)

        while frontier and len(trace.steps) < self.cfg.max_steps:
            branch = self._pop_frontier(frontier)
            if branch is None or branch.status != BranchStatus.ACTIVE or branch.pending_candidate is None:
                continue
            if branch.next_step > self.cfg.max_steps:
                continue

            if not self._restore_to_path(branch.path_actions):
                continue

            current_state = self.explorer.get_state()
            branch.current_page = current_state.get("page_id", branch.current_page)

            aligned = [
                e for e in branch.evidence_context
                if e.confidence >= self.cfg.injection_confidence and e.type != EvidenceType.AVOID_HINT
            ]
            aligned_count = len(aligned)

            rec, applied_slots, action_executed = self._run_action(
                task,
                branch,
                branch.pending_candidate,
                branch.next_step,
                list(branch.candidate_page_options),
                aligned_count,
                required,
                executed_after_skip=False,
            )
            branch.next_step += 1
            trace.steps.append(rec)

            trace.explore_pages_count += 1 + int(rec.from_page != rec.to_page)
            trace.rollback_count += int(rec.rollback)
            trace.wait_count += int(rec.planned_wait)
            trace.suppress_count += int(rec.suppress)
            branch.slot_hits.append(rec.slot_complete)
            branch.selected_pages.append(rec.to_page)
            branch.evidence_context.extend(rec.evidences)
            branch.filled_slots = list(rec.applied_slots)
            trace.final_filled_slots = list(rec.applied_slots)

            if set(branch.filled_slots) >= set(required):
                trace.success = True

            if action_executed and not rec.suppress and not rec.rollback and not rec.planned_wait:
                if rec.planned_wait:
                    pass
                else:
                    self._global_path = list(branch.path_actions)
                    branch.path_actions.append(rec.actual_action if rec.actual_action.get("type") != "SUPPRESS" else branch.pending_candidate.action)
                    self._global_path = list(branch.path_actions)

            branch.current_page = self.explorer.get_current_page()

            if (not rec.rollback and not rec.suppress and self._should_aggressive_skip(branch, rec)):
                skip_to = branch.next_step + 1
                if skip_to <= self.cfg.max_steps and branch.depth + 1 < self.cfg.max_depth:
                    skip_bundle = self._plan(task, self.explorer.get_state(), required, branch.filled_slots, skip_to, branch.evidence_context)
                    skip_cands = self._stratify(skip_bundle.candidates, self.cfg.top_k)
                    if skip_cands:
                        skip = skip_cands[0]
                        skip_plan_opts = [c.target_page for c in skip_cands]
                        rec2, applied_slots2, _ = self._run_action(
                            task,
                            branch,
                            skip,
                            skip_to,
                            skip_plan_opts,
                            aligned_count,
                            required,
                            executed_after_skip=True,
                        )
                        rec2.aggressive_skip = True
                        rec2.skipped_to_step = skip_to
                        rec2.executed_after_skip = True
                        trace.steps.append(rec2)
                        trace.aggressive_skip_events.append(
                            {
                                "branch_id": branch.branch_id,
                                "from_step": rec.task_step,
                                "to_step": skip_to,
                                "confidence": rec.confidence,
                                "page": rec.to_page,
                                "success": bool(rec2.action_executed and not rec2.rollback and not rec2.suppress),
                            }
                        )
                        trace.explore_pages_count += 1 + int(rec2.from_page != rec2.to_page)
                        trace.rollback_count += int(rec2.rollback)
                        trace.wait_count += int(rec2.planned_wait)
                        trace.suppress_count += int(rec2.suppress)
                        branch.slot_hits.append(rec2.slot_complete)
                        branch.selected_pages.append(rec2.to_page)
                        branch.evidence_context.extend(rec2.evidences)
                        branch.filled_slots = list(applied_slots2)
                        trace.final_filled_slots = list(applied_slots2)
                        branch.next_step = skip_to + 1
                        if set(branch.filled_slots) >= set(required):
                            trace.success = True

            branch.depth += 1

            if trace.success:
                branch.status = BranchStatus.DONE
            if rec.rollback or rec.suppress:
                branch.status = BranchStatus.BLOCKED
            if branch.status != BranchStatus.ACTIVE:
                continue

            if self._can_expand(branch, rec) and branch.depth < self.cfg.max_steps:
                new_state = self.explorer.get_state()
                new_injected = [
                    e for e in branch.evidence_context
                    if e.confidence >= self.cfg.injection_confidence and e.type != EvidenceType.AVOID_HINT
                ]
                next_bundle = self._plan(task, new_state, required, branch.filled_slots, branch.next_step, new_injected)
                cand_list = self._stratify(next_bundle.candidates, self.cfg.top_k)
                if cand_list:
                    cand_pages = [c.target_page for c in cand_list]
                    for c in cand_list:
                        child_score = self._score_branch(c, branch, required, branch.depth)
                        child = self._clone_child(
                            parent=branch,
                            required_slots=required,
                            candidate=c,
                            candidate_pages=cand_pages,
                            score=child_score,
                            depth=branch.depth,
                            next_step=branch.next_step,
                            selected_pages=branch.selected_pages,
                        )
                        child.pending_cache_hit = next_bundle.cache_hit
                        self._push_frontier(frontier, child)

            if len(frontier) > self.cfg.max_frontier:
                frontier = frontier[: self.cfg.max_frontier]
                heapq.heapify(frontier)

        trace.end_ts = time.time()
        if trace.success and trace.baseline_status == "failure":
            trace.rescued = True
        if (not trace.success) and trace.baseline_status == "success":
            trace.broken = True

        self._collect_branch_records(trace)
        return trace

    def run(self) -> List[TaskTrace]:
        traces: List[TaskTrace] = []
        for task in self.tasks:
            traces.append(self.run_task(task))
        self.cache.save()
        return traces


def summarize(all_traces: List[TaskTrace]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    agg = {
        "total_tasks": len(all_traces),
        "success_count": sum(1 for t in all_traces if t.success),
        "rescued_count": sum(1 for t in all_traces if t.rescued),
        "broken_count": sum(1 for t in all_traces if t.broken),
        "avg_steps": statistics.mean([max(1, t.step_count) for t in all_traces]) if all_traces else 0.0,
    }
    agg["success_rate"] = agg["success_count"] / max(1, agg["total_tasks"])

    app_stats: Dict[str, Dict[str, Any]] = {}
    for t in all_traces:
        s = app_stats.setdefault(t.app, {
            "total": 0,
            "success": 0,
            "steps": [],
            "skip_count": 0,
            "skip_success": 0,
            "slot_steps": 0,
            "slot_hits": 0,
            "rollback": 0,
            "wait": 0,
            "suppress": 0,
        })
        s["total"] += 1
        s["success"] += int(t.success)
        s["steps"].append(t.step_count)
        s["skip_count"] += len(t.aggressive_skip_events)
        s["skip_success"] += sum(1 for e in t.aggressive_skip_events if e.get("success", False))
        s["rollback"] += t.rollback_count
        s["wait"] += t.wait_count
        s["suppress"] += t.suppress_count
        for st in t.steps:
            s["slot_steps"] += 1
            s["slot_hits"] += int(st.slot_complete)

    for s in app_stats.values():
        s["success_rate"] = s["success"] / max(1, s["total"])
        s["avg_steps"] = statistics.mean(s["steps"]) if s["steps"] else 0.0
        s["skip_success_rate"] = s["skip_success"] / max(1, s["skip_count"])
        s["slot_coverage"] = s["slot_hits"] / max(1, s["slot_steps"])

    mode_stats: Dict[str, Dict[str, Any]] = {}
    for t in all_traces:
        s = mode_stats.setdefault(t.mode, {"total": 0, "success": 0, "steps": []})
        s["total"] += 1
        s["success"] += int(t.success)
        s["steps"].append(t.step_count)
    for s in mode_stats.values():
        s["success_rate"] = s["success"] / max(1, s["total"])
        s["avg_steps"] = statistics.mean(s["steps"]) if s["steps"] else 0.0

    return agg, app_stats, mode_stats


def write_outputs(traces: List[TaskTrace], cfg: ExperimentConfig, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    trace_path = out_dir / f"traces_{ts}.json"
    csv_path = out_dir / f"summary_{ts}.csv"
    report_path = out_dir / f"report_{ts}.md"
    chart_dir = out_dir / f"charts_{ts}"
    chart_dir.mkdir(parents=True, exist_ok=True)

    trace_path.write_text(json.dumps([to_jsonable(t) for t in traces], ensure_ascii=False, indent=2), encoding="utf-8")

    agg, app_stats, mode_stats = summarize(traces)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task_id", "app", "mode", "baseline_status", "success", "step_count",
                "rescue", "broken", "rollbacks", "wait", "suppress",
                "aggressive_skips", "aggressive_success", "slot_coverage", "slot_coverage_success",
                "final_filled", "required_slots",
            ],
        )
        w.writeheader()
        for t in traces:
            skip_total = len(t.aggressive_skip_events)
            skip_ok = sum(1 for e in t.aggressive_skip_events if e.get("success", False))
            slot_steps = len(t.steps) if t.steps else 1
            slot_hits = sum(1 for s in t.steps if s.slot_complete)
            w.writerow(
                {
                    "task_id": t.task_id,
                    "app": t.app,
                    "mode": t.mode,
                    "baseline_status": t.baseline_status,
                    "success": int(t.success),
                    "step_count": t.step_count,
                    "rescue": int(t.rescued),
                    "broken": int(t.broken),
                    "rollbacks": t.rollback_count,
                    "wait": t.wait_count,
                    "suppress": t.suppress_count,
                    "aggressive_skips": skip_total,
                    "aggressive_success": skip_ok,
                    "slot_coverage": f"{slot_hits/slot_steps:.3f}",
                    "slot_coverage_success": f"{len(set(t.final_filled_slots))/max(1, len(t.required_slots)):.3f}",
                    "final_filled": ",".join(t.final_filled_slots),
                    "required_slots": ",".join(t.required_slots),
                }
            )

    charts = {}
    if plt is not None:
        apps = list(app_stats.keys())
        success = [app_stats[a]["success_rate"] * 100.0 for a in apps]
        steps = [app_stats[a]["avg_steps"] for a in apps]
        skip = [app_stats[a]["skip_success_rate"] * 100.0 for a in apps]
        slot_cov = [app_stats[a]["slot_coverage"] * 100.0 for a in apps]
        rollback = [app_stats[a]["rollback"] for a in apps]
        wait = [app_stats[a]["wait"] for a in apps]
        suppress = [app_stats[a]["suppress"] for a in apps]

        def _bar(x, y, title, ylab, path, fmt="%.2f"):
            plt.figure(figsize=(8, 4))
            plt.bar(x, y)
            plt.title(title)
            plt.ylabel(ylab)
            plt.xticks(rotation=25)
            for i, v in enumerate(y):
                plt.text(i, v + (0.02 * max(1.0, max(y))), fmt % v, ha="center", fontsize=8)
            plt.tight_layout()
            plt.savefig(path)
            plt.close()

        p1 = chart_dir / "success_rate.png"
        p2 = chart_dir / "avg_steps.png"
        p3 = chart_dir / "skip_success_rate.png"
        p4 = chart_dir / "slot_coverage.png"
        p5 = chart_dir / "rollback_wait_suppress.png"

        _bar(apps, success, "按 App 成功率（%）", "success %", p1)
        _bar(apps, steps, "按 App 平均步骤数", "steps", p2)
        _bar(apps, skip, "按 App Aggressive Skip 成功率（%）", "skip success %", p3)
        _bar(apps, slot_cov, "按 App Slot 覆盖率（%）", "slot coverage %", p4)

        plt.figure(figsize=(8, 4))
        x = list(range(len(apps)))
        w = 0.25
        plt.bar([i - w for i in x], rollback, width=w, label="rollback")
        plt.bar(x, wait, width=w, label="WAIT")
        plt.bar([i + w for i in x], suppress, width=w, label="suppress")
        plt.title("按 App 异常控制次数")
        plt.ylabel("count")
        plt.xticks(x, apps, rotation=25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(p5)
        plt.close()

        charts = {
            "success": str(p1),
            "steps": str(p2),
            "skip": str(p3),
            "slot": str(p4),
            "ctrl": str(p5),
        }

    lines = []
    lines.append("# MobileExplorer 20-task 扩展实验（slot-complete best-first + aggressive skip）")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 任务数：{agg['total_tasks']}")
    lines.append(f"- 成功率：{agg['success_rate']:.2%}")
    lines.append(f"- 平均步骤数：{agg['avg_steps']:.2f}")
    lines.append(f"- rescued: {agg['rescued_count']}")
    lines.append(f"- broken: {agg['broken_count']}")
    lines.append("")

    lines.append("## 1. 按应用统计")
    lines.append("| app | 总任务 | 成功 | 成功率 | 平均步数 | Aggressive Skip 成功率 | Slot 命中率 | rollback | WAIT | suppress |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for app, s in app_stats.items():
        lines.append(
            f"| {app} | {s['total']} | {s['success']} | {s['success_rate']:.2%} | "
            f"{s['avg_steps']:.2f} | {s['skip_success_rate']:.2%} | {s['slot_coverage']:.2%} | "
            f"{s['rollback']} | {s['wait']} | {s['suppress']} |"
        )

    lines.append("")
    lines.append("## 2. 按任务模式统计")
    lines.append("| mode | 总任务 | 成功 | 成功率 | 平均步数 |")
    lines.append("|---|---:|---:|---:|---:|")
    for mode, s in mode_stats.items():
        lines.append(f"| {mode} | {s['total']} | {s['success']} | {s['success_rate']:.2%} | {s['avg_steps']:.2f} |")

    lines.append("")
    lines.append("## 3. 文件")
    lines.append(f"- JSON trace: `{trace_path}`")
    lines.append(f"- CSV summary: `{csv_path}`")
    lines.append(f"- Markdown report: `{report_path}`")

    if charts:
        lines.append("\n## 4. 图表")
        lines.append(f"![success rate]({charts['success']})")
        lines.append(f"![avg steps]({charts['steps']})")
        lines.append(f"![skip success]({charts['skip']})")
        lines.append(f"![slot coverage]({charts['slot']})")
        lines.append(f"![controls]({charts['ctrl']})")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "trace_path": str(trace_path),
        "summary_csv": str(csv_path),
        "report_md": str(report_path),
        "charts_dir": str(chart_dir),
        "overall": agg,
        "app_stats": app_stats,
        "mode_stats": mode_stats,
        "charts": charts,
    }


def build_runner(dry_run: bool, output_dir: Path, max_depth: int, max_steps: int, cache_file: Path, seed: int):
    cfg = ExperimentConfig(
        max_depth=max_depth,
        max_steps=max_steps,
        output_dir=output_dir,
        cache_file=cache_file,
    )

    if dry_run:
        planner = MockPlanner(seed=seed)
        explorer = MockExplorer(seed=seed)
    else:
        raise RuntimeError("Please replace MockPlanner/MockExplorer with your real V12 planner and explorer in this script.")

    tasks = build_default_tasks()
    return ExperimentRunner(planner=planner, explorer=explorer, cfg=cfg, tasks=tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="run in mock mode")
    parser.add_argument("--output-dir", default="mobile_explorer_20task_outputs")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--cache-file", default="exploration_evidence_cache.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg_out = Path(args.output_dir)
    runner = build_runner(
        dry_run=args.dry_run,
        output_dir=cfg_out,
        max_depth=args.max_depth,
        max_steps=args.max_steps,
        cache_file=Path(args.cache_file),
        seed=args.seed,
    )
    traces = runner.run()
    outputs = write_outputs(traces, runner.cfg, cfg_out)
    print("Saved:")
    print(outputs["trace_path"])
    print(outputs["summary_csv"])
    print(outputs["report_md"])
    print(outputs["charts_dir"])


if __name__ == "__main__":
    main()
