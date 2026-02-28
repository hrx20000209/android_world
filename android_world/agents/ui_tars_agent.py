from absl import flags
import io
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Dict, Optional
from android_world.agents import base_agent
from android_world.env import interface, json_action
from android_world.agents.ui_tars import UITARS15Agent
from android_world.agents.ui_tars_utils import (
    execute_ui_tars_response,
    take_screenshot_adb,
)


# UI_TARS_MAX_LOOP = 100
logger = logging.getLogger(__name__)
UI_TARS_DEFAULT_MAX_STEPS = 20

def _extract_task_info(goal: str) -> Optional[Dict]:
    """从goal中提取对应的任务信息。

    Args:
        goal: 输入的任务目标字符串

    Returns:
        匹配到的任务元数据字典,如果未匹配则返回None
    """
    # 读取任务元数据
    task_path = os.path.join(os.path.dirname(__file__), "../task_metadata.json")
    with open(task_path, "r") as file:
        task_metadata = json.load(file)

    # 对每个任务模板进行精确匹配
    for task in task_metadata:
        template = task["task_template"]

        # 将模板转换为正则表达式模式
        # 1. 转义正则表达式特殊字符
        pattern = re.escape(template)

        # 2. 将{param}形式的参数替换为通配符
        pattern = re.sub(r'\\\{[^}]*\\\}', '(.*?)', pattern)

        # 3. 尝试匹配
        if re.match(pattern, goal):
            return task

    # 如果精确匹配失败，使用相似度匹配
    def normalize_text(text):
        # 移除标点符号和多余空格
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return ' '.join(text.split())

    def calculate_similarity(text1, text2):
        # 使用词集合的Jaccard相似度
        words1 = set(normalize_text(text1).split())
        words2 = set(normalize_text(text2).split())
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0

    # 寻找相似度最高的模板
    best_match = None
    highest_similarity = 0.6  # 设置一个最低相似度阈值

    for task in task_metadata:
        template = task["task_template"]
        # 将模板中的参数占位符替换为空格
        template = re.sub(r'\{[^}]*\}', '', template)
        similarity = calculate_similarity(goal, template)

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = task

    return best_match


def extract_task_info(goal: str) -> Optional[Dict]:
    exp_name = os.environ.get("EXP_NAME")
    process_id = os.environ.get("PROCESS_ID")
    task_hint_path = (
        Path(f"./task_{exp_name}_{process_id}.txt")
        if exp_name and process_id
        else None
    )

    # Prefer explicit task hint file when provided by launcher.
    if task_hint_path and task_hint_path.exists():
        task_name = task_hint_path.read_text().strip()
    else:
        if task_hint_path:
            logger.warning(
                f"Task hint file not found: {task_hint_path}. Fallback to goal template matching."
            )
        return _extract_task_info(goal)

    task_path = os.path.join(os.path.dirname(__file__), "../task_metadata.json")
    with open(task_path, "r") as file:
        task_metadata = json.load(file)

    for task in task_metadata:
        if task["task_name"] == task_name:
            return task
    return _extract_task_info(goal)


class UiTarsAgent(base_agent.EnvironmentInteractingAgent):

    def __init__(
            self,
            env: interface.AsyncEnv,
            name: str = 'UiTarsAgent',
            verbose: bool = False,
    ):
        """Initializes a UiTarsAgent.

        Args:
          env: The environment.
          name: The agent name.
          verbose: True if the grounder should produce verbose updates.
        """
        super().__init__(env, name)
        self._verbose = verbose
        self.serial_port = flags.FLAGS.console_port
        self.task_name = flags.FLAGS.tasks[0] if flags.FLAGS.tasks else "unknown_task"
        self.exp_name = os.environ.get("EXP_NAME") or "default_exp"
        self.process_id = os.environ.get("PROCESS_ID") or "0"
        self._serial = f"emulator-{self.serial_port}"
        self._image_resize_factor = 1.0
        self._ui_tars_agent: UITARS15Agent | None = None
        self._ui_tars_step_index = 0
        self._current_goal = ""
        self._img_save_dir: Path | None = None

    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home)
        self._ui_tars_agent = None
        self._ui_tars_step_index = 0
        self._current_goal = ""
        self._img_save_dir = None

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        """See base class."""
        task_info = extract_task_info(goal)
        if task_info and task_info.get("task_name"):
            task_name_for_dir = task_info["task_name"]
        else:
            task_name_for_dir = self.task_name
        max_allowed_steps = min(
            self._max_steps if self._max_steps is not None else UI_TARS_DEFAULT_MAX_STEPS,
            UI_TARS_DEFAULT_MAX_STEPS,
        )

        if self._ui_tars_agent is None or self._current_goal != goal:
            meta_info_dir = Path(f"ui_tars/{self.exp_name}_{self.process_id}/{task_name_for_dir}")
            meta_info_dir.mkdir(parents=True, exist_ok=True)
            self._img_save_dir = meta_info_dir / "images"
            self._img_save_dir.mkdir(parents=True, exist_ok=True)
            self._ui_tars_agent = UITARS15Agent(
                use_thinking=False,
                meta_info_dir=str(meta_info_dir),
            )
            self._ui_tars_step_index = 0
            self._current_goal = goal

        if self._ui_tars_step_index >= max_allowed_steps:
            step_data = {
                "ui_tars_finished": False,
                "ui_tars_content": "",
                "ui_tars_error_info": f"Reached max steps ({max_allowed_steps}).",
                "ui_tars_step_index": self._ui_tars_step_index,
            }
            return base_agent.AgentInteractionResult(True, step_data)

        step_start = time.perf_counter()
        try:
            screenshot = take_screenshot_adb(serial=self._serial)
            screenshot = screenshot.resize(
                (
                    round(screenshot.width * self._image_resize_factor),
                    round(screenshot.height * self._image_resize_factor),
                )
            )
            screenshot.save(f"current_screenshot-{self._serial}.png")
            if self._img_save_dir is not None:
                screenshot.save(self._img_save_dir / f"step_{self._ui_tars_step_index}.png")

            screenshot_bytes_io = io.BytesIO()
            screenshot.save(screenshot_bytes_io, format="PNG")
            screenshot_bytes = screenshot_bytes_io.getvalue()

            if self._ui_tars_agent is None:
                raise RuntimeError("ui_tars core agent is not initialized")
            self._ui_tars_agent.history_images.append(screenshot_bytes)
            obs = {"screenshot": screenshot_bytes, "accessibility_tree": ""}
            parsed_response = self._ui_tars_agent._predict_uitars15(goal, obs)
            execution_result = execute_ui_tars_response(
                parsed_response,
                self._image_resize_factor,
                self._serial,
            )

            self._ui_tars_step_index += 1
            done = bool(execution_result.finished)
            if not done and self._ui_tars_step_index >= max_allowed_steps:
                done = True

            if done and execution_result.content:
                action_details = {'action_type': 'answer', 'text': execution_result.content}
                self.env.execute_action(json_action.JSONAction(**action_details))

            step_data = {
                "ui_tars_finished": bool(execution_result.finished),
                "ui_tars_content": execution_result.content,
                "ui_tars_error_info": execution_result.error_info,
                "ui_tars_step_index": self._ui_tars_step_index,
                "step_latency_sec": max(0.0, float(time.perf_counter() - step_start)),
            }
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.exception("UiTarsAgent step failed: %s", e)
            self._ui_tars_step_index += 1
            done = self._ui_tars_step_index >= max_allowed_steps
            step_data = {
                "ui_tars_finished": False,
                "ui_tars_content": "",
                "ui_tars_error_info": repr(e),
                "ui_tars_step_index": self._ui_tars_step_index,
                "step_latency_sec": max(0.0, float(time.perf_counter() - step_start)),
            }
        return base_agent.AgentInteractionResult(
            done,
            step_data,
        )

    def save_summary(self, path: str = "androidworld_exec_profile_summary.json") -> None:
        """Keep interface compatibility with runners that always call save_summary()."""
        logger.info("UiTarsAgent does not collect profiling summary. skip writing %s", path)
