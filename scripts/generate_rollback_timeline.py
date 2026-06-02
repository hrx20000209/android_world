#!/usr/bin/env python3
"""Generate horizontal rollback timeline images for level2/failure cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _safe_name(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-"}:
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    return "".join(keep)[:80] or "task"


def _event_interesting(event: dict[str, Any]) -> bool:
    return bool(event.get("level2_triggered")) or not bool(event.get("success", True))


def _candidate_images(task_dir: Path, event: dict[str, Any]) -> list[tuple[str, Path]]:
    step = int(event.get("step") or event.get("source_step") or 0)
    branch = int(event.get("branch_id") or event.get("candidate_id") or 0)
    pairs: list[tuple[str, Path]] = []
    names = [
        ("root", f"explore_step_{step:02d}_root.png"),
        ("depth1", f"explore_step_{step:02d}_b{branch}_d1.png"),
        ("depth2", f"explore_step_{step:02d}_b{branch}_d2.png"),
    ]
    for label, name in names:
        path = task_dir / name
        if path.exists():
            pairs.append((label, path))
    rollback_patterns = [
        f"*step_{step:02d}*rollback*.png",
        f"*rollback*step_{step:02d}*.png",
        f"rollback_failed_step_{step:02d}*.png",
        f"*b{branch}*rollback*.png",
    ]
    seen = {p for _, p in pairs}
    for pattern in rollback_patterns:
        for path in sorted(task_dir.glob(pattern)):
            if path not in seen:
                pairs.append(("rollback/final", path))
                seen.add(path)
    explicit = event.get("screenshot_path") or event.get("screenshot")
    if explicit:
        path = Path(str(explicit))
        if path.exists() and path not in seen:
            pairs.append(("event screenshot", path))
    return pairs[:6]


def _draw_timeline(pairs: list[tuple[str, Path]], out_path: Path, event: dict[str, Any]) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return False
    if not pairs:
        return False
    thumbs = []
    for label, path in pairs:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        img.thumbnail((220, 390))
        canvas = Image.new("RGB", (240, 440), "white")
        canvas.paste(img, ((240 - img.width) // 2, 30))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("Arial.ttf", 13)
        except Exception:
            font = ImageFont.load_default()
        draw.text((8, 6), label[:28], fill="black", font=font)
        draw.text((8, 410), f"success={bool(event.get('success', True))}", fill="black", font=font)
        thumbs.append(canvas)
    if not thumbs:
        return False
    width = sum(t.width for t in thumbs) + 40 * (len(thumbs) - 1)
    final = Image.new("RGB", (width, 440), "white")
    draw = ImageDraw.Draw(final)
    x = 0
    for idx, thumb in enumerate(thumbs):
        final.paste(thumb, (x, 0))
        x += thumb.width
        if idx < len(thumbs) - 1:
            draw.text((x + 12, 205), "->", fill="black")
            x += 40
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(out_path)
    return True


def generate(trace_root: Path, output_dir: Path) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_lines = ["# Rollback Timeline Cases", ""]
    count_events = 0
    count_images = 0
    for rb_file in sorted(trace_root.rglob("rollback_events.jsonl")):
        task_dir = rb_file.parent
        for event in _read_jsonl(rb_file):
            if not _event_interesting(event):
                continue
            count_events += 1
            task = _safe_name(str(event.get("task_id") or task_dir.name))
            step = int(event.get("step") or 0)
            branch = int(event.get("branch_id") or 0)
            pairs = _candidate_images(task_dir, event)
            out_img = output_dir / f"rollback_timeline_{task}_step{step:02d}_branch{branch:02d}.png"
            made = _draw_timeline(pairs, out_img, event)
            if made:
                count_images += 1
            case_md = output_dir / f"rollback_case_{task}_step{step:02d}_branch{branch:02d}.md"
            lines = [
                f"# Rollback case: {task}",
                "",
                f"- step: `{step}`",
                f"- branch: `{branch}`",
                f"- rollback level: `{event.get('rollback_level') or event.get('level')}`",
                f"- level2 triggered: `{event.get('level2_triggered')}`",
                f"- level2 trigger reason: `{event.get('level2_trigger_reason')}`",
                f"- success: `{event.get('success')}`",
                f"- failure reasons: `{event.get('failure_reasons') or event.get('failure_reason')}`",
                f"- evidence discarded: `{event.get('branch_evidence_discarded')}`",
                f"- prompt history contaminated: `{event.get('prompt_history_contaminated')}`",
                "",
            ]
            if made:
                lines.append(f"![timeline]({out_img.resolve()})")
            else:
                lines.append("No timeline image generated; screenshot inputs were missing or Pillow was unavailable.")
            case_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
            md_lines.append(f"- `{case_md.name}` image=`{out_img.name if made else ''}` success=`{event.get('success')}`")
    (output_dir / "rollback_timeline_index.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return count_events, count_images


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_root", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    events, images = generate(Path(args.trace_root).expanduser().resolve(), Path(args.output_dir).expanduser().resolve())
    print(json.dumps({"events": events, "images": images}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
