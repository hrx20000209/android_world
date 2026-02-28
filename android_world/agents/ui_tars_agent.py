from absl import flags
import json
import logging
import os
from pathlib import Path
import re
from typing import Dict, Optional
from android_world.agents import base_agent
from android_world.env import interface, json_action
from android_world.agents.ui_tars import ui_tars15_work_loop


# UI_TARS_MAX_LOOP = 100
logger = logging.getLogger(__name__)

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

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        """See base class."""
        task_info = extract_task_info(goal)
        if task_info and task_info.get("task_name"):
            task_name_for_dir = task_info["task_name"]
        else:
            task_name_for_dir = self.task_name

        # get_ui_tars_response = init_get_ui_tars_response(
        #     base_url='http://127.0.0.1:8888',
        #     api_key=''
        # )
        def run_task(task_description: str):
            meta_info_dir = Path(f"ui_tars/{self.exp_name}_{self.process_id}/{task_name_for_dir}")
            meta_info_dir.mkdir(parents=True, exist_ok=True)
            img_save_dir = meta_info_dir / "images"
            img_save_dir.mkdir(parents=True, exist_ok=True)
            max_loop_time = self._max_steps if self._max_steps is not None else 50
            return ui_tars15_work_loop(
                instruction=task_description,
                serial=f"emulator-{self.serial_port}",
                max_loop_time=max_loop_time,
                img_save_dir=img_save_dir,
                meta_info_dir=meta_info_dir,
            )

        #         instruction = f'''
        # You are asked to complete the following task on the current device: {task_description}.
        # Note:
        # 1. You should use `open_app(app_name)` to open the specific app. For example, use `open_app('contacts')` to start the `contacts` app.
        # '''
        # return ui_tars_work_loop(
        #     local_mode=True,
        #     get_ui_tars_response=get_ui_tars_response,
        #     get_ui_tars_mobile_prompt=get_ui_tars_mobile_prompt_api,
        #     instruction=instruction
        # )

        # 还需要加上这个功能
        # if exec_res.answer:
        #   action_details = {'action_type': 'answer', 'text': exec_res.answer}
        #   self.env.execute_action(json_action.JSONAction(**action_details))
        try:
            result = run_task(goal)
            print(result)
            if result.content:
                action_details = {'action_type': 'answer', 'text': result.content}
                self.env.execute_action(json_action.JSONAction(**action_details))
            step_data = {
                "ui_tars_finished": bool(result.finished),
                "ui_tars_content": result.content,
                "ui_tars_error_info": result.error_info,
            }
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.exception("UiTarsAgent step failed: %s", e)
            step_data = {
                "ui_tars_finished": False,
                "ui_tars_content": "",
                "ui_tars_error_info": repr(e),
            }
        done = True
        return base_agent.AgentInteractionResult(
            done,
            step_data,
        )

    def save_summary(self, path: str = "androidworld_exec_profile_summary.json") -> None:
        """Keep interface compatibility with runners that always call save_summary()."""
        logger.info("UiTarsAgent does not collect profiling summary. skip writing %s", path)
