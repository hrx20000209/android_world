import json
import os
import imagehash
from collections import defaultdict

from PIL import Image
from android_world.agents.t3a import T3A


# ================= pHash functions =================
def phash(img_path):
    img = Image.open(img_path).convert("L")
    return imagehash.phash(img)


def check_same_image(img1, img2, threshold=8):
    h1 = phash(img1)
    h2 = phash(img2)
    diff = h1 - h2
    print(f"pHash diff = {diff}")
    return diff <= threshold


# ================= Profiling Agent =================
class T3AExecProfilingAgent(T3A):
    """
    Execute tasks normally + profile screen dynamics.
    Screenshot source is the same as MAIUIAgent: state.pixels
    """

    def __init__(self, env, llm,
                 name="T3A-Exec-Profiling",
                 log_path="androidworld_exec_profile.jsonl"):
        super().__init__(env, llm, name)

        self.log_path = log_path
        self.step_id = 0

        self.action_counter = defaultdict(int)
        self.screen_change_counter = {
            "large_change": 0,
            "small_change": 0
        }

        os.makedirs("tmp_prof", exist_ok=True)

    # --------------------------
    # action_type 提取
    # --------------------------
    def _extract_action_type(self, action_output):
        try:
            json_part = action_output.split("Action:", 1)[1].strip()
            action_json = json.loads(json_part)
            return action_json.get("action_type", "unknown")
        except Exception:
            return "parse_error"

    # --------------------------
    # 用 MAI 同源 screenshot
    # --------------------------
    def _save_screen(self, img_array, tag):
        path = f"tmp_prof/step{self.step_id}_{tag}.png"
        Image.fromarray(img_array).save(path)
        return path

    # --------------------------
    # step()
    # --------------------------
    def step(self, goal):

        # 先给这个 step 分配 id，保证文件名对应
        self.step_id += 1

        # before 截图：直接用 state.pixels（和 MAIUIAgent 一样）
        state_before = self.get_post_transition_state()
        before_path = self._save_screen(state_before.pixels, "before")

        # 正常执行
        result = super().step(goal)

        # result.data 才是 step_data
        step_data = getattr(result, "data", None)
        if not isinstance(step_data, dict):
            return result

        # after 截图
        state_after = self.get_post_transition_state()
        after_path = self._save_screen(state_after.pixels, "after")

        action_output = step_data.get("action_output", "")

        # pHash 比较
        same = check_same_image(before_path, after_path)
        large_change = not same

        if large_change:
            self.screen_change_counter["large_change"] += 1
        else:
            self.screen_change_counter["small_change"] += 1

        # action_type 统计
        action_type = self._extract_action_type(action_output)
        self.action_counter[action_type] += 1

        # 写日志
        record = {
            "step": self.step_id,
            "action_type": action_type,
            "same_screen": same,
            "large_change": large_change,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"[PROFILE] Step {self.step_id} | {action_type} | same_screen={same}")

        return result

    # --------------------------
    # 汇总
    # --------------------------
    def save_summary(self, path="androidworld_exec_profile_summary.json"):
        summary = {
            "total_steps": self.step_id,
            "action_distribution": dict(self.action_counter),
            "screen_change_stats": self.screen_change_counter
        }

        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[PROFILE] Summary saved to {path}")
