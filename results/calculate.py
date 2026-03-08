import re
import math
import pandas as pd


def load_androidworld_result(file_path: str) -> pd.DataFrame:
    """
    Parse AndroidWorld result txt and keep only:
    - task
    - mean_success_rate
    - mean_episode_length

    Rules:
    - stop parsing when reaching '========= Average ========='
    - NaN in success/steps is treated as 0
    """

    pattern = re.compile(
        r"^(\S+)\s+"                       # task
        r"(\d+)\s+"                       # task_num
        r"([0-9.]+|NaN)\s+"               # num_complete_trials
        r"([0-9.]+|NaN)\s+"               # mean_success_rate
        r"([0-9.]+|NaN)\s+"               # mean_episode_length
        r"([0-9.]+|NaN)\s+"               # total_runtime_s
        r"([0-9.]+|NaN)\s+"               # mean_step_latency_s
        r"([0-9.]+|NaN)$"                 # num_fail_trials
    )

    rows = []
    in_task_table = False

    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if not line.strip():
                continue

            if line.strip() == "task":
                in_task_table = True
                continue

            if line.startswith("========= Average ========="):
                break

            if not in_task_table:
                continue

            m = pattern.match(line.strip())
            if not m:
                continue

            task = m.group(1)

            def parse_float(x: str) -> float:
                return float("nan") if x == "NaN" else float(x)

            mean_success_rate = parse_float(m.group(4))
            mean_episode_length = parse_float(m.group(5))

            # NaN -> 0
            if math.isnan(mean_success_rate):
                mean_success_rate = 0.0
            if math.isnan(mean_episode_length):
                mean_episode_length = 0.0

            rows.append(
                {
                    "task": task,
                    "mean_success_rate": mean_success_rate,
                    "mean_episode_length": mean_episode_length,
                }
            )

    return pd.DataFrame(rows)


def compute_metrics(file_path: str) -> None:
    df = load_androidworld_result(file_path)

    total_tasks = len(df)
    success_df = df[df["mean_success_rate"] == 1.0]

    success_rate = 100.0 * len(success_df) / total_tasks if total_tasks > 0 else 0.0
    avg_steps_all = df["mean_episode_length"].mean() if total_tasks > 0 else 0.0
    avg_steps_success = success_df["mean_episode_length"].mean() if len(success_df) > 0 else 0.0

    print(f"File: {file_path}")
    print(f"Total tasks: {total_tasks}")
    print(f"Successful tasks: {len(success_df)}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Average steps (all tasks): {avg_steps_all:.2f}")
    print(f"Average steps (successful tasks): {avg_steps_success:.2f}")


if __name__ == "__main__":
    file_path = input("Enter result file path: ").strip()
    compute_metrics(file_path)