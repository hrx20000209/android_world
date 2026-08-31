# 并行 UI Exploration 可行性 Harness：第 1 阶段

本阶段只验证两个独立进程、共享事件协议和关键路径延迟计时。Explorer 目前执行的是可中断的模拟工作，不会调用 adb，也不会产生真实 UI probe 数据。

## 运行 mock 验证

```bash
python scripts/run_parallel_exploration_phase1.py \
  --config configs/parallel_exploration_phase1_mock.json \
  --events-jsonl /tmp/parallel_exploration_phase1_events.jsonl
```

正常情况下，事件应包含：

```text
EXPLORER_READY
INFERENCE_START
FIRST_TOKEN
DECODE_PROGRESS
INFERENCE_END
ABORT
RESTORED
TRIAL_COMPLETE
```

`TRIAL_COMPLETE.payload.critical_path_extension_ms` 由 Process A 从发送 `ABORT` 开始计时，到 Process A 收到 `RESTORED` 或 `RESTORE_FAILED` 为止。它包含 Explorer 恢复耗时和本地进程间通信延迟。

## 运行测试

```bash
python -m pytest -q \
  android_world/parallel_exploration/protocol_test.py \
  android_world/parallel_exploration/processes_test.py
```

## 后端配置

`inference.backend` 支持：

- `mock`：固定、均匀或正态分布的模拟推理时间。
- `http`：OpenAI-compatible SSE streaming endpoint。
- `llamacpp`：本地 `llama-cpp-python`，仅选用时才导入该可选依赖。

HTTP 流中的 `token_count` 当前是收到的非空 SSE 文本 chunk 数，不应解释为 tokenizer 计算出的精确 token 数。mock 与 llama.cpp 模式同样只使用流式更新序号。后续如果分析需要精确 token 数，应由具体后端提供 tokenizer 计数，不能用当前字段伪装成精确值。

## 本阶段尚未实现

- adb UI 探测
- candidate ranker
- 三种状态签名
- recovery ladder
- trial runner 和最终分析脚本

这些部分会严格按照任务中的顺序，在本阶段运行确认后继续。
