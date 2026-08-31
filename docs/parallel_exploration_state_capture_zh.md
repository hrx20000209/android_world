# 并行 Exploration 状态采集延迟修复

## 根因

原实现每次状态采集都会执行：

- `adb shell uiautomator dump`：命令内部固定等待 UI idle；持续产生 accessibility event 时会等到约 10 秒后失败。
- `adb exec-out screencap -p`：设备端编码 1080×2400 PNG，再通过 ADB 传输；本机 emulator 的中位数约 1458ms。
- `adb shell content read`：FastA11yService 本身只需要约 14ms，但每次启动 Android `content` 命令使 host 端耗时达到约 1222ms。

## 修复

- Screenshot 使用 AndroidWorld/AndroidEnv 已有的 emulator gRPC `getScreenshot` 长连接。
- FastA11yService 增加 `localabstract:androidworld_fast_a11y` 持久 socket。
- host 只执行一次 `adb forward tcp:8765 localabstract:androidworld_fast_a11y`，后续在同一 TCP 连接上请求树。
- 保留 activity、structural hash、pHash 三种独立签名，不改变相等性定义。

Python 入口：

```python
from android_world.parallel_exploration.state import create_optimized_state_capture

capture = create_optimized_state_capture()
state = capture.capture()
print(state.timings_ms)
```

## 实测

Settings 页面连续 50 次：

| 指标 | Mean | P50 | P95 | Max |
|---|---:|---:|---:|---:|
| 完整三签名状态采集 | 118.9ms | 114.3ms | 167.3ms | 244.8ms |
| Activity dump | 53.7ms | 51.0ms | 95.6ms | 111.6ms |
| A11y socket E2E | 17.7ms | 16.6ms | 26.9ms | 83.2ms |
| A11y device service | 14.1ms | 13.5ms | 24.3ms | 64.9ms |
| gRPC screenshot | 30.5ms | 29.3ms | 38.3ms | 75.3ms |
| pHash | 9.1ms | 9.0ms | 9.3ms | 10.8ms |

旧 AndroidWorld benchmark 中完整状态获取均值约 4.46s；优化后中位数约 114ms，约快 39 倍。

## APK 构建与安装

```bash
python tools/fast_a11y_dumper/build_apk.py
adb -s emulator-5554 install -r tools/fast_a11y_dumper/build/fast-a11y.apk
```

构建脚本在 macOS 上优先使用 Android Studio 自带 JBR，避免系统 JDK 22 在 `-bootclasspath android-36` 组合下的异常慢编译。

如果设备上旧 APK 使用不同签名，Android 会拒绝覆盖安装。确认该测试工具不含需要保留的数据后，先卸载 `com.androidworld.fasta11y` 再安装。卸载会删除该工具 App 自身数据。
