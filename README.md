# Ascend 310B4 Object Detect Model Convert & Qt GUI

本项目是一个基于 **Ascend 310B4 (Orange Pi AIpro)** 硬件平台的目标检测解决方案。项目包含两个核心部分：

1. **ModelConvert**: 转个锤子直接用sample里的模型(将 YOLOv11 模型从 PyTorch (`.pt`) 转换为 ONNX，最终转换为昇腾 NPU 专用的 `.om` (Offline Model) 格式)。
2. **HumanDetect**: 一个基于 Qt6 C++ 的图形化界面程序，通过嵌入式 Python (PyBind11) 调用 NPU 进行实时推理。

## 📋 项目环境与依赖

* **硬件平台**: Orange Pi AIpro (Ascend 310B4)
* **操作系统**: Ubuntu 22.04 (镜像: `opiaipro_ubuntu22.04_desktop_aarch64_20240318.img`)
* **开发框架**:
* **Qt**: Qt 6.x
* **OpenCV**: 4.x (系统默认或自行编译)
* **Python**: Python 3.9 (推荐使用 Miniconda)
* **CANN Toolkit**: 华为昇腾开发套件 (用于 `atc` 模型转换和推理)



## 📂 目录结构

```text
├── HumanDetect/            # Qt C++ 目标检测主程序
│   ├── HumanDetect.pro     # Qt 项目配置文件
│   ├── main.cpp            # 程序入口
│   ├── mainwindow.cpp      # UI 逻辑
│   ├── yolo_npu_wrapper.cpp# C++ 调用 Python 的封装层 (PyBind11)
│   ├── npu_wrapper.py      # Python 端 NPU 推理接口
│   └── ...
└── README.md

```

---

## 🚀 第一部分：模型转换--不需要了 (ModelConvert)

此步骤将 YOLOv11n 模型转换为 NPU 可执行的 `.om` 文件。

### 1. 导出 ONNX

使用 `export.py` 将 `.pt` 权重导出为 ONNX 格式。

```bash
cd ModelConvert
python export.py
# 输出: yolo11n.onnx

```

### 2. 转换为 OM 模型

使用 `convert.py` 脚本调用昇腾 ATC 工具进行转换。

> **注意**: 默认配置针对 `Ascend310B4`，输入尺寸为 `640x640`。

```bash
python convert.py --onnx_path yolo11n.onnx --om_path yolo11n.om

```

*转换脚本核心参数说明 (`convert.py`):*

* `--soc_version=Ascend310B4`: 指定芯片版本。
* `--disable_reuse_memory=1`: 禁用内存复用（解决部分模型转换错误）。
* `input_shape`: `images:1,3,640,640`。

---

## 🖥️ 第二部分：Qt 目标检测程序 (HumanDetect)

该程序使用 C++ 编写界面，通过 PyBind11 调用 Python 脚本 (`npu_wrapper.py`) 来利用 NPU 进行推理。

### 1. 关键配置修改 (重要)

由于代码中包含硬编码的路径，编译运行前请务必检查以下文件：

**A. `HumanDetect.pro**`
检查 Python 路径是否与您的 Conda 安装位置一致：

```qmake
# 确保此路径指向您的 miniconda 或 anaconda 安装目录
PYTHON_HOME = /usr/local/miniconda3

```

**B. `yolo_npu_wrapper.cpp**`
检查 Python 初始化路径：

```cpp
// 必须指向正确的 Python Home 目录
const wchar_t* python_home = L"/usr/local/miniconda3";

// 检查 sys.path 插入的路径，特别是 site-packages 和 lib-dynload
std::vector<std::string> required_paths = {
    "/home/HwHiAiUser/HumanDetect",                       // 修改为您的实际构建/源码路径
    "/usr/local/miniconda3/lib/python3.9/site-packages",
    "/usr/local/miniconda3/lib/python3.9/lib-dynload",    // 修复 numpy 导入错误的关键
    "/usr/local/miniconda3/lib/python3.9"
};

```

### 2. 编译项目

在项目根目录下执行：

```bash
cd HumanDetect
qmake
make -j4

```

### 3. 运行程序

编译成功后，可执行文件会在同级目录或 Release 目录下生成。确保 `npu_wrapper.py` 和转换好的 `.om` 模型文件在正确的位置。

```bash
# 运行编译出的程序
./HumanDetect

```

*注意：`HumanDetect.pro` 已配置构建时自动复制 `npu_wrapper.py` 到输出目录。*

---

## 🛠️ 常见问题 (Troubleshooting)

1. **Python 环境初始化失败**
* 错误现象：程序启动崩溃，提示 Python 环境错误。
* 解决：检查 `yolo_npu_wrapper.cpp` 中的 `python_home` 路径是否正确，确保该路径下有 `bin/python3.9`。


2. **ImportError: numpy.core.multiarray failed to import**
* 原因：C++ 嵌入 Python 时缺少 `lib-dynload` 路径。
* 解决：代码中已通过添加 `/usr/local/miniconda3/lib/python3.9/lib-dynload` 到 `sys.path` 修复此问题，请确保该路径真实存在。


3. **模型推理报错**
* 请确保 `npu_wrapper.py` 中加载 `.om` 模型的路径是绝对路径或相对于执行文件的正确相对路径。


4. **OpenCV 链接错误**
* 项目默认使用 `pkg-config --libs opencv4`。如果未安装 opencv 开发包，请运行 `sudo apt install libopencv-dev`。
