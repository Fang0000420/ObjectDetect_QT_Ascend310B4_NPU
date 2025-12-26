QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17
QMAKE_PROJECT_DEPTH = 0

# ==========================================
# 全局配置：匹配你的实际文件结构
# ==========================================
HEADERS += \
    YOLODetector.h \
    mainwindow.h

SOURCES += \
    main.cpp \
    mainwindow.cpp \
    yolo_npu_wrapper.cpp

FORMS += \
    mainwindow.ui

# 复制npu_wrapper.py到构建目录（避免脚本路径找不到）
DISTFILES += npu_wrapper.py \
    det_utils.py
# Linux下自动复制脚本到构建目录
unix:!macx {
    COPIES += NpuScript
    NpuScript.files = npu_wrapper.py
    NpuScript.path = $$OUT_PWD
}

# ==========================================
# Windows 配置（保留你的原有配置）
# ==========================================
win32 {
    OPENCV_DIR = D:\CLearn\opencv\build

    INCLUDEPATH += $$OPENCV_DIR/include
    INCLUDEPATH += $$OPENCV_DIR/include/opencv2

    LIBS += -L$$OPENCV_DIR/x64/vc16/lib
    CONFIG(debug, debug|release): LIBS += -lopencv_world4120d
    else: LIBS += -lopencv_world4120

    # Windows下Python依赖（示例：Python3.8）
    PYTHON_DIR = C:\Python38
    INCLUDEPATH += $$PYTHON_DIR/include
    LIBS += -L$$PYTHON_DIR/libs -lpython38
}

# ==========================================
# Linux 配置（香橙派 AIpro 环境：使用你查到的实际路径）
# ==========================================
unix:!macx {
    CONFIG += link_pkgconfig
    PKGCONFIG += opencv4

    # 1. 指向 Miniconda 的头文件和库路径
    PYTHON_INC = /usr/local/miniconda3/include/python3.9
    PYTHON_LIB = /usr/local/miniconda3/lib

    INCLUDEPATH += $$PYTHON_INC
    # 2. 链接 python3.9 库
    LIBS += -L$$PYTHON_LIB -lpython3.9

    # 3. PyBind11 路径
    PYBIND11_INC = /usr/include/pybind11
    INCLUDEPATH += $$PYBIND11_INC

    LIBS += -ldl -lpthread -lutil
}

# ==========================================
# 部署配置
# ==========================================
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
