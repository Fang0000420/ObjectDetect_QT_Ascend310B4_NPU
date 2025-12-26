#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include "YOLODetector.h"

namespace py = pybind11;
using namespace pybind11::literals;

// 全局Python解释器初始化（确保只初始化一次）
static bool python_inited = false;
static void init_python()
{
    if (!python_inited)
    {
        // 初始化 Python 解释器
        Py_Initialize();

        try
        {
            // 导入 sys 模块来修改搜索路径
            py::module_ sys = py::module_::import("sys");

            // 1. 添加当前运行目录，确保能找到 npu_wrapper.py
            sys.attr("path").attr("append")(".");

            // 2. 添加你的 Miniconda 库路径 [根据你的输入修改]
            sys.attr("path").attr("append")("/usr/local/miniconda3/lib/python3.9/site-packages");

            // 3. (可选) 添加系统基础库路径，防止某些基础组件丢失
            sys.attr("path").attr("append")("/usr/lib/python3/dist-packages");

            std::cout << "Python 环境路径初始化成功" << std::endl;
        }
        catch (py::error_already_set &e)
        {
            std::cerr << "路径设置失败: " << e.what() << std::endl;
        }

        python_inited = true;
    }
}

// 转换cv::Mat到Python numpy数组
static py::array mat_to_numpy(const cv::Mat &mat)
{
    auto dtype = py::dtype(py::format_descriptor<unsigned char>::format());
    auto shape = py::array::ShapeContainer({mat.rows, mat.cols, mat.channels()});
    auto strides = py::array::StridesContainer({mat.step[0], mat.step[1], sizeof(unsigned char)});
    return py::array(dtype, shape, strides, mat.data, py::none());
}

// 转换Python numpy数组到cv::Mat
static cv::Mat numpy_to_mat(const py::array &arr)
{
    py::buffer_info buf = arr.request();
    int rows = buf.shape[0];
    int cols = buf.shape[1];
    int channels = buf.ndim == 3 ? buf.shape[2] : 1;

    cv::Mat mat(rows, cols, CV_8UC(channels), buf.ptr);
    return mat.clone();
}

// YOLODetector实现（通过PyBind11调用Python）
class YOLODetectorImpl
{
public:
    YOLODetectorImpl(const std::string &modelPath, const std::string &labelPath,
                     float confThres, float iouThres, int inputShape)
    {
        init_python();

        // 导入Python封装类
        py::module_ npu_module = py::module_::import("npu_wrapper");
        py::class_<py::object> detector_cls = npu_module.attr("YOLODetectorWrapper");

        // 创建Python对象
        m_py_detector = detector_cls(
            modelPath, labelPath, "conf_thres"_a = confThres,
            "iou_thres"_a = iouThres, "input_shape"_a = inputShape);
    }

    std::vector<DetectResult> detectFrame(const cv::Mat &frame, cv::Mat &outFrame)
    {
        // 转换帧到numpy数组
        py::array py_frame = mat_to_numpy(frame);

        // 调用Python检测方法
        py::tuple result_tuple = m_py_detector.attr("detect_frame")(py_frame).cast<py::tuple>();
        py::list py_results = result_tuple[0].cast<py::list>();
        py::array py_out_frame = result_tuple[1].cast<py::array>();

        // 转换检测结果
        std::vector<DetectResult> results;
        for (auto &item : py_results)
        {
            py::dict res_dict = item.cast<py::dict>();
            DetectResult res;
            res.className = res_dict["class"].cast<std::string>();
            res.confidence = res_dict["confidence"].cast<float>();
            auto bbox = res_dict["bbox"].cast<std::vector<int>>();
            res.bbox = cv::Rect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
            results.push_back(res);
        }

        // 转换输出帧
        outFrame = numpy_to_mat(py_out_frame);
        return results;
    }

    bool saveResult(const cv::Mat &detectFrame, const std::vector<DetectResult> &results,
                    const std::string &savePath)
    {
        // 转换帧到numpy
        py::array py_frame = mat_to_numpy(detectFrame);

        // 转换结果到Python列表
        py::list py_results;
        for (const auto &res : results)
        {
            py::dict res_dict;
            res_dict["class"] = res.className;
            res_dict["confidence"] = res.confidence;
            res_dict["bbox"] = std::vector<int>{res.bbox.x, res.bbox.y, res.bbox.x + res.bbox.width, res.bbox.y + res.bbox.height};
            py_results.append(res_dict);
        }

        // 调用Python保存方法
        return m_py_detector.attr("save_result")(py_frame, py_results, savePath).cast<bool>();
    }

private:
    py::object m_py_detector;
};

// YOLODetector类的实现（C++接口）
YOLODetector::YOLODetector(const std::string &modelPath, const std::string &labelPath,
                           float confThres, float iouThres, int inputShape)
{
    d_ptr = new YOLODetectorImpl(modelPath, labelPath, confThres, iouThres, inputShape);
}

YOLODetector::~YOLODetector()
{
    delete static_cast<YOLODetectorImpl *>(d_ptr);
    if (python_inited)
    {
        Py_Finalize();
        python_inited = false;
    }
}

std::vector<DetectResult> YOLODetector::detectFrame(const cv::Mat &frame, cv::Mat &outFrame)
{
    return static_cast<YOLODetectorImpl *>(d_ptr)->detectFrame(frame, outFrame);
}

bool YOLODetector::saveResult(const cv::Mat &detectFrame, const std::vector<DetectResult> &results,
                              const std::string &savePath)
{
    return static_cast<YOLODetectorImpl *>(d_ptr)->saveResult(detectFrame, results, savePath);
}

// PyBind11模块导出（生成动态库）
PYBIND11_MODULE(yolo_npu_cpp, m)
{
    m.doc() = "YOLO NPU 目标检测 C++ 接口库";

    // 导出DetectResult结构体
    py::class_<DetectResult>(m, "DetectResult")
        .def(py::init<>())
        .def_readwrite("className", &DetectResult::className)
        .def_readwrite("confidence", &DetectResult::confidence)
        .def_readwrite("bbox", &DetectResult::bbox);

    // 导出YOLODetector类
    py::class_<YOLODetector>(m, "YOLODetector")
        .def(py::init<const std::string &, const std::string &, float, float, int>(),
             "modelPath"_a, "labelPath"_a, "confThres"_a = 0.4f, "iouThres"_a = 0.5f, "inputShape"_a = 640)
        .def("detectFrame", &YOLODetector::detectFrame)
        .def("saveResult", &YOLODetector::saveResult);
}
