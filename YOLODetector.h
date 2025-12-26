#ifndef YOLODETECTOR_H
#define YOLODETECTOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 检测结果结构体
struct DetectResult {
    std::string className;   // 类别名称
    float confidence;        // 置信度
    cv::Rect bbox;           // 边界框（x1,y1,x2,y2）
};

// Python封装的YOLO检测器类（C++接口）
class YOLODetector {
public:
    // 构造函数：初始化模型和标签
    YOLODetector(const std::string& modelPath, const std::string& labelPath,
                 float confThres = 0.4f, float iouThres = 0.5f, int inputShape = 640);

    // 析构函数
    ~YOLODetector();

    // 检测单帧图像
    std::vector<DetectResult> detectFrame(const cv::Mat& frame, cv::Mat& outFrame);

    // 保存检测结果（图像+文本）
    bool saveResult(const cv::Mat& detectFrame, const std::vector<DetectResult>& results,
                    const std::string& savePath = "detect_result");

private:
    // 隐藏Python相关实现（通过PyBind11封装）
    void* d_ptr;  //  opaque pointer 指向Python对象
};

#endif // YOLODETECTOR_H
