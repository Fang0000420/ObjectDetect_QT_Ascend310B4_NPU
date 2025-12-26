#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QThread>
#include <opencv2/opencv.hpp>
#include "YOLODetector.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    // 按钮点击事件
    void on_chooseVideoBtn_clicked();
    void on_playBtn_clicked();
    void on_detectBtn_clicked(); // 这个将变为“开启/停止检测”的开关
    void on_saveBtn_clicked();

    // 视频帧更新
    void updateVideoFrame();

private:
    Ui::MainWindow *ui;

    // 视频相关
    cv::VideoCapture m_videoCap;  // 视频捕获对象
    QTimer m_videoTimer;          // 视频播放定时器
    bool m_isPlaying = false;     // 播放状态
    bool m_isDetecting = false;   // 【新增】检测状态标志：是否开启实时检测

    std::string m_videoPath;      // 视频路径
    cv::Mat m_currentFrame;       // 当前帧（原图）
    cv::Mat m_detectFrame;        // 检测后的帧（画框图）

    // 检测器
    YOLODetector* m_detector = nullptr;  // YOLO检测器实例
    std::vector<DetectResult> m_lastResults;  // 最近一帧的检测结果

    // 辅助函数：Mat转QImage
    QImage matToQImage(const cv::Mat& mat);

    // 初始化检测器
    bool initDetector();
};

#endif // MAINWINDOW_H
