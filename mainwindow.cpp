#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QDateTime>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow) {
    ui->setupUi(this);

    // 初始化定时器（视频播放帧率：33ms ≈ 30fps）
    m_videoTimer.setInterval(33);
    connect(&m_videoTimer, &QTimer::timeout, this, &MainWindow::updateVideoFrame);

    // 初始化YOLO检测器
    if (!initDetector()) {
        QMessageBox::critical(this, "错误", "检测器初始化失败！请检查模型和标签文件路径");
    } else {
        // 如果检测器初始化成功，先把按钮设为可用状态（但在没视频时依然禁用）
        ui->detectBtn->setEnabled(false);
    }

    // 初始化按钮状态
    ui->playBtn->setEnabled(false);
    ui->saveBtn->setEnabled(false); // 保存按钮仅在有结果时启用

    // 初始时禁用检测标志
    m_isDetecting = false;
}

MainWindow::~MainWindow() {
    m_videoTimer.stop();
    m_videoCap.release();
    if (m_detector) {
        delete m_detector;
        m_detector = nullptr;
    }
    delete ui;
}

bool MainWindow::initDetector() {
    // 请确保这些路径与你实际的文件位置一致
    std::string modelPath = "yolo5.om";
    std::string labelPath = "coco_names.txt";

    try {
        m_detector = new YOLODetector(modelPath, labelPath);
        return true;
    } catch (const std::exception& e) {
        qDebug() << "检测器初始化失败，原因：" << e.what();
        return false;
    }
}

QImage MainWindow::matToQImage(const cv::Mat& mat) {
    if (mat.empty()) return QImage();

    // 保持长宽比缩放，避免图像拉伸变形
    // 注意：如果在update中频繁调用，最好只转换不缩放，在Label的setScaledContents中处理，
    // 但为了显示效果，这里先手动缩放。
    cv::Mat rgbMat;
    if (mat.channels() == 3) {
        cv::cvtColor(mat, rgbMat, cv::COLOR_BGR2RGB);
    } else if (mat.channels() == 1) {
        cv::cvtColor(mat, rgbMat, cv::COLOR_GRAY2RGB);
    }

    QImage qimg(rgbMat.data, rgbMat.cols, rgbMat.rows, rgbMat.step, QImage::Format_RGB888);
    // 使用 copy() 确保数据所有权分离，防止崩溃
    return qimg.copy();
}

void MainWindow::on_chooseVideoBtn_clicked() {
    QString fileName = QFileDialog::getOpenFileName(
        this, "选择视频文件", "",
        "Video Files (*.mp4 *.avi *.mov *.mkv *.flv)");

    if (fileName.isEmpty()) return;
    m_videoPath = fileName.toStdString();

    // 停止之前的播放
    m_videoTimer.stop();
    m_isPlaying = false;
    m_isDetecting = false; // 换视频时重置检测状态

    // 打开视频
    m_videoCap.release();
    if (!m_videoCap.open(m_videoPath)) {
        QMessageBox::warning(this, "警告", "无法打开视频文件！");
        return;
    }

    // 读取第一帧预览
    if (m_videoCap.read(m_currentFrame)) {
        // 显示第一帧
        QImage img = matToQImage(m_currentFrame);
        ui->videoLabel->setPixmap(QPixmap::fromImage(img).scaled(ui->videoLabel->size(), Qt::KeepAspectRatio));

        // 清空右侧检测结果
        ui->detectLabel->clear();
        ui->detectLabel->setText("等待检测...");
    }

    // 更新按钮状态
    ui->playBtn->setEnabled(true);
    ui->playBtn->setText("播放"); // 重置文字

    // 如果检测器就绪，允许点击检测按钮
    if (m_detector) {
        ui->detectBtn->setEnabled(true);
        ui->detectBtn->setText("开始检测");
    }

    ui->saveBtn->setEnabled(false);
}

void MainWindow::on_playBtn_clicked() {
    if (!m_videoCap.isOpened()) return;

    if (m_isPlaying) {
        // 暂停
        m_videoTimer.stop();
        m_isPlaying = false;
        ui->playBtn->setText("播放"); // 修正：原来你这里写的是 ui->chooseVideoBtn
    } else {
        // 播放
        m_videoTimer.start();
        m_isPlaying = true;
        ui->playBtn->setText("暂停");
    }
}

// 【修改】检测按钮现在作为“开关”使用
void MainWindow::on_detectBtn_clicked() {
    if (!m_detector) return;

    if (m_isDetecting) {
        // 正在检测 -> 停止检测
        m_isDetecting = false;
        ui->detectBtn->setText("开始检测");
        ui->statusbar->showMessage("检测已停止");
    } else {
        // 未检测 -> 开始检测
        m_isDetecting = true;
        ui->detectBtn->setText("停止检测");
        ui->statusbar->showMessage("实时检测中...");

        // 如果视频没在播放，自动开始播放，方便用户直接看到效果
        if (!m_isPlaying) {
            on_playBtn_clicked();
        }
    }
}

void MainWindow::on_saveBtn_clicked() {
    // 保存当前暂停的那一帧，或者正在播放的这一帧
    if (m_detectFrame.empty() || m_lastResults.empty()) {
        QMessageBox::warning(this, "警告", "当前无检测结果可保存！\n请先播放并开启检测。");
        return;
    }

    // 暂停播放以便保存
    bool wasPlaying = m_isPlaying;
    if (wasPlaying) {
        on_playBtn_clicked(); // 暂停
    }

    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
    std::string savePath = ("detect_result_" + timestamp).toStdString();

    if (m_detector->saveResult(m_detectFrame, m_lastResults, savePath)) {
        QMessageBox::information(this, "成功",
                                 QString("当前帧已保存到：%1").arg(QString::fromStdString(savePath)));
    } else {
        QMessageBox::warning(this, "失败", "保存结果失败！");
    }

    // 如果之前是播放状态，恢复播放（可选，看你需求）
    // if (wasPlaying) on_playBtn_clicked();
}

// 【核心修改】视频帧更新循环
void MainWindow::updateVideoFrame() {
    // 1. 读取视频帧
    if (!m_videoCap.read(m_currentFrame)) {
        // 视频播放结束，循环播放或停止
        // 这里演示循环播放：
        m_videoCap.set(cv::CAP_PROP_POS_FRAMES, 0);
        return;

        /* 如果想停止播放：
        m_videoTimer.stop();
        m_isPlaying = false;
        ui->playBtn->setText("播放");
        ui->statusbar->showMessage("视频播放结束");
        return;
        */
    }

    // 2. 显示左侧原视频
    // 缩放以适应Label大小
    QImage leftImg = matToQImage(m_currentFrame);
    ui->videoLabel->setPixmap(QPixmap::fromImage(leftImg).scaled(ui->videoLabel->size(), Qt::KeepAspectRatio));

    // 3. 处理检测逻辑 (同步模式)
    if (m_isDetecting && m_detector) {
        // 调用检测（注意：这会阻塞主线程，如果推理很慢，界面会卡顿）
        // Orange Pi NPU 推理通常很快，应该能跟上
        m_lastResults = m_detector->detectFrame(m_currentFrame, m_detectFrame);

        // 显示右侧检测视频
        QImage rightImg = matToQImage(m_detectFrame);
        ui->detectLabel->setPixmap(QPixmap::fromImage(rightImg).scaled(ui->detectLabel->size(), Qt::KeepAspectRatio));

        // 启用保存按钮
        ui->saveBtn->setEnabled(true);

        // 更新状态栏信息
        ui->statusbar->showMessage(QString("检测中... 目标数: %1").arg(m_lastResults.size()));
    } else {
        // 如果没开启检测，右侧可以清空，或者显示“未检测”
        // 这里选择保持上一帧或清空，看个人喜好
        // ui->detectLabel->clear();
    }
}
