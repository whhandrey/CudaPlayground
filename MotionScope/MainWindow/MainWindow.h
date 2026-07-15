#pragma once
#include <QtWidgets/QMainWindow>
#include <memory>

class QLabel;
class QSlider;
class QPushButton;
class QTimer;
class QCheckBox;
class QComboBox;

namespace GpuApp {
    class Controller;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(std::unique_ptr<GpuApp::Controller> controller, QWidget *parent = nullptr);
    ~MainWindow();

private:
    QLabel* CreateImagePlaceholder(const QString& text);
    void SyncSliderState();

    // Handlers
    void OpenFolder();
    void ShowFrame(int idx);
    void TogglePlay();

    void FillComboView(QComboBox* comboBox);

private slots:
    void ShowImages(QImage prev, QImage curr, QImage view1, QImage view2);

private:
    std::unique_ptr<GpuApp::Controller> m_controller;

    QLabel* m_prevImgLabel = nullptr;
    QLabel* m_currImgLabel = nullptr;
    QLabel* m_view1ImgLabel = nullptr;
    QLabel* m_view2ImgLabel = nullptr;

    QSlider* m_frameSlider = nullptr;

    QPushButton* m_playBtn = nullptr;
    QPushButton* m_openFolderBtn = nullptr;
    QPushButton* m_nextFrameBtn = nullptr;
    QPushButton* m_prevFrameBtn = nullptr;
    QCheckBox* m_loopCheckBox = nullptr;

    QComboBox* m_view1Combo = nullptr;
    QComboBox* m_view2Combo = nullptr;

    QTimer* m_playTimer = nullptr;
};
