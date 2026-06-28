#pragma once
#include <QtWidgets/QMainWindow>
#include <memory>

class QLabel;
class QSlider;
class QPushButton;
class QTimer;

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

    // Handlers
    void OpenFolder();
    void ShowFrame(int idx);
    void TogglePlay();

private slots:
    void ShowImages(QImage prev, QImage curr, QImage conf, QImage vis);

private:
    std::unique_ptr<GpuApp::Controller> m_controller;

    QLabel* m_prevImgLabel = nullptr;
    QLabel* m_currImgLabel = nullptr;
    QLabel* m_confImgLabel = nullptr;
    QLabel* m_visImgLabel = nullptr;

    QSlider* m_frameSlider = nullptr;

    QPushButton* m_playBtn = nullptr;
    QPushButton* m_openFolderBtn = nullptr;

    QStringList m_frameFiles;
    int m_currentFrame = 0;
    QTimer* m_playTimer = nullptr;
};

