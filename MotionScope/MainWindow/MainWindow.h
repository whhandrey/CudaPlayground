#pragma once
#include <QtWidgets/QMainWindow>

class QLabel;
class QSlider;
class QPushButton;
class QTimer;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    QLabel* CreateImagePlaceholder(const QString& text);

    // Handlers
    void OpenFolder();
    void ShowFrame(int idx);
    void TogglePlay();

private:
    QLabel* m_prevImgLabel = nullptr;
    QLabel* m_currImgLabel = nullptr;
    QLabel* m_confImgLabel = nullptr;

    QSlider* m_frameSlider = nullptr;

    QPushButton* m_playBtn = nullptr;
    QPushButton* m_openFolderBtn = nullptr;

    QStringList m_frameFiles;
    int m_currentFrame = 0;
    QTimer* m_playTimer = nullptr;
};

