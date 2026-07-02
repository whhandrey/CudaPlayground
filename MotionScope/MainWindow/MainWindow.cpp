#include "../Controller/Controller.h"
#include "MainWindow.h"

#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QWidget>
#include <QTimer>
#include <QFileDialog>
#include <QMessageBox>
#include <QSignalBlocker>
#include <QCheckBox>

using GpuApp::PlayMode;

namespace {
    QPixmap FitToLabel(const QImage& image, QSize labelSize) {
        return QPixmap::fromImage(image).scaled(
            labelSize,
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation
        );
    }
}

MainWindow::MainWindow(std::unique_ptr<GpuApp::Controller> controller, QWidget* parent)
    : QMainWindow(parent)
    , m_controller{ std::move(controller) }
{
    setWindowTitle("Cuda motion scope");

    auto* central = new QWidget(this);
    setCentralWidget(central);

    // this registers mainLayout in the central one.
    auto* mainLayout = new QVBoxLayout(central);

    m_prevImgLabel = CreateImagePlaceholder("PrevFrame");
    m_currImgLabel = CreateImagePlaceholder("CurrFrame");
    m_confImgLabel = CreateImagePlaceholder("ConfImage");
    m_visImgLabel = CreateImagePlaceholder("VisualizationImage");

    QVBoxLayout* leftLayout = new QVBoxLayout();
    leftLayout->addWidget(m_prevImgLabel);
    leftLayout->addWidget(m_currImgLabel);

    QVBoxLayout* rightLayout = new QVBoxLayout();
    rightLayout->addWidget(m_confImgLabel);
    rightLayout->addWidget(m_visImgLabel);

    QHBoxLayout* fullImageLayout = new QHBoxLayout();
    fullImageLayout->addLayout(leftLayout);
    fullImageLayout->addLayout(rightLayout);

    m_frameSlider = new QSlider(Qt::Horizontal, central);
    m_frameSlider->setRange(0, 0);

    m_openFolderBtn = new QPushButton("Open Folder", central);

    m_playBtn = new QPushButton("Play", central);
    m_prevFrameBtn = new QPushButton("Prev", central);
    m_nextFrameBtn = new QPushButton("Next", central);

    m_loopCheckBox = new QCheckBox("Loop", central);

    auto* controlsLayout = new QHBoxLayout();

    controlsLayout->addWidget(m_openFolderBtn);

    controlsLayout->addSpacing(16);

    controlsLayout->addWidget(m_playBtn);
    controlsLayout->addWidget(m_prevFrameBtn);
    controlsLayout->addWidget(m_nextFrameBtn);
    controlsLayout->addWidget(m_loopCheckBox);

    controlsLayout->addStretch();

    mainLayout->addLayout(fullImageLayout);
    mainLayout->addWidget(m_frameSlider);
    mainLayout->addLayout(controlsLayout);

    {
        m_playTimer = new QTimer(this);
        m_playTimer->setInterval(80); // 12.5 fps

        connect(m_openFolderBtn, &QPushButton::clicked, this, &MainWindow::OpenFolder);
        connect(m_playBtn, &QPushButton::clicked, this, &MainWindow::TogglePlay);
        connect(m_frameSlider, &QSlider::valueChanged, this, &MainWindow::ShowFrame);

        connect(m_loopCheckBox, &QCheckBox::toggled, this, [this](bool checked) {
            const auto mode = checked ? PlayMode::Loop : PlayMode::Normal;
            m_controller->SetPlayMode(mode);
        });
        
        connect(m_prevFrameBtn, &QPushButton::clicked, this, [this]() {
            m_controller->TryStepBackward();
            SyncSliderState();
        });

        connect(m_nextFrameBtn, &QPushButton::clicked, this, [this]() {
            m_controller->TryStepForward();
            SyncSliderState();
        });

        connect(m_playTimer, &QTimer::timeout, this, [this]() {
            const bool moved = m_controller->TryStepForward();
            SyncSliderState();

            if (!moved) {
                TogglePlay();
            }
        });
    }

    {
        connect(m_controller.get(), &GpuApp::Controller::ImagesReady, this, &MainWindow::ShowImages);
    }

    resize(1300, 650);
}

MainWindow::~MainWindow()
{
}

QLabel* MainWindow::CreateImagePlaceholder(const QString& text) {
    auto* label = new QLabel(text, this);

    label->setMinimumSize(360, 270);
    label->setAlignment(Qt::AlignCenter);

    label->setStyleSheet(
        "QLabel {"
        " background-color: #222;"
        " color: white;"
        " border: 1px solid #555;"
        " font-size: 16px;"
        "}"
    );

    label->setScaledContents(false);
    return label;
}

void MainWindow::SyncSliderState()
{
    QSignalBlocker blocker(m_frameSlider);
    m_frameSlider->setValue(m_controller->GetCurrentIndex());
}

void MainWindow::OpenFolder()
{
    const QString folderPath = QFileDialog::getExistingDirectory(this, "Open frames folder");
    if (folderPath.isEmpty())
        return;

    try {
        m_controller->SetFolder(folderPath.toStdString());

        QSignalBlocker blocker(m_frameSlider);
        const auto framesRange = m_controller->GetFramesRange();

        m_frameSlider->setRange(framesRange.first, framesRange.second);
        m_frameSlider->setValue(framesRange.first);
    }
    catch (const std::exception& e) {
        QMessageBox::critical(this, "Open folder failed", QString::fromUtf8(e.what()));
    }
}

void MainWindow::ShowFrame(int idx)
{
    m_controller->SetFrame(idx);
    SyncSliderState();
}

void MainWindow::TogglePlay()
{
    if (m_playTimer->isActive()) {
        m_playTimer->stop();
        m_playBtn->setText("Play");
    }
    else {
        m_playTimer->start();
        m_playBtn->setText("Pause");
    }
}

void MainWindow::ShowImages(QImage prev, QImage curr, QImage conf, QImage vis)
{
    m_prevImgLabel->setPixmap(FitToLabel(prev, m_prevImgLabel->size()));
    m_currImgLabel->setPixmap(FitToLabel(curr, m_currImgLabel->size()));
    m_confImgLabel->setPixmap(FitToLabel(conf, m_confImgLabel->size()));
    m_visImgLabel->setPixmap(FitToLabel(vis, m_visImgLabel->size()));
}
