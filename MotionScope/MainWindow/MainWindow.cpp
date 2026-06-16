#include "MainWindow.h"

#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QWidget>
#include <QTimer>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("Cuda motion scope");

    auto* central = new QWidget(this);
    setCentralWidget(central);

    // this registers mainLayout in the central one.
    auto* mainLayout = new QVBoxLayout(central);

    m_prevImgLabel = CreateImagePlaceholder("PrevFrame");
    m_currImgLabel = CreateImagePlaceholder("CurrFrame");
    m_confImgLabel = CreateImagePlaceholder("ConfImage");

    QVBoxLayout* leftLayout = new QVBoxLayout();
    leftLayout->addWidget(m_prevImgLabel);
    leftLayout->addWidget(m_currImgLabel);

    QHBoxLayout* fullImageLayout = new QHBoxLayout();
    fullImageLayout->addLayout(leftLayout);
    fullImageLayout->addWidget(m_confImgLabel);

    m_frameSlider = new QSlider(Qt::Horizontal, central);
    m_frameSlider->setRange(0, 0);

    auto* controlsLayout = new QHBoxLayout();

    m_openFolderBtn = new QPushButton("Open Folder", central);
    m_playBtn = new QPushButton("Play", central);

    controlsLayout->addWidget(m_openFolderBtn);
    controlsLayout->addWidget(m_playBtn);

    mainLayout->addLayout(fullImageLayout);
    mainLayout->addWidget(m_frameSlider);
    mainLayout->addLayout(controlsLayout);

    {
        m_playTimer = new QTimer(this);
        m_playTimer->setInterval(80); // 12.5 fps

        connect(m_openFolderBtn, &QPushButton::clicked, this, &MainWindow::OpenFolder);
        connect(m_playBtn, &QPushButton::clicked, this, &MainWindow::TogglePlay);
        connect(m_frameSlider, &QSlider::valueChanged, this, &MainWindow::ShowFrame);

        connect(m_playTimer, &QTimer::timeout, this, [this]() {
            if (m_frameFiles.isEmpty())
                return;

            int next = m_currentFrame + 1;
            if (next > m_frameFiles.size()) {
                next = 0;
            }

            m_frameSlider->setValue(next);
            m_currentFrame = next;
        });
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

void MainWindow::OpenFolder()
{
}

void MainWindow::ShowFrame(int idx)
{
}

void MainWindow::TogglePlay()
{
}
