#include "../Controller/Controller.h"
#include "../DisplayTypes/DisplayTypes.h"
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
#include <QGridLayout>
#include <QComboBox>
#include <QGroupBox>
#include <QFormLayout>
#include <QSpinBox>

namespace {
    QPixmap FitToLabel(const QImage& image, QSize labelSize) {
        return QPixmap::fromImage(image).scaled(
            labelSize,
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation
        );
    }

    std::string GetSelectedViewId(QComboBox* comboBox) {
        return comboBox->currentData().toString().toStdString();
    }

    void InitOffsetCtrl(QSpinBox* spinBox) {
        spinBox->setMinimum(1);
        spinBox->setMaximum(1);
        spinBox->setValue(1);

        spinBox->setSingleStep(1);

        spinBox->setPrefix("+");
        spinBox->setSuffix(" frame(s)");

        spinBox->setKeyboardTracking(false);
    }
}

namespace app {
    MainWindow::MainWindow(std::unique_ptr<app::Controller> controller, QWidget* parent)
        : QMainWindow(parent)
        , m_controller{ std::move(controller) }
    {
        setWindowTitle("Cuda motion scope");

        auto* central = new QWidget(this);
        setCentralWidget(central);

        // this registers mainLayout in the central one.
        auto* root = new QVBoxLayout(central);

        m_prevImgLabel = CreateImagePlaceholder("PrevFrame");
        m_currImgLabel = CreateImagePlaceholder("CurrFrame");
        m_view1ImgLabel = CreateImagePlaceholder("View1Image");
        m_view2ImgLabel = CreateImagePlaceholder("View2Image");

        auto* imageGrid = new QGridLayout();

        imageGrid->addWidget(m_prevImgLabel, 0, 0);
        imageGrid->addWidget(m_currImgLabel, 1, 0);
        imageGrid->addWidget(m_view1ImgLabel, 0, 1);
        imageGrid->addWidget(m_view2ImgLabel, 1, 1);

        imageGrid->setColumnStretch(0, 1);
        imageGrid->setColumnStretch(1, 1);
        imageGrid->setRowStretch(0, 1);
        imageGrid->setRowStretch(1, 1);

        auto* viewsGroup = new QGroupBox("Views", central);
        auto* viewsLayout = new QFormLayout(viewsGroup);

        m_view1Combo = new QComboBox(viewsGroup);
        m_view2Combo = new QComboBox(viewsGroup);

        FillComboView(m_view1Combo);
        FillComboView(m_view2Combo);

        viewsLayout->addRow("View 1:", m_view1Combo);
        viewsLayout->addRow("View 2:", m_view2Combo);

        m_openFolderBtn = new QPushButton("Open Folder", central);

        // Algo group
        auto* algoGroup = new QGroupBox("Algo", central);
        auto* algoLayout = new QFormLayout(algoGroup);

        m_motionAlgoCombo = new QComboBox(algoGroup);

        algoLayout->addRow("Motion Algo:", m_motionAlgoCombo);

        // Offset selector
        auto* offsetGroup = new QGroupBox("Frame Offset", central);
        auto* offsetLayout = new QFormLayout(offsetGroup);

        m_offsetSelector = new QSpinBox(offsetGroup);

        offsetLayout->addRow("Offset:", m_offsetSelector);

        auto* rightPanel = new QVBoxLayout();
        rightPanel->addWidget(m_openFolderBtn);
        rightPanel->addWidget(viewsGroup);
        rightPanel->addWidget(offsetGroup);
        rightPanel->addWidget(algoGroup);
        rightPanel->addStretch();

        auto* mainRow = new QHBoxLayout();
        mainRow->addLayout(imageGrid, 1);
        mainRow->addLayout(rightPanel);

        m_playBtn = new QPushButton("Play", central);
        m_prevFrameBtn = new QPushButton("Prev", central);
        m_nextFrameBtn = new QPushButton("Next", central);
        m_loopCheckBox = new QCheckBox("Loop", central);

        auto* playbackRow = new QHBoxLayout();

        playbackRow->addWidget(m_playBtn);
        playbackRow->addWidget(m_prevFrameBtn);
        playbackRow->addWidget(m_nextFrameBtn);
        playbackRow->addWidget(m_loopCheckBox);

        playbackRow->addSpacing(16);
        playbackRow->addStretch();

        m_frameSlider = new QSlider(Qt::Horizontal, central);
        m_frameSlider->setRange(0, 0);

        root->addLayout(mainRow, 1);
        root->addWidget(m_frameSlider);
        root->addLayout(playbackRow);

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
                SyncUiControls();
            });

            connect(m_nextFrameBtn, &QPushButton::clicked, this, [this]() {
                m_controller->TryStepForward();
                SyncUiControls();
            });

            connect(m_playTimer, &QTimer::timeout, this, [this]() {
                const bool moved = m_controller->TryStepForward();
                SyncUiControls();

                if (!moved) {
                    TogglePlay();
                }
            });

            connect(m_view1Combo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) {
                m_controller->SetView(app::ViewSlot::View1, GetSelectedViewId(m_view1Combo));
            });

            connect(m_view2Combo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) {
                m_controller->SetView(app::ViewSlot::View2, GetSelectedViewId(m_view2Combo));
            });

            connect(m_offsetSelector, qOverload<int>(&QSpinBox::valueChanged), this, [this](int offset) {
                m_controller->SetPairOffset(offset);
                SyncFrameOffset();
            });
        }

        {
            connect(m_controller.get(), &Controller::ImagesReady, this, &MainWindow::ShowImages);
            connect(m_controller.get(), &Controller::RenderedViewReady, this, &MainWindow::ShowViews);
        }

        m_controller->SetView(app::ViewSlot::View1, GetSelectedViewId(m_view1Combo));
        m_controller->SetView(app::ViewSlot::View2, GetSelectedViewId(m_view2Combo));

        InitOffsetCtrl(m_offsetSelector);

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

    void MainWindow::SyncFrameOffset()
    {
        QSignalBlocker blocker(m_offsetSelector);

        auto range = m_controller->GetOffsetRange();

        m_offsetSelector->setMinimum(range.first);
        m_offsetSelector->setMaximum(range.second);

        m_offsetSelector->setValue(m_controller->GetPairOffset());
    }

    void MainWindow::SyncUiControls()
    {
        SyncSliderState();
        SyncFrameOffset();
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

            SyncFrameOffset();
        }
        catch (const std::exception& e) {
            QMessageBox::critical(this, "Open folder failed", QString::fromUtf8(e.what()));
        }
    }

    void MainWindow::ShowFrame(int idx)
    {
        m_controller->SetFrame(idx);
        SyncUiControls();
    }

    void MainWindow::TogglePlay()
    {
        if (m_playTimer->isActive()) {
            m_playTimer->stop();
            m_playBtn->setText("Play");
            m_controller->SetPlayState(PlayState::Pause);
        }
        else {
            m_controller->PreparePlaybackStart();
            SyncUiControls();

            m_playTimer->start();
            m_playBtn->setText("Pause");
            m_controller->SetPlayState(PlayState::Play);
        }
    }

    void MainWindow::FillComboView(QComboBox* comboBox)
    {
        QSignalBlocker blocker(comboBox);

        comboBox->clear();

        for (const auto& viewOpt : m_controller->AllViewOptions()) {
            comboBox->addItem(QString::fromStdString(viewOpt.label), QString::fromStdString(viewOpt.id));
        }
    }

    void MainWindow::ShowImages(QImage prev, QImage curr, QImage view1, QImage view2)
    {
        m_prevImgLabel->setPixmap(FitToLabel(prev, m_prevImgLabel->size()));
        m_currImgLabel->setPixmap(FitToLabel(curr, m_currImgLabel->size()));
        m_view1ImgLabel->setPixmap(FitToLabel(view1, m_view1ImgLabel->size()));
        m_view2ImgLabel->setPixmap(FitToLabel(view2, m_view2ImgLabel->size()));
    }

    void MainWindow::ShowViews(std::map<app::ViewSlot, QImage> views)
    {
        for (const auto& view : views) {
            switch (view.first)
            {
            case app::ViewSlot::View1:
                m_view1ImgLabel->setPixmap(FitToLabel(view.second, m_view1ImgLabel->size()));
                break;
            case app::ViewSlot::View2:
                m_view2ImgLabel->setPixmap(FitToLabel(view.second, m_view2ImgLabel->size()));
                break;
            }
        }
    }
}
