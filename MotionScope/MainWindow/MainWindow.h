#pragma once
#include "../DisplayTypes/DisplayTypes.h"
#include "../Stats/StatsProvider.h"
#include <QtWidgets/QMainWindow>
#include <memory>

class QLabel;
class QSlider;
class QPushButton;
class QTimer;
class QCheckBox;
class QComboBox;
class QSpinBox;
class QTreeWidget;
class QDockWidget;

namespace app {
    using ::motion::debug::StatsPacket;

    class Controller;
    class QImageView;

    enum class ViewSlot;

    class MainWindow : public QMainWindow
    {
        Q_OBJECT

    public:
        MainWindow(std::unique_ptr<app::Controller> controller, QWidget* parent = nullptr);
        ~MainWindow();

    private:
        void SyncSliderState();
        void SyncFrameOffset();

        void SyncUiControls();

        // Handlers
        void OpenFolder();
        void ShowFrame(int idx);
        void TogglePlay();

        void FillComboView(QComboBox* comboBox);

        void InitStatsDock();

    private slots:
        void ShowImages(QImage prev, QImage curr, QImage view1, QImage view2);
        void ShowViews(std::vector<SlottedImage> views);
        void ShowDebugStats(StatsProvider provider);

    private:
        std::unique_ptr<app::Controller> m_controller;

        QImageView* m_prevImgLabel = nullptr;
        QImageView* m_currImgLabel = nullptr;
        QImageView* m_view1ImgLabel = nullptr;
        QImageView* m_view2ImgLabel = nullptr;

        QSlider* m_frameSlider = nullptr;

        QPushButton* m_playBtn = nullptr;
        QPushButton* m_openFolderBtn = nullptr;
        QPushButton* m_nextFrameBtn = nullptr;
        QPushButton* m_prevFrameBtn = nullptr;
        QCheckBox* m_loopCheckBox = nullptr;

        QComboBox* m_view1Combo = nullptr;
        QComboBox* m_view2Combo = nullptr;

        QComboBox* m_motionAlgoCombo = nullptr;
        QSpinBox* m_offsetSelector = nullptr;

        QTreeWidget* m_statsTree = nullptr;
        QDockWidget* m_statsDock = nullptr;
        QPushButton* m_statsBtn = nullptr;

        QTimer* m_playTimer = nullptr;
    };
}
