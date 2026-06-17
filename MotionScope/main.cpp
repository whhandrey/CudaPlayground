#include "MainWindow/MainWindow.h"
#include "Controller/Controller.h"
#include "GpuProcessor/AsyncGpuWorker.h"
#include <QtWidgets/QApplication>

class DummyGpuWorker : public gpu::IMotionGpuProcessor {
public:
    ImageRGBA8 Process(const ImageViewRGBA8& /*prev*/, const ImageViewRGBA8& /*next*/) override {
        return {};
    }
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    auto gpuWorker = std::make_unique<gpu::motion::AsyncGpuWorker>(std::make_unique<DummyGpuWorker>());
    auto controller = std::make_unique<GpuApp::Controller>(std::move(gpuWorker));

    MainWindow window(std::move(controller));
    window.show();

    return app.exec();
}
