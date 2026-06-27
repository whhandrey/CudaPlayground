#include "MainWindow/MainWindow.h"
#include "Controller/Controller.h"
#include "GpuWorker/AsyncGpuWorker.h"
#include <GpuProcessor/IMotionViewProcessor.h>
#include <QtWidgets/QApplication>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    auto matcher = cuda::motion::IMotionViewProcessor::Create();

    auto gpuWorker = std::make_unique<gpu::motion::AsyncGpuWorker>(std::move(matcher));
    auto controller = std::make_unique<GpuApp::Controller>(std::move(gpuWorker));

    MainWindow window(std::move(controller));
    window.show();

    return app.exec();
}
