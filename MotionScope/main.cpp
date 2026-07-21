#include "MainWindow/MainWindow.h"
#include "Controller/Controller.h"
#include "GpuWorker/AsyncGpuWorker.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    auto proc = cuda::motion::IMotionViewProcessor::Create();
    auto procWrapper = std::make_unique<app::motion::MotionViewProcessor>(std::move(proc));

    auto worker = std::make_unique<app::worker::AsyncGpuWorker>(std::move(procWrapper));
    auto controller = std::make_unique<app::Controller>(std::move(worker));

    app::MainWindow window(std::move(controller));
    window.show();

    return app.exec();
}
