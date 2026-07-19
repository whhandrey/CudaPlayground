#include "MainWindow/MainWindow.h"
#include "Controller/Controller.h"
#include "GpuWorker/GpuWorkerFactory.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    auto gpuWorkerFactory = app::worker::GpuWorkerFactory();
    auto controller = std::make_unique<app::Controller>(gpuWorkerFactory);

    app::MainWindow window(std::move(controller));
    window.show();

    return app.exec();
}
