#include "MainWindow.h"

#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QWidget>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("Cuda motion scope");

    auto* central = new QWidget(this);
    setCentralWidget(central);

    auto* mainLayout = new QVBoxLayout(central);

    auto* imageLayout = new QHBoxLayout();

    m_prevImgLabel = CreateImagePlaceholder("PrevFrame");
    m_currImgLabel = CreateImagePlaceholder("CurrFrame");
    m_confImgLabel = CreateImagePlaceholder("ConfImage");

    imageLayout->addWidget(m_prevImgLabel);
    imageLayout->addWidget(m_currImgLabel);
    imageLayout->addWidget(m_confImgLabel);


}

MainWindow::~MainWindow()
{
}

QLabel* MainWindow::CreateImagePlaceholder(const QString& text)
{
    return nullptr;
}
    