#include "QImageView.h"
#include <QPainter>

namespace app {
    QImageView::QImageView(QWidget* parent)
        : QWidget(parent)
    {
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        setMinimumSize(100, 100);
    }

    void QImageView::SetImage(QImage image) {
        m_image = std::move(image);
        update();
    }

    void QImageView::paintEvent(QPaintEvent*) {
        QPainter painter(this);
        painter.fillRect(rect(), Qt::black);

        if (m_image.isNull()) {
            return;
        }

        const QSize targetSize = m_image.size().scaled(size(), Qt::KeepAspectRatio);

        const QPoint topLeft{
            (width() - targetSize.width()) / 2,
            (height() - targetSize.height()) / 2
        };

        const QRect targetRect(topLeft, targetSize);

        painter.setRenderHint(QPainter::SmoothPixmapTransform, false);
        painter.drawImage(targetRect, m_image);
    }
}
