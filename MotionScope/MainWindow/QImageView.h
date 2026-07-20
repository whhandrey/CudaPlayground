#pragma once
#include <QWidget>
#include <QImage>

namespace app {
	class QImageView : public QWidget {
		Q_OBJECT

	public:
        explicit QImageView(QWidget* parent = nullptr);

        void SetImage(QImage image);

    protected:
        void paintEvent(QPaintEvent*) override;

    private:
        QImage m_image;
	};
}
