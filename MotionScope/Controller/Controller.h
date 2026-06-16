#pragma once
#include "../ImageLoader/ImageLoader.h"
#include <string>
#include <memory>
#include <QImage>

namespace loader {
	class ThreadPool;
}

class Controller {
public:
	Controller(const std::string& folderPath);
	~Controller();

private:
	image::Loader m_loader;
	std::unique_ptr<loader::ThreadPool> m_pool;
};
