#pragma once
#include <string>
#include <memory>
#include <QImage>

namespace image {
	class Loader {
	public:
		Loader(const std::string& folderPath);

	public:
		QImage Load(const std::string& name);

	private:
		const std::string m_folder;
		std::vector<std::string> m_framePaths;
	};
}
