#pragma once
#include <string>
#include <memory>
#include <QImage>

namespace image {
	class Loader {
	public:
		Loader(const std::string& folderPath);

	public:
		QImage Load(size_t index) const;
		size_t NumImages() const;

	private:
		const std::string m_folder;
		std::vector<std::string> m_framePaths;
	};
}
