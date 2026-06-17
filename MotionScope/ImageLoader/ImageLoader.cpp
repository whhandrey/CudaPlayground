#include "ImageLoader.h"
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

namespace folder {
	std::string ToLower(const std::string& str) {
		std::string out = str;

		std::transform(out.begin(), out.end(), out.begin(), [](char c) {
			return std::tolower(c);
	    });

		return out;
	}

	bool IsImage(const fs::path& path) {
		if (!path.has_extension())
			return false;

		const std::string ext = ToLower(path.extension().string());

		return ext == ".png"
			|| ext == ".jpg"
			|| ext == ".jpeg"
			|| ext == ".bmp"
			|| ext == ".tif"
			|| ext == ".tiff";
	}

	std::vector<std::string> LoadFilePaths(const std::string& path) {
		if (!fs::exists(path)) {
			throw std::logic_error("Loader: path does not exist");
		}

		if (!fs::is_directory(path)) {
			throw std::logic_error("Loader: path is not a folder");
		}

		std::vector<std::string> filePaths;
		for (const auto& entry : fs::directory_iterator(path)) {
			if (!entry.is_regular_file())
				continue;

			const fs::path file = entry.path();
			if (IsImage(path))
				continue;

			filePaths.push_back(file.string());
		}

		return filePaths;
	}
}

namespace image {
	Loader::Loader(const std::string& folderPath)
		: m_folder{ folderPath }
	{
		m_framePaths = folder::LoadFilePaths(m_folder);
		std::sort(m_framePaths.begin(), m_framePaths.end());

		if (m_framePaths.size() < 2) {
			throw std::logic_error("Loader: requires at least two frames");
		}
	}

	QImage Loader::Load(size_t index) const {
		if (index < 0 || index >= m_framePaths.size()) {
			throw std::logic_error("Loader: invalid index");
		}

		QImage img(QString::fromStdString(m_framePaths[index]));
		if (img.isNull()) {
			throw std::logic_error("Loader: image load failed");
		}

		return img.convertToFormat(QImage::Format_RGBA8888);
	}

	size_t Loader::NumImages() const
	{
		return m_framePaths.size();
	}
}
