#include "Collector.h"
#include <Debug/StatsProvider.h>
#include <Debug/Formatter.h>
#include <Debug/StatsField.h>
#include <algorithm>
#include <iterator>
#include <stdexcept>

namespace detail {
	using namespace ::motion::debug;
	using cuda::motion::BlockMatchingParams;

	std::vector<std::string> FieldNames(const std::map<std::string, float>& stats) {
		std::vector<std::string> statsNames;

		std::transform(stats.begin(), stats.end(), std::back_inserter(statsNames), [](const auto& p) {
			return p.first;
		});

		return statsNames;
	}

	class StatsProvider : public IStatsProvider {
	public:
		StatsProvider(
			BlockMatchingParams params,
			std::map<std::string, float>&& cpuStats,
			std::map<std::string, float>&& gpuStats
		);

	public:
		FieldGroupMap GetFieldNamesByGroups() const override;
		std::string GetField(const StatsField& field, IFormatter& formatter) const override;

	private:
		std::string BlockMatchParamToString(const std::string& name, IFormatter& formatter) const;

	private:
		BlockMatchingParams m_params;
		std::map<std::string, float> m_cpuStats;
		std::map<std::string, float> m_gpuStats;
	};

	StatsProvider::StatsProvider(BlockMatchingParams params, std::map<std::string, float>&& cpuStats, std::map<std::string, float>&& gpuStats)
		: m_params{ params }
		, m_cpuStats{ std::move(cpuStats) }
		, m_gpuStats{ std::move(gpuStats) }
	{
	}

	FieldGroupMap StatsProvider::GetFieldNamesByGroups() const {
		std::vector<std::string> blockMatchingParams {
			"BlockDim",
			"MacroBlockDim",
			"SearchRad"
		};

		FieldGroupMap output;

		output.emplace("BlockMatching", blockMatchingParams);
		output.emplace("CpuStats", FieldNames(m_cpuStats));
		output.emplace("GpuStats", FieldNames(m_gpuStats));

		return output;
	}

	std::string StatsProvider::GetField(const StatsField& field, IFormatter& formatter) const {
		if (field.group == "BlockMatching") {
			return BlockMatchParamToString(field.name, formatter);
		}

		if (field.group == "CpuStats") {
			if (m_cpuStats.find(field.name) == m_cpuStats.end()) {
				throw std::logic_error("StatsProvider::GetField: unknown CPU field");
			}

			return formatter.Format(m_cpuStats.at(field.name));
		}

		if (field.group == "GpuStats") {
			if (m_gpuStats.find(field.name) == m_gpuStats.end()) {
				throw std::logic_error("StatsProvider::GetField: unknown GPU field");
			}

			return formatter.Format(m_gpuStats.at(field.name));
		}

		throw std::logic_error("StatsProvider::GetField: unknown group");
	}

	std::string StatsProvider::BlockMatchParamToString(const std::string& name, IFormatter& formatter) const {
		if (name == "BlockDim") {
			return formatter.Format(int(m_params.blockDim.x), int(m_params.blockDim.y));
		}
		
		if (name == "MacroBlockDim") {
			return formatter.Format(int(m_params.macroBlockDim.x), int(m_params.macroBlockDim.y));
		}

		if (name == "SearchRad") {
			return formatter.Format(m_params.search_halfsize.x, m_params.search_halfsize.y);
		}

		throw std::logic_error("StatsProvider::BlockMatchParamToString: unknown field");
	}
}

namespace cuda::motion::debug {
	Collector::Collector(StatsCallback&& callback)
		: m_callback{ std::move(callback) }
	{
	}

	void Collector::AddParams(const BlockMatchingParams& p)
	{
		m_params = p;
	}

	void Collector::AddCpuStat(const std::string& name, float time)
	{
		m_cpuStats.insert_or_assign(name, time);
	}

	void Collector::AddGpuStat(const std::string& name, float time)
	{
		m_gpuStats.insert_or_assign(name, time);
	}

	void Collector::IssueCallback()
	{
		if (!m_callback) {
			return;
		}

		m_callback(std::make_unique<detail::StatsProvider>(m_params, std::move(m_cpuStats), std::move(m_gpuStats)));

		ClearState();
	}

	void Collector::ClearState()
	{
		m_params = {};
		m_cpuStats.clear();
		m_gpuStats.clear();
	}
}
