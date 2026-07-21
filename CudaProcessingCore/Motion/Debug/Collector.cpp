#include "Collector.h"

namespace cuda::motion::debug {
	using  namespace ::motion::debug;

	void Collector::AddParams(const BlockMatchingParams& p)
	{
		m_stats["Algo/Params"].push_back(StatsField{ "BlockDim", p.blockDim, Unit::px });
		m_stats["Algo/Params"].push_back(StatsField{ "MacroBlockDim", p.macroBlockDim, Unit::px });
		m_stats["Algo/Params"].push_back(StatsField{ "SearchRad", p.search_halfsize, Unit::px });
	}

	void Collector::AddStat(const std::string& scope, const std::string& group, const std::string& name, float time)
	{
		const std::string key = scope + "/" + group;
		m_stats[key].push_back(StatsField{ name, time, Unit::ms });
	}

	StatsPacket Collector::TakeStats()
	{
		return std::exchange(m_stats, {});
	}
}
