#pragma once
#include "MotionPipelineConfig.h"
#include <Graph/GraphRuntime.h>

namespace pipeline {
	using dataflow::GraphRuntime;
	using dataflow::NodeDefinition;

	class CudaMotionPipeline {
	public:
		void Create(const MotionPipelineConfig& config);

		void Execute();

	private:
		std::vector<NodeDefinition> CreateNodesDefs(const MotionPipelineConfig& config);

	private:
		GraphRuntime m_runtime;
	};
}
