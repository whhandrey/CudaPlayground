#pragma once
#include <Node/Node.h>
#include <Port/ResourcePort.h>

namespace pipeline {
	using namespace dataflow;
	using image::CpuImageView;
	using image::GpuImageView;

	struct DownloadNodeParams {
		std::string gpuInputName;
		std::string cpuOutputName;
		INodeChangeListener& listener;
		IPortRegistry& registry;
		IResourceRegistry& resRegistry;
		ResourceDesc gpuInputDesc;
		ResourceDesc cpuOutputDesc;
	};

	class DownloadNode : public dataflow::NodeBase {
	public:
		DownloadNode(NodeId id, const DownloadNodeParams& params);

	public:
		void Execute(const CudaExecutionContext& ctx) override;

	private:
		ResourceInPort<GpuImageView<uchar4>> m_gpuInput;
		ResourceOutPort<CpuImageView<image::vec4uc>> m_cpuOutput;
	};
}
