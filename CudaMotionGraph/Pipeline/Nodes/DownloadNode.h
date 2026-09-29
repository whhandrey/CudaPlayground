#pragma once
#include <Node/Node.h>
#include <Node/NodeBuildContext.h>
#include <Port/ResourcePort.h>

namespace pipeline {
	using namespace dataflow;
	using image::PinnedImageView;
	using image::GpuImageView;

	struct DownloadNodeParams {
		ResourceDesc gpuInputDesc;
		ResourceDesc cpuOutputDesc;
	};

	class DownloadNode : public dataflow::NodeBase {
	public:
		DownloadNode(NodeId id, const NodeBuildContext& ctx, const DownloadNodeParams& params);

	public:
		void Execute(const CudaExecutionContext& ctx) override;

	private:
		ResourceInPort<GpuImageView<uchar4>> m_gpuInput;
		ResourceOutPort<PinnedImageView<image::vec4uc>> m_cpuOutput;
	};
}
