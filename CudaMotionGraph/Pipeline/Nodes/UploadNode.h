#pragma once
#include <Node/Node.h>
#include <Node/NodeBuildContext.h>
#include <Port/ResourcePort.h>

namespace pipeline {
	using namespace dataflow;
	using image::PinnedImageView;
	using image::GpuImageView;

	struct UploadNodeParams {
		std::string cpuInputName;
		std::string gpuOutputName;
		ResourceDesc cpuInputDesc;
		ResourceDesc gpuOutputDesc;
	};

	class UploadNode : public dataflow::NodeBase {
	public:
		UploadNode(NodeId id, const NodeBuildContext& ctx, const UploadNodeParams& params);

	public:
		void Execute(const CudaExecutionContext& ctx) override;

	private:
		ResourceInPort<PinnedImageView<image::vec4uc>> m_cpuInput;
		ResourceOutPort<GpuImageView<uchar4>> m_gpuOutput;
	};
}
