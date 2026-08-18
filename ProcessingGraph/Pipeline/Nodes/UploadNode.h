#pragma once
#include "../../Node/Node.h"
#include "../../Port/ResourcePort.h"

namespace pipeline {
	using namespace dataflow;
	using image::PinnedImageView;
	using image::GpuImageView;

	struct UploadNodeParams {
		std::string cpuInputName;
		std::string gpuOutputName;
		INodeChangeListener& listener;
		IPortRegistry& registry;
		IResourceRegistry& resRegistry;
		ResourceDesc cpuInputDesc;
		ResourceDesc gpuOutputDesc;
	};

	class UploadNode : public dataflow::NodeBase {
	public:
		UploadNode(NodeId id, const UploadNodeParams& params);

	public:
		void Execute(const CudaExecutionContext& ctx) override;

	private:
		ResourceInPort<PinnedImageView<image::vec4uc>> m_cpuInput;
		ResourceOutPort<GpuImageView<uchar4>> m_gpuOutput;
	};
}
