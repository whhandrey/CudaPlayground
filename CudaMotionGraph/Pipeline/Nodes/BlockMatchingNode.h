#pragma once
#include <Node/Node.h>
#include <Port/ResourcePort.h>
#include <Port/ParamPort.h>
#include <Cuda/Motion/BlockMatching.h>
#include <Image/ImageView.h>

namespace pipeline {
	using cuda::motion::BlockMatchingParams;
	using cuda::motion::BlockMatchStats;
	using namespace dataflow;

	struct BlockMatchingNodeParams {
		INodeChangeListener& listener;
		IPortRegistry& registry;
		IResourceRegistry& resRegistry;
		ResourceDesc prevImageDesc;
		ResourceDesc currImageDesc;
		ResourceDesc statsDesc;
	};

	class BlockMatchingNode : public NodeBase {
	public:
		BlockMatchingNode(NodeId id, const BlockMatchingNodeParams& params);

	public:
		void Execute(const CudaExecutionContext& ctx) override;

	private:
		InParamPort<BlockMatchingParams> m_params;

		ResourceInPort<image::GpuImageView<uchar4>> m_prev;
		ResourceInPort<image::GpuImageView<uchar4>> m_curr;

		ResourceOutPort<image::GpuImageView<BlockMatchStats>> m_stats;
	};
}
