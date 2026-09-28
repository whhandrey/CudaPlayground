#include "CudaMotionPipeline.h"
#include "Nodes/UploadNode.h"

namespace pipeline {
	using processing::resource::MemoryDomain;

	template <class T>
	ResourceDesc MakeResourceDesc(image::vec2ui dim, MemoryDomain domain) {
		return {
			.dim = dim,
			.sizeOfElemBytes = sizeof(T),
			.sampleType = typeid(T),
			.domain = domain
		};
	}

	NodeDefinition MakeUploadNodeDef(image::vec2ui dim) {
		ResourceDesc cpuInputDesc = MakeResourceDesc<image::vec4uc>(dim, MemoryDomain::HostPinned);
		ResourceDesc gpuInputDesc = MakeResourceDesc<uchar4>(dim, MemoryDomain::CudaDevice);

		UploadNodeParams params{
			.cpuInputName = "cpu.firstFrame",
			.gpuOutputName = "gpu.firstFrame",
			.cpuInputDesc = cpuInputDesc,
			.gpuOutputDesc = gpuInputDesc
		};

		NodeCreateFunc createUploadNode = [params](NodeId id, const NodeBuildContext& ctx) -> NodeBase::Ptr {
			return std::make_unique<UploadNode>(id, ctx, params);
		};

		return {
			"uploadNode",
			createUploadNode
		};
	}

	void CudaMotionPipeline::Create(const MotionPipelineConfig& config) {

	}

	std::vector<NodeDefinition> CudaMotionPipeline::CreateNodesDefs(const MotionPipelineConfig& config) {
		return { MakeUploadNodeDef(config.dim) };
	}
}
