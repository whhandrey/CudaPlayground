#include "CudaMotionPipeline.h"
#include "Nodes/UploadNode.h"

namespace pipeline {
	using processing::resource::MemoryDomain;

	void CudaMotionPipeline::Create(const MotionPipelineConfig& config) {

	}

	std::vector<NodeDefinition> CudaMotionPipeline::CreateNodesDefs(const MotionPipelineConfig& config) {
		ResourceDesc cpuInputDesc {
			.dim = config.dim,
			.sizeOfElemBytes = sizeof(image::vec4uc),
			.sampleType = typeid(image::vec4uc),
			.domain = MemoryDomain::HostPinned
		};

		ResourceDesc gpuInputDesc {
			.dim = config.dim,
			.sizeOfElemBytes = sizeof(uchar4),
			.sampleType = typeid(uchar4),
			.domain = MemoryDomain::CudaDevice
		};

		UploadNodeParams params {
			.cpuInputName = "cpu.firstFrame",
			.gpuOutputName = "gpu.firstFrame",
			.cpuInputDesc = cpuInputDesc,
			.gpuOutputDesc = gpuInputDesc
		};

		NodeCreateFunc createUploadNode = [params](NodeId id, const NodeBuildContext& ctx) -> NodeBase::Ptr {
			return std::make_unique<UploadNode>(id, ctx, params);
		};

		NodeDefinition uploadDef {
			"uploadNode",
			createUploadNode
		};

		return { uploadDef };
	}
}
