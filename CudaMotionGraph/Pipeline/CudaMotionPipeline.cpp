#include "CudaMotionPipeline.h"
#include "Nodes/UploadNode.h"
#include "Nodes/BlockMatchingNode.h"
#include "Nodes/DownloadNode.h"

namespace {
	using namespace dataflow;
	using namespace processing::resource;
	using namespace pipeline;

	NodeDefinition MakeUploadNodeDef(image::vec2ui dim, const std::string& cpuInName, const std::string& gpuOutName) {
		ResourceDesc cpuInputDesc = MakeResourceDesc<image::vec4uc>(cpuInName, dim, MemoryDomain::HostPinned);
		ResourceDesc gpuInputDesc = MakeResourceDesc<uchar4>(gpuOutName, dim, MemoryDomain::CudaDevice);

		UploadNodeParams params {
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

	NodeDefinition MakeBlockMatchingNodeDef(
		const MotionPipelineConfig& config,
		const std::string& gpuFirstFrameKey,
		const std::string& gpuSecondFrameKey,
		const std::string& statsKey)
	{
		ResourceDesc prevImageDesc = MakeResourceDesc<uchar4>(gpuFirstFrameKey, config.imageDim, MemoryDomain::CudaDevice);
		ResourceDesc currImageDesc = MakeResourceDesc<uchar4>(gpuSecondFrameKey, config.imageDim, MemoryDomain::CudaDevice);
		ResourceDesc statsDesc = MakeResourceDesc<BlockMatchStats>(statsKey, config.statsDim, MemoryDomain::CudaDevice);

		BlockMatchingNodeParams params{
			.prevImageDesc = prevImageDesc,
			.currImageDesc = currImageDesc,
			.statsDesc = statsDesc
		};

		NodeCreateFunc createBlockMatchNode = [params](NodeId id, const NodeBuildContext& ctx) -> NodeBase::Ptr {
			return std::make_unique<BlockMatchingNode>(id, ctx, params);
		};

		return {
			"blockMatchingNode",
			createBlockMatchNode
		};
	}

	NodeDefinition MakeDownloadNodeDef(
		image::vec2ui dim,
		const std::string& gpuInKey,
		const std::string& cpuOutKey)
	{
		ResourceDesc gpuInputDesc = MakeResourceDesc<uchar4>(gpuInKey, dim, MemoryDomain::CudaDevice);
		ResourceDesc cpuOutputDesc = MakeResourceDesc<uchar4>(cpuOutKey, dim, MemoryDomain::HostPinned);

		DownloadNodeParams params{
			.gpuInputDesc = gpuInputDesc,
			.cpuOutputDesc = cpuOutputDesc,
		};

		NodeCreateFunc createDownloadNode = [params](NodeId id, const NodeBuildContext& ctx) -> NodeBase::Ptr {
			return std::make_unique<DownloadNode>(id, ctx, params);
		};

		return {
			"downloadNode",
			createDownloadNode
		};
	}

}

namespace pipeline {
	using namespace processing::resource;

	void CudaMotionPipeline::Create(const MotionPipelineConfig& config) {

	}

	std::vector<NodeDefinition> CudaMotionPipeline::CreateNodesDefs(const MotionPipelineConfig& config) {
		const std::string cpuFirstFrameKey = "cpu.firstFrame";
		const std::string cpuSecondFrameKey = "cpu.secondFrame";

		const std::string gpuFirstFrameKey = "gpu.firstFrame";
		const std::string gpuSecondFrameKey = "gpu.secondFrame";

		const std::string gpuStatsKey = "gpu.blockMatchStats";
		const std::string cpuStatsKey = "cpu.blockMatchStats";

		return {
			MakeUploadNodeDef(config.imageDim, cpuFirstFrameKey, gpuFirstFrameKey),
			MakeUploadNodeDef(config.imageDim, cpuSecondFrameKey, gpuSecondFrameKey),
			MakeBlockMatchingNodeDef(config, gpuFirstFrameKey, gpuSecondFrameKey, gpuStatsKey),
			MakeDownloadNodeDef(config.statsDim, gpuStatsKey, cpuStatsKey)
		};
	}
}
