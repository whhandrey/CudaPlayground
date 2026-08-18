#include "UploadNode.h"
#include <Cuda/CudaCheck.h>

namespace {
	template<class Dst, class Src>
	struct IsUploadCompatible
		: std::bool_constant<std::is_same_v<Dst, Src>>
	{
	};

	template<>
	struct IsUploadCompatible<uchar4, image::vec4uc>
		: std::true_type
	{
	};

	// TODO: this must be moved out of cudaprocessingcore to common lib
	template <class DstSample, class SrcSample>
	void UploadCompatible(
		const image::PinnedImageView<SrcSample>& in_cpu,
		image::GpuImageView<DstSample> out_gpu,
		cudaStream_t stream)
	{
		static_assert(IsUploadCompatible<DstSample, SrcSample>::value, "Source and destination sample representations are incompatible");
		static_assert(sizeof(DstSample) == sizeof(SrcSample), "Src/Dst sample sizes must match for upload");

		if (in_cpu.m_dim.x != out_gpu.m_dim.x || in_cpu.m_dim.y != out_gpu.m_dim.y) {
			throw std::logic_error("UploadCompatible: trying to upload an image with wrong dim");
		}

		const size_t rowBytes = in_cpu.m_dim.x * sizeof(SrcSample);
		if (in_cpu.m_pitch < rowBytes || out_gpu.m_pitch < rowBytes) {
			throw std::logic_error("UploadCompatible: pitch is smaller than row size");
		}

		cudaCheck(cudaMemcpy2DAsync(
			out_gpu.m_ptr,
			out_gpu.m_pitch,
			in_cpu.m_ptr,
			in_cpu.m_pitch,
			in_cpu.m_dim.x * sizeof(SrcSample),
			in_cpu.m_dim.y,
			cudaMemcpyHostToDevice,
			stream
		));
	}
}

namespace pipeline {
	UploadNode::UploadNode(NodeId id, const UploadNodeParams& params)
		: NodeBase(id, params.listener)
		, m_cpuInput(id, params.cpuInputName, params.registry, params.resRegistry, params.cpuInputDesc)
		, m_gpuOutput(id, params.gpuOutputName, params.registry, params.resRegistry, params.gpuOutputDesc)
	{
	}

	void UploadNode::Execute(const CudaExecutionContext& ctx) {
		UploadCompatible(m_cpuInput.View(), m_gpuOutput.View(), ctx.stream);
	}
}
