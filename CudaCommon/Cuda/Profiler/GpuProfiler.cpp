#include "GpuProfiler.h"
#include "../CudaCheck.h"
#include <cassert>

namespace cuda::profile {
	BasicGpuProfiler::ProfileSession BasicGpuProfiler::CreateSession(const std::string& scope, const std::string& kernelLabel) {
		m_sessionsById.emplace(m_nextSessionId, SessionInfo{ scope, std::string{}, kernelLabel, {} });
		return ProfileSession(m_nextSessionId++, *this);
	}

	ProfileSampleId BasicGpuProfiler::BeginSample(SessionId id, const std::string& kernelName, cudaStream_t stream) {
		const auto it = m_sessionsById.find(id);

		assert(it != m_sessionsById.end());
		assert(!kernelName.empty());

		auto& info = it->second;
		if (info.kernelName.empty()) {
			info.kernelName = kernelName;
		}
		else if (info.kernelName != kernelName) {
			throw std::logic_error("BasicGpuProfiler::BeginSample: invalid usage of session for more than one kernel");
		}

		info.samples.emplace_back();
		cudaCheck(cudaEventRecord(info.samples.back().start.Get(), stream));

		return info.samples.size() - 1;
	}

	void BasicGpuProfiler::EndSample(SessionId sessionId, ProfileSampleId sampleId, cudaStream_t stream) {
		const auto it = m_sessionsById.find(sessionId);
		assert(it != m_sessionsById.end());

		auto& info = it->second;
		assert(sampleId < info.samples.size());

		auto& sample = info.samples.at(sampleId);

		if (sample.ended) {
			throw std::logic_error("BasicGpuProfiler::EndSample: sample already ended");
		}

		cudaCheck(cudaEventRecord(sample.stop.Get(), stream));
		sample.ended = true;
	}

	void BasicGpuProfiler::CancelSample(SessionId sessionId, ProfileSampleId sampleId) {
		const auto it = m_sessionsById.find(sessionId);
		assert(it != m_sessionsById.end());

		auto& info = it->second;
		assert(sampleId < info.samples.size());

		info.samples.at(sampleId).canceled = true;
	}

	void BasicGpuProfiler::FinishSession(SessionId id) noexcept {
		const auto it = m_sessionsById.find(id);
		assert(it != m_sessionsById.end());

		if (it != m_sessionsById.end()) {
			it->second.finished = true;
		}
	}

	std::vector<GpuProfileResult> BasicGpuProfiler::GetResults() const {
		std::vector<GpuProfileResult> results;
		results.reserve(m_sessionsById.size());

		for (auto& [id, info] : m_sessionsById) {
			assert(info.finished);

			GpuProfileResult result {
				.scope = info.scope,
				.kernelName = info.kernelName + info.kernelLabel
			};

			for (auto& sample : info.samples) {
				if (sample.canceled) {
					continue;
				}

				assert(sample.ended);

				const auto status = cudaEventQuery(sample.stop.Get());

				if (status == cudaErrorNotReady) {
					throw std::logic_error("SimpleGpuProfiler::GetResults: GPU execution has not been completed");
				}

				cudaCheck(status);

				float milliseconds = 0.0f;
				cudaCheck(cudaEventElapsedTime(&milliseconds, sample.start.Get(), sample.stop.Get()));

				result.sampleDurationsMs.push_back(milliseconds);
			}

			results.emplace_back(std::move(result));
		}

		return results;
	}

	void BasicGpuProfiler::Clear() {
		m_nextSessionId = 0;
		m_sessionsById.clear();
	}

	BasicGpuProfiler::ProfileSession::ProfileSession(SessionId id, BasicGpuProfiler& parent)
		: m_id{ id }
		, m_parent{ parent }
	{
	}

	BasicGpuProfiler::ProfileSession::~ProfileSession() noexcept{
		m_parent.FinishSession(m_id);
	}

	ProfileSampleId BasicGpuProfiler::ProfileSession::BeginSample(const std::string& kernelName, cudaStream_t stream) {
		return m_parent.BeginSample(m_id, kernelName, stream);
	}

	void BasicGpuProfiler::ProfileSession::EndSample(ProfileSampleId id, cudaStream_t stream) {
		m_parent.EndSample(m_id, id, stream);
	}

	void BasicGpuProfiler::ProfileSession::CancelSample(ProfileSampleId id) {
		m_parent.CancelSample(m_id, id);
	}
}
