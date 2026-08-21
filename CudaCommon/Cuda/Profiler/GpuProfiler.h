#pragma once
#include "IGpuProfiler.h"
#include "GpuEvent.h"
#include "GpuProfileResult.h"
#include <map>
#include <vector>

namespace cuda::profile {
	class BasicGpuProfiler {
	private:
		using SessionId = ProfileSampleId;

		struct KernelSample {
			GpuEvent start;
			GpuEvent stop;

			bool ended = false;
			bool canceled = false;
		};

		struct SessionInfo {
			std::string scope;

			std::string kernelName;
			std::string kernelLabel;

			std::vector<KernelSample> samples;
			bool finished = false;
		};

	public:
		class ProfileSession : public IGpuProfiler {
		public:
			ProfileSession(SessionId id, BasicGpuProfiler& parent);
			~ProfileSession() noexcept;

			ProfileSession(const ProfileSession&) = delete;
			ProfileSession& operator=(const ProfileSession&) = delete;

			ProfileSession(ProfileSession&&) = delete;
			ProfileSession& operator=(ProfileSession&&) = delete;

			ProfileSampleId BeginSample(const std::string& kernelName, cudaStream_t stream) override;
			void EndSample(ProfileSampleId id, cudaStream_t stream) override;

			void CancelSample(ProfileSampleId id) override;

		private:
			const SessionId m_id;
			BasicGpuProfiler& m_parent;
		};

		ProfileSession CreateSession(const std::string& scope, const std::string& kernelLabel);

		std::vector<GpuProfileResult> GetResults() const;
		void Clear();

	private:
		ProfileSampleId BeginSample(SessionId id, const std::string& kernelName, cudaStream_t stream);
		void EndSample(SessionId sessionId, ProfileSampleId sampleId, cudaStream_t stream);

		void CancelSample(SessionId sessionId, ProfileSampleId sampleId);

		void FinishSession(SessionId id) noexcept;

	private:
		SessionId m_nextSessionId = 0;
		std::map<SessionId, SessionInfo> m_sessionsById;
	};
}
