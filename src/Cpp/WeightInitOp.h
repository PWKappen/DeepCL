#pragma once

#include <random>

#include "Defines.h"
#include "OpenCLBackend.h"
#include "NNBuffer.h"

namespace DeepCL
{
	namespace NNSystem
	{
		//Base class from which all other operations derive, which are used ot inialize the parameter buffers.
		class InitOp
		{
		public: 
			InitOp(const NNBufferIdx w) : w(w)
			{}

			virtual ~InitOp()
			{delete[] data;}

			//Is called when the hardware buffers where created. It creates the data and loads it into the hardware buffer.
			virtual void Instantiate(const std::vector<NNBuffer*>& bufferList, BackendSystem::OpenCLBackend& backend) = 0;
		protected:
			NNBufferIdx w;
			float* data;

			//Almost all initalize operations use random numbers
			static std::default_random_engine rnd;
			static const size_t seed;
		};

		class InitUniformRnd : public InitOp
		{
		public:
			InitUniformRnd(const NNBufferIdx w, const float minValue, const float maxValue):
				InitOp(w), minValue(minValue), maxValue(maxValue), distribution(minValue, maxValue)
			{}

			virtual ~InitUniformRnd()
			{InitOp::~InitOp();}

			virtual void Instantiate(const std::vector<NNBuffer*>& bufferList, BackendSystem::OpenCLBackend& backend);

		protected:
			const float minValue;
			const float maxValue;
			std::uniform_real_distribution<float> distribution;
		};

		class InitUniform : public InitOp
		{
		public:
			InitUniform(const NNBufferIdx w, const float value) :
				InitOp(w), value(value)
			{}

			virtual ~InitUniform()
			{InitOp::~InitOp();}

			virtual void Instantiate(const std::vector<NNBuffer*>& bufferList, BackendSystem::OpenCLBackend& backend);

		protected:
			const float value;
		};

		class InitNormalRnd : public InitOp
		{
		public:
			InitNormalRnd(const NNBufferIdx w, const float mean = 0.0f, const float stddev = 1.0f) :
				InitOp(w), mean(mean), stddev(stddev), distribution(mean, stddev)
			{}

			virtual ~InitNormalRnd()
			{InitOp::~InitOp();}

			virtual void Instantiate(const std::vector<NNBuffer*>& bufferList, BackendSystem::OpenCLBackend& backend);

		protected:
			const float mean;
			const float stddev;
			std::normal_distribution<float> distribution;
		};

		class InitTruncatedNormalRnd : public InitOp
		{
		public:
			InitTruncatedNormalRnd(const NNBufferIdx w, const float mean = 0.0, const float stddev = 1.0):
				InitOp(w), mean(mean), stddev(stddev), distribution(mean, stddev)
			{}

			virtual ~InitTruncatedNormalRnd()
			{InitOp::~InitOp();}

			virtual void Instantiate(const std::vector<NNBuffer*>& bufferList, BackendSystem::OpenCLBackend& backend);

		protected:
			const float mean;
			const float stddev;
			std::normal_distribution<float> distribution;
		};

		class InitTruncatedNormalXavier : public InitOp
		{
		public:
			InitTruncatedNormalXavier(const NNBufferIdx w, const float mean = 0.0):
				InitOp(w), mean(mean)
			{}

				virtual ~InitTruncatedNormalXavier()
			{
				InitOp::~InitOp();
			}

			virtual void Instantiate(const std::vector<NNBuffer*>& bufferList, BackendSystem::OpenCLBackend& backend);

		protected:
			const float mean;
		};
	}
}