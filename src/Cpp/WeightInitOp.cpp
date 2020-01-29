#include "WeightInitOp.h"

namespace DeepCL
{
	namespace NNSystem
	{
		//Seed for the rng
		const size_t InitOp::seed = 99;
		std::default_random_engine InitOp::rnd(InitOp::seed);

		//All Instantiate functions work essentially the same. They sample the data from some distribution and transfer it into the buffer.
		void InitUniformRnd::Instantiate(const std::vector<NNBuffer*>& bufferList, BackendSystem::OpenCLBackend& backend)
		{
			NNBuffer* buffer = bufferList[w];

			size_t size = buffer->size.sizeW * buffer->size.sizeX * buffer->size.sizeY * buffer->size.sizeZ;

			float* data = new float[size];

			//Sample different randum numbers for each element of the parameter buffer
			for (size_t i = 0; i < size; ++i)
				data[i] = distribution(rnd);

			//Transfer the sampled data into the OpenCL buffer.
			backend.WriteDataBuffer(buffer->ForwardBuffer(), data, 0, size * sizeof(float));
			delete[] data;
		}

		void InitUniform::Instantiate(const std::vector<NNBuffer*>& bufferList, BackendSystem::OpenCLBackend& backend)
		{
			NNBuffer* buffer = bufferList[w];

			size_t size = buffer->size.sizeW * buffer->size.sizeX * buffer->size.sizeY * buffer->size.sizeZ;

			float* data = new float[size];

			for (size_t i = 0; i < size; ++i)
				data[i] = value;

			backend.WriteDataBuffer(buffer->ForwardBuffer(), data, 0, size * sizeof(float));
			delete[] data;
		}

		void InitNormalRnd::Instantiate(const std::vector<NNBuffer*>& bufferList, BackendSystem::OpenCLBackend& backend)
		{
			NNBuffer* buffer = bufferList[w];

			size_t size = buffer->size.sizeW * buffer->size.sizeX * buffer->size.sizeY * buffer->size.sizeZ;

			float* data = new float[size];

			float tmpValue;

			for (size_t i = 0; i < size; ++i)
			{
				tmpValue = distribution(rnd);
				data[i] = tmpValue;
			}

			backend.WriteDataBuffer(buffer->ForwardBuffer(), data, 0, size * sizeof(float));
			delete[] data;
		}

		void InitTruncatedNormalRnd::Instantiate(const std::vector<NNBuffer*>& bufferList, BackendSystem::OpenCLBackend& backend)
		{
			NNBuffer* buffer = bufferList[w];

			size_t size = buffer->size.sizeW * buffer->size.sizeX * buffer->size.sizeY * buffer->size.sizeZ;

			float* data = new float[size];

			float tmpValue;

			for (size_t i = 0; i < size; ++i)
			{
				tmpValue = distribution(rnd);
				while ((tmpValue - mean) > 2.f * stddev && (tmpValue - mean) < -2.f * stddev)
					tmpValue = distribution(rnd);
				data[i] = tmpValue;

			}

			backend.WriteDataBuffer(buffer->ForwardBuffer(), data, 0, size * sizeof(float));
			delete[] data;
		}

		void InitTruncatedNormalXavier::Instantiate(const std::vector<NNBuffer*>& bufferList, BackendSystem::OpenCLBackend& backend)
		{
			NNBuffer* buffer = bufferList[w];

			size_t size = buffer->size.sizeW * buffer->size.sizeX * buffer->size.sizeY * buffer->size.sizeZ;

			float stddev = sqrt(2.f / static_cast<float>(buffer->size.sizeY));
			std::normal_distribution<float> distribution(mean, stddev);

			float* data = new float[size];

			float tmpValue;

			for (size_t i = 0; i < size; ++i)
			{
				tmpValue = distribution(rnd);
				while ((tmpValue - mean) > 2.f * stddev && (tmpValue - mean) < -2.f * stddev)
					tmpValue = distribution(rnd);
				data[i] = tmpValue;

			}

			backend.WriteDataBuffer(buffer->ForwardBuffer(), data, 0, size * sizeof(float));
			delete[] data;
		}
	}
}