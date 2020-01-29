#include "NNOptimizer.h"

namespace DeepCL
{
	namespace NNSystem
	{

		NNOptimizer::NNOptimizer()
		{
		}


		NNOptimizer::~NNOptimizer()
		{
		}

		size_t NNOptimizer::GetNumBuffer() const
		{
			return 0;
		}

		void NNGradientDescent::Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList, NNBufferIdx weightBuffer)
		{
			KernelIdx kernel = backend.GetKernelIdx("GradientDecent");
			NNBuffer* buffer = bufferList[weightBuffer];
			size_t totalSize = buffer->size.sizeW*buffer->size.sizeZ*buffer->size.sizeY*buffer->size.sizeX;

			const int WORK_GROUP_SIZE_X = 64;


			Tuple<BufferIdx, BufferIdx, std::pair<size_t, float>, dataPair> tuple(buffer->BackwardBuffer(), buffer->ForwardBuffer(),
				std::pair<size_t, float>(sizeof(float), alpha), dataPair(sizeof(int), totalSize));


			OperationIdx  matOp = backend.AddOperation<4, BufferIdx, BufferIdx, std::pair<size_t, float>, dataPair>(kernel, tuple, cl::NullRange, cl::NDRange((totalSize + WORK_GROUP_SIZE_X - (totalSize%WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::UPDATE);
		}

		size_t NNAdam::GetNumBuffer() const
		{
			return 2;
		}

		void NNAdam::Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList, NNBufferIdx weightBuffer)
		{
			KernelIdx kernel = backend.GetKernelIdx("Adam");
			NNBuffer* buffer = bufferList[weightBuffer];
			size_t totalSize = buffer->size.sizeW*buffer->size.sizeZ*buffer->size.sizeY*buffer->size.sizeX;

			const int WORK_GROUP_SIZE_X = 64;
			//The ForwardBuffer of the weightBuffer contains the auxilary buffers at specific time step. (ForwardBuffer(0) contains the first auxilary buffer).
			Tuple<BufferIdx, BufferIdx, BufferIdx, BufferIdx, std::pair<size_t, float>, std::pair<size_t, float>, std::pair<size_t, float>, std::pair<size_t, float>, dataPair, dataPair> tuple(buffer->BackwardBuffer(), buffer->ForwardBuffer(0), buffer->ForwardBuffer(1), buffer->ForwardBuffer(),
				std::pair<size_t, float>(sizeof(float), alpha), std::pair<size_t, float>(sizeof(float), beta1), std::pair<size_t, float>(sizeof(float), beta2), std::pair<size_t, float>(sizeof(float), epsilon), dataPair(sizeof(int), totalSize), dataPair(sizeof(int), 1));


			OperationIdx  matOp = backend.AddOperation<9, 10, BufferIdx, BufferIdx, BufferIdx, BufferIdx, std::pair<size_t, float>, std::pair<size_t, float>, std::pair<size_t, float>, std::pair<size_t, float>, dataPair, dataPair>(kernel, tuple, cl::NullRange, cl::NDRange((totalSize + WORK_GROUP_SIZE_X - (totalSize%WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::UPDATE);
			ops.push_back(matOp);
		}
	}
}