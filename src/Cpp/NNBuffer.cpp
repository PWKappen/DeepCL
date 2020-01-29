#include "NNBuffer.h"


namespace DeepCL
{
	namespace NNSystem
	{
		size_t NNParamBuffer::numAuxBuffer = 0;

		void NNBuffer::SetBatchSize(const size_t batchSize)
		{
			size.sizeW = batchSize;
		}

		void NNBuffer::Reset(BackendSystem::OpenCLBackend& backend)
		{
			//Sets all backward buffers to zero
			for (size_t i = 0; i < backwardBuffer.size(); ++i)
				backend.ResetBuffer(backwardBuffer[i], size.sizeX * size.sizeY * size.sizeZ * size.sizeW * sizeof(float));
		}

		void NNInputBuffer::Instantiate(BackendSystem::OpenCLBackend& backend)
		{
			size_t totalSize = size.sizeW*size.sizeZ*size.sizeY*size.sizeX * sizeof(float);

			//Create the necessary forward and backward buffer
			BufferIdx newForwardBuffer = backend.CreateBuffer(totalSize, BackendSystem::MEM_FLAG::READ_ONLY, sequenceSize);
			baseFwdBuffer = newForwardBuffer;

			BufferIdx newBackwardBuffer = backend.CreateBuffer(totalSize, BackendSystem::MEM_FLAG::READ_WRITE, sequenceSize);
			baseBwdBuffer = newBackwardBuffer;

			//Create for each time step one sub buffer
			for (size_t j = 0; j < sequenceSize; ++j)
			{
				BufferIdx newBuffer = backend.CreateSubBuffer(baseFwdBuffer, totalSize, BackendSystem::MEM_FLAG::READ_ONLY, j);
				forwardBuffer[j] = newBuffer;

				newBuffer = backend.CreateSubBuffer(baseBwdBuffer, totalSize, BackendSystem::MEM_FLAG::READ_WRITE, j);
				backwardBuffer[j] = newBuffer;
			}
		}

		void NNIntBuffer::Instantiate(BackendSystem::OpenCLBackend& backend)
		{
			size_t totalSize = size.sizeW*size.sizeZ*size.sizeY*size.sizeX * sizeof(float);

			//Create the necessary forward and backward buffer
			BufferIdx newForwardBuffer = backend.CreateBuffer(totalSize, BackendSystem::MEM_FLAG::READ_WRITE, sequenceSize);
			baseFwdBuffer = newForwardBuffer;

			BufferIdx newBackwardBuffer = backend.CreateBuffer(totalSize, BackendSystem::MEM_FLAG::READ_WRITE, sequenceSize);
			baseBwdBuffer = newBackwardBuffer;

			//Create for each time step one sub buffer
			for (size_t j = 0; j < sequenceSize; ++j)
			{
				BufferIdx newBuffer = backend.CreateSubBuffer(baseFwdBuffer, totalSize, BackendSystem::MEM_FLAG::READ_WRITE, j);
				forwardBuffer[j] = newBuffer;

				newBuffer = backend.CreateSubBuffer(baseBwdBuffer, totalSize, BackendSystem::MEM_FLAG::READ_WRITE, j);
				backwardBuffer[j] = newBuffer;
			}
		}

		void NNParamBuffer::SetNumAuxBuffer(const size_t numAuxBuffer)
		{
			NNParamBuffer::numAuxBuffer = numAuxBuffer;
		}

		void NNParamBuffer::Instantiate(BackendSystem::OpenCLBackend& backend)
		{
			size_t totalSize = size.sizeW*size.sizeZ*size.sizeY*size.sizeX * sizeof(float);

			//Create the necessary forward and backward buffer
			//The sub buffers are created to stay consistend with the rest of the buffers.
			BufferIdx newBuffer = backend.CreateBuffer(totalSize, BackendSystem::MEM_FLAG::READ_WRITE, 1);
			baseFwdBuffer = newBuffer;
			forwardBuffer[0] = backend.CreateSubBuffer(baseFwdBuffer, totalSize, BackendSystem::MEM_FLAG::READ_WRITE, 0);
			

			newBuffer = backend.CreateBuffer(totalSize, BackendSystem::MEM_FLAG::READ_WRITE, 1);
			baseBwdBuffer = newBuffer;
			backwardBuffer[0] = backend.CreateSubBuffer(baseBwdBuffer, totalSize, BackendSystem::MEM_FLAG::READ_WRITE, 0);;
			
			//Create the necessary auxilary buffers and store them in the auxBuffer vector.
			for (size_t j = 0; j < numAuxBuffer; ++j)
			{
				BufferIdx newBuffer = backend.CreateBuffer(totalSize, BackendSystem::MEM_FLAG::READ_WRITE, 1);
				auxBuffer.push_back(newBuffer);
			}
		}

		void NNStateBuffer::Instantiate(BackendSystem::OpenCLBackend& backend)
		{
			size_t totalSize = size.sizeW*size.sizeZ*size.sizeY*size.sizeX * sizeof(float);

			//Creates base buffers that can contain an additioanl sub buffer
			BufferIdx newForwardBuffer = backend.CreateBuffer(totalSize, BackendSystem::MEM_FLAG::READ_WRITE, sequenceSize + 1);
			baseFwdBuffer = newForwardBuffer;

			BufferIdx newBackwardBuffer = backend.CreateBuffer(totalSize, BackendSystem::MEM_FLAG::READ_WRITE, sequenceSize + 1);
			baseBwdBuffer = newBackwardBuffer;

			//Create all necessary sub buffers.
			for (size_t j = 0; j < sequenceSize + 1; ++j)
			{
				BufferIdx newBuffer = backend.CreateSubBuffer(baseFwdBuffer, totalSize, BackendSystem::MEM_FLAG::READ_WRITE, j);
				forwardBuffer[j] = newBuffer;

				newBuffer = backend.CreateSubBuffer(baseBwdBuffer, totalSize, BackendSystem::MEM_FLAG::READ_WRITE, j);
				backwardBuffer[j] = newBuffer;
			}
		}

		//Resets all sub buffers.
		void NNStateBuffer::Reset(BackendSystem::OpenCLBackend& backend)
		{
			for (size_t i = 0; i < backwardBuffer.size(); ++i)
				backend.ResetBuffer(backwardBuffer[i], size.sizeX * size.sizeY * size.sizeZ * size.sizeW * sizeof(float));
			for (size_t i = 0; i < forwardBuffer.size(); ++i)
				backend.ResetBuffer(forwardBuffer[i], size.sizeX * size.sizeY * size.sizeZ * size.sizeW * sizeof(float));
		}
		
		void NNTmpBuffer::Instantiate(BackendSystem::OpenCLBackend& backend)
		{
			size_t totalSize = size.sizeW*size.sizeZ*size.sizeY*size.sizeX * sizeof(float);

			//Needs only one forward Buffer, since they are only used to store intermediate results and don't need to store data for another pass. 
			BufferIdx newForwardBuffer = backend.CreateBuffer(totalSize, BackendSystem::MEM_FLAG::READ_WRITE, 1);
			baseFwdBuffer = newForwardBuffer;
			forwardBuffer[0] = backend.CreateSubBuffer(baseFwdBuffer, totalSize, BackendSystem::MEM_FLAG::READ_WRITE, 0);;
		}
	}
}