#pragma once

#include "OpenCLBackend.h"

//Contains different objects which represent the edges in the computation graph.
//Those objects represent data that is transfered between operations

namespace DeepCL
{
	namespace NNSystem
	{

		//Used to store the size of an buffer and to pass a size as parameter.
		//The number of time steps is not included in this vector.
		struct SizeVec
		{
		public:
			size_t sizeX;
			size_t sizeY;
			size_t sizeZ;
			size_t sizeW;

			SizeVec() : sizeX(0), sizeY(0), sizeZ(0), sizeW(0)
			{}

			SizeVec(const size_t sizeX, const size_t sizeY = 1, const size_t sizeZ = 1, const size_t sizeW = 1) :
				sizeX(sizeX), sizeY(sizeY), sizeZ(sizeZ), sizeW(sizeW)
			{}

			SizeVec Max(const SizeVec& other)
			{
				SizeVec result = other;
				result.sizeX = other.sizeX >= sizeX ? other.sizeX : sizeX;
				result.sizeY = other.sizeY >= sizeY ? other.sizeY : sizeY;
				result.sizeZ = other.sizeZ >= sizeZ ? other.sizeZ : sizeZ;
				result.sizeW = other.sizeW >= sizeW ? other.sizeW : sizeW;

				return result;
			}
		};

		//Basic buffer object from which all other buffer objects derive.
		class NNBuffer
		{
		public:
			//The operation that leads calculates this buffer.
			NNOperationIdx from;

			//An operation that uses this buffer. (Should be adapted to be a vector since multiple operations can use this buffer due to implicit copying).
			NNOperationIdx to;

			//The size of this buffer.
			SizeVec size;

			//The time offset for when this buffer is used
			size_t timeOffset;

			//The length of the sequence
			const size_t sequenceSize;

			//All indices that are not set to a specific real object are set to the maximal possible unsigned integer, thereby hinting at that the index is uninialized.
			NNBuffer(size_t sizeX, size_t sizeY, size_t sizeZ, size_t sizeW, const size_t sequenceSize = 1, const size_t timeOffset = 0) :
				size(sizeX, sizeY, sizeZ, sizeW), sequenceSize(sequenceSize), timeStep(0), forwardBuffer(), backwardBuffer(),
				from(MAX_UNSIGNED_INT), to(MAX_UNSIGNED_INT), timeOffset(timeOffset)
			{
				//This buffer contains a indices on a hardware sub buffer for each time step in forward and backward direction.
				for (size_t i = 0; i < sequenceSize; ++i)
				{
					forwardBuffer.push_back(MAX_UNSIGNED_INT);
					backwardBuffer.push_back(MAX_UNSIGNED_INT);
				}
			}

			NNBuffer(const NNBuffer &other) :
				size(other.size), forwardBuffer(other.forwardBuffer), backwardBuffer(other.backwardBuffer),
				from(other.from), to(other.to), sequenceSize(other.sequenceSize), timeStep(other.timeStep), timeOffset(other.timeOffset), baseFwdBuffer(other.baseFwdBuffer), baseBwdBuffer(other.baseBwdBuffer)
			{}

			const NNBuffer& operator=(const NNBuffer& other)
			{
				forwardBuffer = other.forwardBuffer;
				backwardBuffer = other.backwardBuffer;
				from = other.from;
				to = other.to;

				baseFwdBuffer = other.baseFwdBuffer;
				baseBwdBuffer = other.baseBwdBuffer;

				size = other.size;
				timeStep = other.timeStep;
				timeOffset = other.timeOffset;

				return *this;
			}

			//Returns a specific forward/backward sub buffer indice. ForwardBuffer is virtual since some buffers require a specific access.
			virtual inline BufferIdx& ForwardBuffer(const size_t num){ return forwardBuffer[num]; }
			inline BufferIdx& BackwardBuffer(const size_t num){ return backwardBuffer[num]; }

			//Returns the forward/backward sub buffer of the current time step. This way the operation don't needs any explicit information about the time step since only the buffer need this information
			inline BufferIdx& ForwardBuffer(){ return forwardBuffer[timeStep]; }
			inline BufferIdx& BackwardBuffer(){ return backwardBuffer[timeStep]; }

			inline void UpdateStep(){ timeStep = (timeStep + 1) % sequenceSize; } //Increases the current time step by one thereby returning the sub buffer for the next time step when ForwardBuffer/BackwardBuffer is called.
			inline void Reset(){ timeStep = 0; };

			//Returns references on the vector of forward/backward buffers.
			inline std::vector<BufferIdx>& GetCompleteBackwardBuffer(){ return backwardBuffer; }
			inline std::vector<BufferIdx>& GetCompleteForwardBuffer(){ return forwardBuffer; }

			//Returns the current time step.
			inline size_t GetCurTimeStep() const { return timeStep; }

			//Returns the base hardware buffer which contains all sub buffers.
			inline BufferIdx BaseForwardBuffer() const { return baseFwdBuffer; }
			inline BufferIdx BaseBackwardBuffer() const { return baseBwdBuffer; }

			//Sets the base hardware buffer.
			inline void BaseFwdBuffer(BufferIdx fwdBuffer) { baseFwdBuffer = fwdBuffer; }
			inline void BaseBwdBuffer(BufferIdx bwdBuffer) { baseBwdBuffer = bwdBuffer; }

			//Creates the necessary hardware buffer and sub buffers.
			virtual void Instantiate(BackendSystem::OpenCLBackend& backend){}

			//Sets the w component of the size to be equal to the batchSize
			virtual void SetBatchSize(const size_t batchSize);
			
			//Sets each backward buffer to zero. (Not all buffers need to do this)
			virtual void Reset(BackendSystem::OpenCLBackend& backend);


		protected:
			//Stores the indices on the hardware buffer that contains all sub buffers
			BufferIdx baseFwdBuffer; 
			BufferIdx baseBwdBuffer;

			//Stores the indices on all hardware sub buffers.
			std::vector<BufferIdx> forwardBuffer;
			std::vector<BufferIdx> backwardBuffer;

			//Stores the current time step.
			size_t timeStep;

		};

		class NNInputBuffer : public NNBuffer
		{
		public:
			NNInputBuffer(size_t sizeX, size_t sizeY, size_t sizeZ, size_t sizeW, const size_t sequenceSize = 1, const size_t timeOffset = 0) :
				NNBuffer(sizeX, sizeY, sizeZ, sizeW, sequenceSize, timeOffset)
			{}

			NNInputBuffer(const NNInputBuffer& other) :
				NNBuffer(other)
			{}

			const NNInputBuffer& operator=(const NNInputBuffer& other)
			{
				NNBuffer::operator=(other);

				return *this;
			}
			//Specific memory access. Forward input only needs read access
			virtual void Instantiate(BackendSystem::OpenCLBackend& backend);
		};

		class NNIntBuffer : public NNBuffer
		{
		public:
			NNIntBuffer(size_t sizeX, size_t sizeY, size_t sizeZ, size_t sizeW, const size_t sequenceSize = 1, const size_t timeOffset = 0) :
				NNBuffer(sizeX, sizeY, sizeZ, sizeW, sequenceSize, timeOffset)
			{}

			NNIntBuffer(const NNIntBuffer& other) :
				NNBuffer(other)
			{}

			const NNIntBuffer& operator=(const NNIntBuffer& other)
			{
				NNBuffer::operator=(other);

				return *this;
			}
			//All buffers have read write access.
			virtual void Instantiate(BackendSystem::OpenCLBackend& backend);
		};

		class NNParamBuffer : public NNBuffer
		{
		public:
			NNParamBuffer(size_t sizeX, size_t sizeY, size_t sizeZ, size_t sizeW, const size_t sequenceSize = 1, const size_t timeOffset = 0) :
				NNBuffer(sizeX, sizeY, sizeZ, sizeW, sequenceSize, timeOffset)
			{}

			NNParamBuffer(const NNParamBuffer& other) :
				NNBuffer(other)
			{}

			const NNParamBuffer& operator=(const NNParamBuffer& other)
			{
				NNBuffer::operator=(other);

				return *this;
			}

			//This function is used to access auxilary buffers. Since parameter buffers otherwise only contain a buffer for one time step.
			virtual inline BufferIdx& ForwardBuffer(const size_t num)
			{
				if (num < numAuxBuffer)
					return auxBuffer[num];

				std::cerr << "ERROR in GetAuxBuffer! num to high." << std::endl;
				return	baseFwdBuffer;
			}

			//Used to set the number of auxilary buffers. All parameter buffers need the same number of auxilary buffers therefore this function is static.
			static void SetNumAuxBuffer(const size_t numAuxBuffer);

			//This function needs the number of auxilary buffers to be set. It creates a base forward buffer that can contain the normal forward.
			//Only the base backward buffer is created since only the normal buffer has a backward component.
			//The auxilary buffers are its own buffers and not hardware buffers.
			virtual void Instantiate(BackendSystem::OpenCLBackend& backend);

			//The parameter buffers are independent of the batch size.
			virtual void SetBatchSize(const size_t batchSize)
			{}

			virtual BufferIdx GetAuxBuffer(const size_t num)
			{
				return auxBuffer[num];
			}

		private:
			static size_t numAuxBuffer;
			//Stores the indices of the hardware auxilary buffers.
			std::vector<BufferIdx> auxBuffer;
		};

		class NNStateBuffer : public NNBuffer
		{
		public:
			NNStateBuffer(size_t sizeX, size_t sizeY, size_t sizeZ, size_t sizeW, const size_t sequenceSize = 1, const size_t timeOffset = 0) :
				NNBuffer(sizeX, sizeY, sizeZ, sizeW, sequenceSize, timeOffset)
			{
				//The state needs one additional buffer since when the output is computed, the next state is also computed, Thereby one more state is computed. The state at time step zero is often zero but there might be situations where it is initalized different from zero.
				//Therefore the total number of sub buffers needed is the sequence size plus one.
				forwardBuffer.push_back(MAX_UNSIGNED_INT);
				backwardBuffer.push_back(MAX_UNSIGNED_INT);
			}

			NNStateBuffer(const NNStateBuffer& other) :
				NNBuffer(other)
			{}

			const NNStateBuffer& operator=(const NNStateBuffer& other)
			{
				NNBuffer::operator=(other);

				return *this;
			}
			//It is the same as intermediate buffer but with an additional sub buffer for both passes
			virtual void Instantiate(BackendSystem::OpenCLBackend& backend);
			//Sets all buffers to zero. (Forward and backward)
			virtual void Reset(BackendSystem::OpenCLBackend& backend);
		};

		class NNTmpBuffer : public NNBuffer
		{
		public:
			NNTmpBuffer(size_t sizeX, size_t sizeY, size_t sizeZ, size_t sizeW, const size_t sequenceSize = 1, const size_t timeOffset = 0) :
				NNBuffer(sizeX, sizeY, sizeZ, sizeW, sequenceSize, timeOffset)
			{
			}

			NNTmpBuffer(const NNTmpBuffer& other) :
				NNBuffer(other)
			{}

			const NNTmpBuffer& operator=(const NNTmpBuffer& other)
			{
				NNBuffer::operator=(other);

				return *this;
			}
			//Nothing needs to be done when reset is called
			virtual void Reset(BackendSystem::OpenCLBackend& backend)
			{}

			//Creates only one sub buffer.
			virtual void Instantiate(BackendSystem::OpenCLBackend& backend);

			//Batch size is already contained in its size when it is created.
			virtual void SetBatchSize(const size_t batchSize)
			{}
		};
	}
}
