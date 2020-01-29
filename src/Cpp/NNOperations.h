#pragma once

#include <type_traits>

#include "NNBuffer.h"


namespace DeepCL
{
	namespace NNSystem
	{
		//All normal operations derive from this class.
		//Parameter initalize operations and optimizers are handled seperately.
		
		class NNOp
		{
		public:
			std::vector<OperationIdx> forwardOpIdx;//The hardware operations needed in the forward pass of this operation
			std::vector<OperationIdx> backwardOpIdx;//The hardware operations needed in the backward pass of this operation

			std::vector<BufferIdx> input;//The buffers, which are used as input to this operation
			std::vector<BufferIdx> output;//The buffers, which are store the computed result of this operation (In almost all cases only one)
			std::vector<BufferIdx> tmpBuffer;//The tempoaray buffers that can be used by this operation
			std::vector<SizeVec> tmpSizes;//The necessary sizes of each temporary buffer.
			size_t timeOffset;//The offset at this operation runs.
			size_t timeSteps;//The number of time steps this operation runs (Is currently determined by the output buffers)

#ifdef _DEBUG
			BufferIdx debugBuffer;
#endif

			NNOp() :forwardOpIdx(), backwardOpIdx(), input(), output(), tmpSizes(), tmpBuffer(), timeOffset(0) {};
			
			NNOp(const NNOp& other):
			forwardOpIdx(other.forwardOpIdx), backwardOpIdx(other.backwardOpIdx), input(other.input),
			output(other.output), tmpSizes(other.tmpSizes), tmpBuffer(other.tmpBuffer), timeOffset(other.timeOffset){};
			
			const NNOp& operator=(const NNOp& other)
			{
				input = other.input;
				output = other.output;

				tmpSizes = other.tmpSizes;
				tmpBuffer = other.tmpBuffer;
				forwardOpIdx = other.forwardOpIdx;
				backwardOpIdx = other.backwardOpIdx;

				timeOffset = other.timeOffset;

				return *this;
			}

			//Lets each function create itself
			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList) = 0;
			
			//The size of the output buffer.
			virtual SizeVec GetOutputType(std::vector<NNBuffer*>& bufferList) = 0;
			//The number of time steps the output buffer must be long given the input. Some operations receive only an input for one time step but generate outputs at multiple time steps.
			//(Example: seperating into windows)
			virtual size_t GetTimeTransform(std::vector<NNBuffer*>& bufferList);

			//Adds the correct from, to indices in the buffer objects used in this operation
			virtual void SetBuffer(std::vector<NNBuffer*>& bufferList, OperationIdx op);

			//Calculates the necessary size for each tmporary buffer. Requires the input sizes to be known
			virtual void SetTmpBuffer(std::vector<NNBuffer*>& bufferList, OperationIdx op) {}
		};

		class NNMatMulOp : public NNOp
		{
		public:
			NNMatMulOp(NNBufferIdx inputA, NNBufferIdx inputB) : NNOp()
			{
				input.push_back(inputA);
				input.push_back(inputB);
			}

			NNMatMulOp(const NNMatMulOp& other): NNOp(other)
			{
			}

			const NNMatMulOp& operator=(const NNMatMulOp& other)
			{
				NNOp::operator=(other);
				return *this;
			}

			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList);
		
			virtual SizeVec GetOutputType(std::vector<NNBuffer*>& bufferList);

			virtual void SetTmpBuffer(std::vector<NNBuffer*>& bufferList, OperationIdx op);
		};

		//Matrix multiplication that flattens all dimensions of the input except the batch size. Allows the application of matrix multiplication on filter volumes.
		class NNMatMulFlatOp : public NNOp
		{
		public:
			NNMatMulFlatOp(NNBufferIdx inputA, NNBufferIdx inputB) : NNOp()
			{
				input.push_back(inputA);
				input.push_back(inputB);
			}

			NNMatMulFlatOp(const NNMatMulFlatOp& other) : NNOp(other)
			{
			}

			const NNMatMulFlatOp& operator=(const NNMatMulFlatOp& other)
			{
				NNOp::operator=(other);
				return *this;
			}

			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList);

			virtual SizeVec GetOutputType(std::vector<NNBuffer*>& bufferList);

			virtual void SetTmpBuffer(std::vector<NNBuffer*>& bufferList, OperationIdx op);
		};

		class NNConvOp : public NNOp
		{
		public:

			enum ConvType
			{
				FULL, SAME, VALID
			};

			NNConvOp(NNBufferIdx inputA, NNBufferIdx inputB, ConvType convType, const size_t stride_x, const size_t stride_y) : NNOp(), convType(convType), pad(-1), strideX(stride_x), strideY(stride_y)
			{
				input.push_back(inputA);
				input.push_back(inputB);
			}

			NNConvOp(NNBufferIdx inputA, NNBufferIdx inputB, const int pad, const size_t stride_x, const size_t stride_y) : NNOp(), pad(pad), convType(ConvType::VALID), strideX(stride_x), strideY(stride_y)
			{
				input.push_back(inputA);
				input.push_back(inputB);
			}


			NNConvOp(const NNConvOp& other) : NNOp(other), pad(other.pad)
			{
			}

			const NNConvOp& operator=(const NNConvOp& other)
			{
				NNOp::operator=(other);
				pad = other.pad;
				return *this;
			}

			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList);

			virtual SizeVec GetOutputType(std::vector<NNBuffer*>& bufferList);

			virtual void SetTmpBuffer(std::vector<NNBuffer*>& bufferList, OperationIdx op);

			ConvType convType;
			int pad;
			size_t strideX;
			size_t strideY;
		};

		class NNCopyInitOp : public NNOp
		{
		public:
			NNCopyInitOp(NNBufferIdx inputA) :
				NNOp()
			{
				input.push_back(inputA);
			};

			NNCopyInitOp(const NNCopyInitOp& other) : NNOp(other)
			{
			}

			const NNCopyInitOp& operator=(const NNCopyInitOp& other)
			{
				NNOp::operator=(other);
				return *this;
			}

			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList);

			virtual SizeVec GetOutputType(std::vector<NNBuffer*>& bufferList);
		};

		class NNReLUOp : public NNOp
		{
		public:
			NNReLUOp(NNBufferIdx inputA) :
				NNOp()
			{
				input.push_back(inputA);
			};

			NNReLUOp(const NNReLUOp& other) : NNOp(other)
			{
			}

			const NNReLUOp& operator=(const NNReLUOp& other)
			{
				NNOp::operator=(other);
				return *this;
			}

			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList);

			virtual SizeVec GetOutputType(std::vector<NNBuffer*>& bufferList);
		};

		class NNTanhOp : public NNOp
		{
		public:
			NNTanhOp(NNBufferIdx inputA) :
				NNOp()
			{
				input.push_back(inputA);
			};

			NNTanhOp(const NNTanhOp& other) : NNOp(other)
			{
			}

			const NNTanhOp& operator=(const NNReLUOp& other)
			{
				NNOp::operator=(other);
				return *this;
			}

			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList);

			virtual SizeVec GetOutputType(std::vector<NNBuffer*>& bufferList);
		};

		class NNElemWiseProductOp : public NNOp
		{
		public:
			NNElemWiseProductOp(NNBufferIdx inputA, NNBufferIdx inputB) :
				NNOp()
			{
				input.push_back(inputA);
				input.push_back(inputB);
			};

			NNElemWiseProductOp(const NNElemWiseProductOp& other) : NNOp(other)
			{
			}

			const NNElemWiseProductOp& operator=(const NNElemWiseProductOp& other)
			{
				NNOp::operator=(other);
				return *this;
			}

			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList);

			virtual SizeVec GetOutputType(std::vector<NNBuffer*>& bufferList);
		};

		class NNSplitOp : public NNOp
		{
		public:
			NNSplitOp(NNBufferIdx inputA, const int w, const int h) :
				NNOp(), w(w), h(h)
			{
				input.push_back(inputA);
			};

			NNSplitOp(const NNSplitOp& other) : NNOp(other), w(other.w), h(other.h)
			{
			}

			const NNSplitOp& operator=(const NNSplitOp& other)
			{
				NNOp::operator=(other);
				w = other.w;
				h = other.h;
				return *this;
			}

			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList);

			virtual SizeVec GetOutputType(std::vector<NNBuffer*>& bufferList);
			virtual size_t GetTimeTransform(std::vector<NNBuffer*>& bufferList);
		
		private:
			int w, h;
		};

		class NNMaxPoolingOp : public NNOp
		{
		public:
			NNMaxPoolingOp(NNBufferIdx inputA, const int padX, const int padY, const int stride, const int size) :
				NNOp(), padX(padX), padY(padY), stride(stride), size(size)
			{
				input.push_back(inputA);
			};

			NNMaxPoolingOp(const NNMaxPoolingOp& other) : 
				NNOp(other), padX(other.padX), padY(other.padY), stride(other.stride), size(other.size)
			{
			}

			const NNMaxPoolingOp& operator=(const NNMaxPoolingOp& other)
			{
				NNOp::operator=(other);
				padX = other.padX;
				padY = other.padY;
				size = other.size;
				stride = other.stride;
				return *this;
			}

			int padX, padY, stride, size;

			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList);

			virtual SizeVec GetOutputType(std::vector<NNBuffer*>& bufferList);
		};

		class NNMatTransposeOp : public NNOp
		{
		public:
			NNMatTransposeOp(NNBufferIdx inputA) :
				NNOp()
			{
				input.push_back(inputA);
			};

			NNMatTransposeOp(const NNMatTransposeOp& other) :
				NNOp(other)
			{
			}

			const NNMatTransposeOp& operator=(const NNMatTransposeOp& other)
			{
				NNOp::operator=(other);

				return *this;
			}

			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList);

			virtual SizeVec GetOutputType(std::vector<NNBuffer*>& bufferList);
		};

		class NNLeastSquaresOp : public NNOp
		{
		public:
			NNLeastSquaresOp(NNBufferIdx inputA, NNBufferIdx labelY) :
				NNOp()
			{
				input.push_back(inputA);
				input.push_back(labelY);
			};

			NNLeastSquaresOp(const NNLeastSquaresOp& other) :
				NNOp(other)
			{
			}

			const NNLeastSquaresOp& operator=(const NNLeastSquaresOp& other)
			{
				NNOp::operator=(other);

				return *this;
			}

			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList);

			virtual SizeVec GetOutputType(std::vector<NNBuffer*>& bufferList);
		};

		class NNAddBiasOp : public NNOp
		{
		public:
			NNAddBiasOp(NNBufferIdx inputA, NNBufferIdx inputB) :
				NNOp()
			{
				input.push_back(inputA);
				input.push_back(inputB);
			};

			NNAddBiasOp(const NNAddBiasOp& other) :
				NNOp(other)
			{
			}

			const NNAddBiasOp& operator=(const NNAddBiasOp& other)
			{
				NNOp::operator=(other);

				return *this;
			}

			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList);

			virtual SizeVec GetOutputType(std::vector<NNBuffer*>& bufferList);
		};

		class NNAddOp : public NNOp
		{
		public:
			NNAddOp(NNBufferIdx inputA, NNBufferIdx inputB, int timeResult = 0) :
				NNOp(), timeResult(timeResult)
			{
				input.push_back(inputA);
				input.push_back(inputB);
			};

			NNAddOp(const NNAddOp& other) :
				NNOp(other), timeResult(other.timeResult)
			{
			}

			const NNAddOp& operator=(const NNAddOp& other)
			{
				NNOp::operator=(other);
				timeResult = other.timeResult;
				return *this;
			}

			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList);

			virtual SizeVec GetOutputType(std::vector<NNBuffer*>& bufferList);
		private:
			size_t timeResult;
		};

		class NNCopyOp : public NNOp
		{
		public:
			NNCopyOp(NNBufferIdx inputA, int timeResult = 0) :
				NNOp(), timeResult(timeResult)
			{
				input.push_back(inputA);
			};

			NNCopyOp(const NNCopyOp& other) :
				NNOp(other), timeResult(other.timeResult)
			{
			}

			const NNCopyOp& operator=(const NNCopyOp& other)
			{
				NNOp::operator=(other);
				timeResult = other.timeResult;
				return *this;
			}

			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList);

			virtual SizeVec GetOutputType(std::vector<NNBuffer*>& bufferList);
		private:
			size_t timeResult;
		};

		class NNSubConstOp : public NNOp
		{
		public:
			NNSubConstOp(NNBufferIdx inputA, const float co) :
				NNOp(), co(co)
			{
				input.push_back(inputA);
			};

			NNSubConstOp(const NNSubConstOp& other) :
				NNOp(other), co(other.co)
			{
			}

			const NNSubConstOp& operator=(const NNSubConstOp& other)
			{
				NNOp::operator=(other);
				co = other.co;
				return *this;
			}

			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList);

			virtual SizeVec GetOutputType(std::vector<NNBuffer*>& bufferList);

		private:
			float co;
		};

		class NNAddBiasConvOp : public NNOp
		{
		public:
			NNAddBiasConvOp(NNBufferIdx inputA, NNBufferIdx inputB) :
				NNOp()
			{
				input.push_back(inputA);
				input.push_back(inputB);
			};

			NNAddBiasConvOp(const NNAddBiasConvOp& other) :
				NNOp(other)
			{
			}

			const NNAddBiasConvOp& operator=(const NNAddBiasConvOp& other)
			{
				NNOp::operator=(other);

				return *this;
			}

			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList);

			virtual SizeVec GetOutputType(std::vector<NNBuffer*>& bufferList);
		};

		class NNCrossEntropyOp : public NNOp
		{
		public:
			NNCrossEntropyOp(NNBufferIdx inputA, NNBufferIdx inputB) :
				NNOp()
			{
				input.push_back(inputA);
				input.push_back(inputB);
			};

			NNCrossEntropyOp(const NNCrossEntropyOp& other) :
				NNOp(other)
			{
			}

			const NNCrossEntropyOp& operator=(const NNCrossEntropyOp& other)
			{
				NNOp::operator=(other);

				return *this;
			}

			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList);

			virtual SizeVec GetOutputType(std::vector<NNBuffer*>& bufferList);
		};


		class NNClassificationRewardOp : public NNOp
		{
		public:
			NNClassificationRewardOp(NNBufferIdx inputA, NNBufferIdx inputB) :
				NNOp()
			{
				input.push_back(inputA);
				input.push_back(inputB);
			};

			NNClassificationRewardOp(const NNClassificationRewardOp& other) :
				NNOp(other)
			{
			}

			const NNClassificationRewardOp& operator=(const NNClassificationRewardOp& other)
			{
				NNOp::operator=(other);

				return *this;
			}

			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList);

			virtual SizeVec GetOutputType(std::vector<NNBuffer*>& bufferList);
		};

		class NNSoftMaxOp : public NNOp
		{
		public:
			NNSoftMaxOp(NNBufferIdx inputA) :
				NNOp()
			{
				input.push_back(inputA);
			};

			NNSoftMaxOp(const NNSoftMaxOp& other) :
				NNOp(other)
			{
			}

			const NNSoftMaxOp& operator=(const NNSoftMaxOp& other)
			{
				NNOp::operator=(other);

				return *this;
			}

			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList);

			virtual SizeVec GetOutputType(std::vector<NNBuffer*>& bufferList);
		};

		class NNSigmoidOp : public NNOp
		{
		public:
			NNSigmoidOp(NNBufferIdx inputA) :
				NNOp()
			{
				input.push_back(inputA);
			};

			NNSigmoidOp(const NNSigmoidOp& other) : NNOp(other)
			{
			}

			const NNSigmoidOp& operator=(const NNSigmoidOp& other)
			{
				NNOp::operator=(other);
				return *this;
			}

			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList);

			virtual SizeVec GetOutputType(std::vector<NNBuffer*>& bufferList);
		};
	}
}