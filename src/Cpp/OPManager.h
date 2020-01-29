#pragma once

#include "NNOptimizer.h"
#include "WeightInitOp.h"

namespace DeepCL
{
	namespace NNSystem
	{
		class NeuralNetwork;

		class OPManager
		{
		public:
			OPManager();
			~OPManager();

			//Allows the creation of different Functions.
			static NNBufferIdx Multiply(const NNBufferIdx a, const NNBufferIdx b, const size_t timeOffset = 0);
			static NNBufferIdx MultiplyFlattened(const NNBufferIdx a, const NNBufferIdx b, const size_t timeOffset = 0);
			static NNBufferIdx Conv2d(const NNBufferIdx a, const NNBufferIdx kernel, const NNConvOp::ConvType convType, const size_t timeOffset = 0);
			static NNBufferIdx Conv2d(const NNBufferIdx a, const NNBufferIdx kernel, const int pad, const size_t timeOffset = 0);
			static NNBufferIdx MultiplyElemWise(const NNBufferIdx a, const NNBufferIdx b, const size_t timeOffset = 0);

			static NNBufferIdx ReLU(const NNBufferIdx a, const size_t timeOffset = 0);
			static NNBufferIdx Sigmoid(const NNBufferIdx a, const size_t timeOffset = 0);
			static NNBufferIdx Tanh(const NNBufferIdx a, const size_t timeOffset = 0);
			static NNBufferIdx Softmax(const NNBufferIdx input, const size_t timeOffset = 0);
			//NNBufferIdx Transpose(const NNBufferIdx a);
			static NNBufferIdx SquaredError(const NNBufferIdx a, const NNBufferIdx b, const size_t timeOffset = 0);
			static NNBufferIdx CrossEntropy(const NNBufferIdx a, const NNBufferIdx label, const size_t timeOffset = 0);
			static NNBufferIdx ClassificationReward(const NNBufferIdx a, const NNBufferIdx y, const size_t timeOffset = 0);
			static NNBufferIdx AddBias(const NNBufferIdx input, const NNBufferIdx bias, const size_t timeOffset = 0);
			static NNBufferIdx AddBiasConv(const NNBufferIdx input, const NNBufferIdx bias, const size_t timeOffset = 0);
			static NNBufferIdx Add(const NNBufferIdx a, const NNBufferIdx b, const size_t timeResult = 0, const size_t timeOffset = 0);
			static NNBufferIdx SubConst(const NNBufferIdx a, const float constant, const size_t timeOffset = 0);
			static NNBufferIdx MaxPooling(const NNBufferIdx input, const int padX, const int padY, const int stride, const int size, const size_t timeOffset = 0);

			static NNBufferIdx Split(const NNBufferIdx a, const int w, const int h, const size_t timeOffset = 0);

			//Functions which allow the result buffer to be an already existing buffer but an other time step.
			static NNBufferIdx Multiply(const NNBufferIdx a, const NNBufferIdx b, const NNBufferIdx result, const size_t timeOffset);
			static NNBufferIdx MultiplyFlattened(const NNBufferIdx a, const NNBufferIdx b, const NNBufferIdx result, const size_t timeOffset);
			static NNBufferIdx Conv2d(const NNBufferIdx a, const NNBufferIdx kernel, const NNBufferIdx result, const NNConvOp::ConvType convType, const size_t timeOffset);
			static NNBufferIdx Conv2d(const NNBufferIdx a, const NNBufferIdx kernel, const NNBufferIdx result, const int pad, const size_t timeOffset);
			static NNBufferIdx MultiplyElemWise(const NNBufferIdx a, const NNBufferIdx b, const NNBufferIdx result, const size_t timeOffset);

			static NNBufferIdx ReLU(const NNBufferIdx a, const NNBufferIdx result, const size_t timeOffset);
			static NNBufferIdx Sigmoid(const NNBufferIdx a, const NNBufferIdx result, const size_t timeOffset);
			static NNBufferIdx Tanh(const NNBufferIdx a, const NNBufferIdx result, const size_t timeOffset);
			static NNBufferIdx Softmax(const NNBufferIdx input, const NNBufferIdx result, const size_t timeOffset);
			//NNBufferIdx Transpose(const NNBufferIdx a);
			static NNBufferIdx SquaredError(const NNBufferIdx a, const NNBufferIdx b, const NNBufferIdx result, const size_t timeOffset);
			static NNBufferIdx CrossEntropy(const NNBufferIdx a, const NNBufferIdx label, const NNBufferIdx result, const size_t timeOffset);
			static NNBufferIdx AddBias(const NNBufferIdx input, const NNBufferIdx bias, const NNBufferIdx result, const size_t timeOffset);
			static NNBufferIdx AddBiasConv(const NNBufferIdx input, const NNBufferIdx bias, const NNBufferIdx result, const size_t timeOffset);
			static NNBufferIdx Add(const NNBufferIdx a, const NNBufferIdx b, const NNBufferIdx result, const size_t timeResult, const size_t timeOffset);
			static NNBufferIdx SubConst(const NNBufferIdx a, const NNBufferIdx result, const float constant, const size_t timeOffset);
			static NNBufferIdx MaxPooling(const NNBufferIdx input, const NNBufferIdx result, const int padX, const int padY, const int stride, const int size, const size_t timeOffset);

			static NNBufferIdx Copy(const NNBufferIdx a, const NNBufferIdx result, const size_t timeResult, const size_t timeOffset);

			static NNBufferIdx Split(const NNBufferIdx a, const NNBufferIdx result, const int w, const int h, const size_t timeOffset);

			static NNBufferIdx GRUUnit(const NNBufferIdx a, const NNBufferIdx s, const NNBufferIdx Uz, const NNBufferIdx Ur, const NNBufferIdx Uh, const NNBufferIdx Wz, const NNBufferIdx Wr, const NNBufferIdx Wh, const NNBufferIdx Bz, const NNBufferIdx Br, const NNBufferIdx Bh);
			static NNBufferIdx CopyInit(const NNBufferIdx a, const NNBufferIdx s);

			//Sets the NerualNetwork object to which all operations and buffer will be added.
			static void SetActiveNN(NeuralNetwork* nn);

			//Functions for adding initlizer operations that will initalize parameter buffers.
			static void InitWeightUniformRnd(const NNBufferIdx w, const float minValue, const float maxValue);
			static void InitWeightUniform(const NNBufferIdx w, const float value);
			static void InitWeightNormalRnd(const NNBufferIdx w, const float mean = 0.0f, const float stddev = 1.0f);
			static void InitWeightTruncatedNormalRnd(const NNBufferIdx w, const float mean = 0.0f, const float stddev = 1.0f);
			static void InitWeightTruncatedNormalXavier(const NNBufferIdx w, const float mean = 0.0f);

		private:
			//The NeuralNetwork object to which the operations and buffers will be added
			static NeuralNetwork* activeNN;
		};
	}

	typedef NNSystem::OPManager OP;
}


#include "NeuralNetwork.h"
