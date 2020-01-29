#include "OPManager.h"

namespace DeepCL
{
	namespace NNSystem
	{

		NeuralNetwork* OPManager::activeNN = nullptr;

		OPManager::OPManager()
		{
		}


		OPManager::~OPManager()
		{
		}

		void OPManager::SetActiveNN(NeuralNetwork* nn)
		{
			activeNN = nn;
		}

		NNBufferIdx OPManager::Multiply(const NNBufferIdx a, const NNBufferIdx b, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}

			NNMatMulOp* operation = new NNMatMulOp(a, b);

			return activeNN->AddOperation(operation, timeOffset);
		}

		NNBufferIdx OPManager::Multiply(const NNBufferIdx a, const NNBufferIdx b, const NNBufferIdx result, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNMatMulOp* operation = new NNMatMulOp(a, b);

			return activeNN->AddOperation(operation, result, timeOffset);
		}

		NNBufferIdx OPManager::MultiplyFlattened(const NNBufferIdx a, const NNBufferIdx b, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNMatMulFlatOp* operation = new NNMatMulFlatOp(a, b);

			return activeNN->AddOperation(operation, timeOffset);
		}

		NNBufferIdx OPManager::MultiplyFlattened(const NNBufferIdx a, const NNBufferIdx b, const NNBufferIdx result, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNMatMulFlatOp* operation = new NNMatMulFlatOp(a, b);

			return activeNN->AddOperation(operation, result, timeOffset);
		}

		NNBufferIdx OPManager::Conv2d(const NNBufferIdx a, const NNBufferIdx b, const NNConvOp::ConvType convType, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNConvOp* operation = new NNConvOp(a, b, convType, 1, 1);

			return activeNN->AddOperation(operation, timeOffset);
		}

		NNBufferIdx OPManager::Conv2d(const NNBufferIdx a, const NNBufferIdx b, const NNBufferIdx result, const NNConvOp::ConvType convType, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNConvOp* operation = new NNConvOp(a, b, convType, 1, 1);

			return activeNN->AddOperation(operation, result, timeOffset);
		}

		NNBufferIdx OPManager::Conv2d(const NNBufferIdx a, const NNBufferIdx b, const int pad, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNConvOp* operation = new NNConvOp(a, b, pad, 1, 1);

			return activeNN->AddOperation(operation, timeOffset);
		}

		NNBufferIdx OPManager::Conv2d(const NNBufferIdx a, const NNBufferIdx b, const NNBufferIdx result, const int pad, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNConvOp* operation = new NNConvOp(a, b, pad, 1, 1);

			return activeNN->AddOperation(operation, result, timeOffset);
		}

		NNBufferIdx OPManager::MultiplyElemWise(const NNBufferIdx a, const NNBufferIdx b, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNElemWiseProductOp* operation = new NNElemWiseProductOp(a, b);

			return activeNN->AddOperation(operation, timeOffset);
		}

		NNBufferIdx OPManager::MultiplyElemWise(const NNBufferIdx a, const NNBufferIdx b, const NNBufferIdx result, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNElemWiseProductOp* operation = new NNElemWiseProductOp(a, b);

			return activeNN->AddOperation(operation, result, timeOffset);
		}

		NNBufferIdx OPManager::CopyInit(const NNBufferIdx a, const NNBufferIdx s)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNCopyInitOp* operation = new NNCopyInitOp(a);

			return activeNN->AddOperation(operation, s, 0);
		}

		NNBufferIdx OPManager::ReLU(const NNBufferIdx a, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNReLUOp* operation = new NNReLUOp(a);

			return activeNN->AddOperation(operation, timeOffset);
		}

		NNBufferIdx OPManager::ReLU(const NNBufferIdx a, const NNBufferIdx result, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNReLUOp* operation = new NNReLUOp(a);

			return activeNN->AddOperation(operation, result, timeOffset);
		}

		NNBufferIdx OPManager::Sigmoid(const NNBufferIdx a, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNSigmoidOp* operation = new NNSigmoidOp(a);

			return activeNN->AddOperation(operation, timeOffset);
		}

		NNBufferIdx OPManager::Sigmoid(const NNBufferIdx a, const NNBufferIdx result, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNSigmoidOp* operation = new NNSigmoidOp(a);

			return activeNN->AddOperation(operation, result, timeOffset);
		}

		NNBufferIdx OPManager::Tanh(const NNBufferIdx a, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNTanhOp* operation = new NNTanhOp(a);

			return activeNN->AddOperation(operation, timeOffset);
		}

		NNBufferIdx OPManager::Tanh(const NNBufferIdx a, const NNBufferIdx result, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNTanhOp* operation = new NNTanhOp(a);

			return activeNN->AddOperation(operation, result, timeOffset);
		}

		NNBufferIdx OPManager::Softmax(const NNBufferIdx a, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNSoftMaxOp* operation = new NNSoftMaxOp(a);

			return activeNN->AddOperation(operation, timeOffset);
		}

		NNBufferIdx OPManager::Softmax(const NNBufferIdx a, const NNBufferIdx result, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNSoftMaxOp* operation = new NNSoftMaxOp(a);

			return activeNN->AddOperation(operation, result, timeOffset);
		}

		NNBufferIdx OPManager::MaxPooling(const NNBufferIdx input, const int padX, const int padY, const int stride, const int size, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNMaxPoolingOp* operation = new NNMaxPoolingOp(input, padX, padY, stride, size);

			return activeNN->AddOperation(operation, timeOffset);
		}

		NNBufferIdx OPManager::MaxPooling(const NNBufferIdx input, const NNBufferIdx result, const int padX, const int padY, const int stride, const int size, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNMaxPoolingOp* operation = new NNMaxPoolingOp(input, padX, padY, stride, size);

			return activeNN->AddOperation(operation, result, timeOffset);
		}

		NNBufferIdx OPManager::SquaredError(const NNBufferIdx a, const NNBufferIdx labelY, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNLeastSquaresOp* operation = new NNLeastSquaresOp(a, labelY);

			return activeNN->AddOperation(operation, timeOffset);
		}

		NNBufferIdx OPManager::SquaredError(const NNBufferIdx a, const NNBufferIdx labelY, const NNBufferIdx result, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNLeastSquaresOp* operation = new NNLeastSquaresOp(a, labelY);

			return activeNN->AddOperation(operation, result, timeOffset);
		}

		NNBufferIdx OPManager::CrossEntropy(const NNBufferIdx a, const NNBufferIdx labelY, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNCrossEntropyOp* operation = new NNCrossEntropyOp(a, labelY);

			return activeNN->AddOperation(operation, timeOffset);
		}

		NNBufferIdx OPManager::ClassificationReward(const NNBufferIdx a, const NNBufferIdx labelY, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNClassificationRewardOp* operation = new NNClassificationRewardOp(a, labelY);

			return activeNN->AddOperation(operation, timeOffset);
		}

		NNBufferIdx OPManager::CrossEntropy(const NNBufferIdx a, const NNBufferIdx labelY, const NNBufferIdx result, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNCrossEntropyOp* operation = new NNCrossEntropyOp(a, labelY);

			return activeNN->AddOperation(operation, result, timeOffset);
		}

		NNBufferIdx OPManager::AddBias(const NNBufferIdx input, const NNBufferIdx bias, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNAddBiasOp* operation = new NNAddBiasOp(input, bias);

			return activeNN->AddOperation(operation, timeOffset);
		}

		NNBufferIdx OPManager::AddBias(const NNBufferIdx input, const NNBufferIdx bias, const NNBufferIdx result, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNAddBiasOp* operation = new NNAddBiasOp(input, bias);

			return activeNN->AddOperation(operation, result, timeOffset);
		}

		NNBufferIdx OPManager::Add(const NNBufferIdx a, const NNBufferIdx b, const size_t timeResult, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNAddOp* operation = new NNAddOp(a, b, timeResult);

			return activeNN->AddOperation(operation, timeOffset);
		}

		NNBufferIdx OPManager::Add(const NNBufferIdx a, const NNBufferIdx b, const NNBufferIdx result, const size_t timeResult, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNAddOp* operation = new NNAddOp(a, b, timeResult);

			return activeNN->AddOperation(operation, result, timeOffset);
		}

		NNBufferIdx OPManager::SubConst(const NNBufferIdx a, const float constant, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNSubConstOp* operation = new NNSubConstOp(a, constant);

			return activeNN->AddOperation(operation, timeOffset);
		}

		NNBufferIdx OPManager::SubConst(const NNBufferIdx a, const NNBufferIdx result, const float constant, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNSubConstOp* operation = new NNSubConstOp(a, constant);

			return activeNN->AddOperation(operation, result, timeOffset);
		}

		NNBufferIdx OPManager::AddBiasConv(const NNBufferIdx input, const NNBufferIdx bias, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNAddBiasConvOp* operation = new NNAddBiasConvOp(input, bias);

			return activeNN->AddOperation(operation, timeOffset);
		}

		NNBufferIdx OPManager::AddBiasConv(const NNBufferIdx input, const NNBufferIdx bias, const NNBufferIdx result, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNAddBiasConvOp* operation = new NNAddBiasConvOp(input, bias);

			return activeNN->AddOperation(operation, result, timeOffset);
		}

		NNBufferIdx OPManager::Split(const NNBufferIdx input, const int w, const int h, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNSplitOp* operation = new NNSplitOp(input, w, h);

			return activeNN->AddOperation(operation, timeOffset);
		}

		NNBufferIdx OPManager::Copy(const NNBufferIdx a, const NNBufferIdx result, const size_t timeResult, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNCopyOp* operation = new NNCopyOp(a, timeResult);

			return activeNN->AddOperation(operation, result, timeOffset);
		}

		NNBufferIdx OPManager::Split(const NNBufferIdx input, const NNBufferIdx result, const int w, const int h, const size_t timeOffset)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNSplitOp* operation = new NNSplitOp(input, w, h);

			return activeNN->AddOperation(operation, result, timeOffset);
		}

		NNBufferIdx OPManager::GRUUnit(const NNBufferIdx a, const NNBufferIdx s, const NNBufferIdx Uz, const NNBufferIdx Ur, const NNBufferIdx Uh, const NNBufferIdx Wz, const NNBufferIdx Wr, const NNBufferIdx Wh, const NNBufferIdx Bz, const NNBufferIdx Br, const NNBufferIdx Bh)
		{
			if (activeNN == nullptr)
			{
				std::cerr << "ERROR there exists no activeNN" << std::endl;
				return NN_DOES_NOT_EXIST;
			}
			NNBufferIdx z = Sigmoid(AddBias(Add(MultiplyFlattened(a, Uz), Multiply(s, Wz)), Bz));
			NNBufferIdx r = Sigmoid(AddBias(Add(MultiplyFlattened(a, Ur), Multiply(s, Wr)), Br));
			NNBufferIdx h = Tanh(AddBias(Add(MultiplyFlattened(a, Uh), Multiply(MultiplyElemWise(s, r), Wh)), Bh));
			NNBufferIdx sNew = Add(MultiplyElemWise(SubConst(z, 1.f), h), MultiplyElemWise(z, s));
			Copy(sNew, s, 1, 0);
			return sNew;
		}

		void OPManager::InitWeightUniformRnd(const NNBufferIdx w, const float minValue, const float maxValue)
		{
			if (activeNN == nullptr)
				std::cerr << "ERROR there exists no activeNN" << std::endl;

			InitOp* op = new InitUniformRnd(w, minValue, maxValue);

			activeNN->AddWeightInitializer(op);
		}

		void OPManager::InitWeightUniform(const NNBufferIdx w, const float value)
		{
			if (activeNN == nullptr)
				std::cerr << "ERROR there exists no activeNN" << std::endl;
			
			InitOp* op = new InitUniform(w, value);

			activeNN->AddWeightInitializer(op);
		}

		void OPManager::InitWeightNormalRnd(const NNBufferIdx w, const float mean, const float stddev)
		{
			if (activeNN == nullptr)
				std::cerr << "ERROR there exists no activeNN" << std::endl;
			
			InitOp* op = new InitNormalRnd(w, mean, stddev);

			activeNN->AddWeightInitializer(op);
		}

		void OPManager::InitWeightTruncatedNormalRnd(const NNBufferIdx w, const float mean, const float stddev)
		{
			if (activeNN == nullptr)
				std::cerr << "ERROR there exists no activeNN" << std::endl;

			InitOp* op = new InitTruncatedNormalRnd(w, mean, stddev);

			activeNN->AddWeightInitializer(op);
		}

		void OPManager::InitWeightTruncatedNormalXavier(const NNBufferIdx w, const float mean)
		{
			if (activeNN == nullptr)
				std::cerr << "ERROR there exists no activeNN" << std::endl;

			InitOp* op = new InitTruncatedNormalXavier(w, mean);

			activeNN->AddWeightInitializer(op);
		}
	}
}