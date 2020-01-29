#pragma once

#include "NNOperations.h"

namespace DeepCL
{
	namespace NNSystem
	{
		//All optimizers derive from this base class
		class NNOptimizer
		{
		public:
			NNOptimizer();
			~NNOptimizer();

			//Returns the necessary number of auxillary buffers.
			virtual size_t GetNumBuffer() const;

			//Instantiates the optimizer for the given weight buffer. This is performed for all weight buffers.
			//It behaves the same as the Instantiate functions of the normal operations with the exception that the operations are added to the update path in general.
			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList, NNBufferIdx weightBuffer) = 0;
		};

		class NNGradientDescent : public NNOptimizer
		{
		public:
			NNGradientDescent(const float alpha) : NNOptimizer(),
				alpha(alpha)
			{}

			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList, NNBufferIdx weightBuffer);
		private:
			const float alpha;
		};

		class NNAdam : public NNOptimizer
		{
		public:
			NNAdam(const float alpha, const float beta1, const float beta2, const float epsilon) : NNOptimizer(),
				alpha(alpha), beta1(beta1), beta2(beta2), epsilon(epsilon)
			{}

			virtual size_t GetNumBuffer() const;
			virtual void Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList, NNBufferIdx weightBuffer);

		private:
			const float alpha;
			const float beta1;
			const float beta2;
			const float epsilon;

			std::vector<OperationIdx> ops;
		};
	}
}