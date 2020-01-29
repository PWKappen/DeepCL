#include "DataTransformer.h"

namespace DeepCL
{
	namespace DataSystem
	{
		MNISTTransformer::MNISTTransformer(const size_t batchSize) : BaseDataTransformer(batchSize)
		{}

		MNISTTransformer::~MNISTTransformer()
		{}

		//This function only copies the elements of the input into the output.
		//This is necessary because the loader only loads the data into an temporary buffer.
		void MNISTTransformer::Transform(BackendSystem::Tuple<std::vector<int>, std::vector<float>>& dataOutput, std::vector<NNSystem::SizeVec>& newSizes, BackendSystem::Tuple<std::vector<int>, std::vector<float>>& data, std::vector<std::vector<size_t>>& offsets, std::vector<std::vector<NNSystem::SizeVec>>& sizes)
		{
			NNSystem::SizeVec labelSize = sizes[0][0];
			NNSystem::SizeVec dataSize = sizes[1][0];
			newSizes.push_back(labelSize);
			newSizes.push_back(dataSize);

			size_t unrolledLabelSize = labelSize.sizeX * labelSize.sizeY * labelSize.sizeZ * labelSize.sizeW;
			size_t unrolledDataSize = dataSize.sizeX * dataSize.sizeY * dataSize.sizeZ * dataSize.sizeW;

			BackendSystem::get<0>(dataOutput).resize(unrolledLabelSize * batchSize);
			BackendSystem::get<1>(dataOutput).resize(unrolledDataSize * batchSize);

			std::memcpy(BackendSystem::get<0>(dataOutput).data(), BackendSystem::get<0>(data).data(), unrolledLabelSize * batchSize * sizeof(int));
			std::memcpy(BackendSystem::get<1>(dataOutput).data(), BackendSystem::get<1>(data).data(), unrolledDataSize * batchSize * sizeof(float));
		}

		//Creates a copy for each thread using a copy constructor
		BaseDataTransformer<int, float>* MNISTTransformer::AllocateCopy()
		{
			return new MNISTTransformer(batchSize);
		}
	}
}
