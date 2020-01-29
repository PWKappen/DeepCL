#pragma once

#include"NNOperations.h"
#include"VariadicTuple.h"

namespace DeepCL
{
	namespace DataSystem
	{
		//Class from which all data transformers should derive.
		//The variadic template specifies the number of different elements are in the data that should be loaded and what type they have.
		//The variadic template will be transformed in a variadic vector of those templates since it is assumed that each type of data is essentially an array.
		//This is especially true because in most cases batches are used so that a tuple contains multiple batches.
		template<typename... varT>
		class BaseDataTransformer
		{
		public:
			BaseDataTransformer(const size_t batchSize);
			~BaseDataTransformer();

			//This function must be called to transform the input. The dataOutput is the tuple of data that can be stored in a batch object. The first vector of sizeVec stores the size for each different type of data.
			//All elements of one type in a batch must be of the same size. For example all sequences in one batch must have the same length.
			//The data tuple stores all data parts that will then be transformed or copied into the dataOutput tuple. The data arrays in this objects can all be of different size. For this reason the offset of each data element is stored in offsets and the size of each in sizes.
			virtual void Transform(BackendSystem::Tuple<std::vector<varT>...>& dataOutput, std::vector<NNSystem::SizeVec>&, BackendSystem::Tuple<std::vector<varT>...>& data, std::vector<std::vector<size_t>>& offsets, std::vector<std::vector<NNSystem::SizeVec>>& sizes) = 0;
			
			//Used to create a Copy of this object. Necessary to create copies for each thread.
			virtual BaseDataTransformer<varT...>* AllocateCopy() = 0;

		protected:
			size_t batchSize;
		};
		
		template<typename... vatT>
		BaseDataTransformer<vatT...>::BaseDataTransformer(const size_t batchSize):
			batchSize(batchSize)
		{
		}

		template<typename... vatT>
		BaseDataTransformer<vatT...>::~BaseDataTransformer()
		{
		}

		//Similar template reasoning as in the IAM class.
		class MNISTTransformer : public BaseDataTransformer<int, float>
		{
		public:
			MNISTTransformer(const size_t batchSize);
			~MNISTTransformer();

			virtual void Transform(BackendSystem::Tuple<std::vector<int>, std::vector<float>>& dataOutput, std::vector<NNSystem::SizeVec>& newSizes, BackendSystem::Tuple<std::vector<int>, std::vector<float>>& data, std::vector<std::vector<size_t>>& offsets, std::vector<std::vector<NNSystem::SizeVec>>& sizes);

			virtual BaseDataTransformer<int, float>* AllocateCopy();
		};
	}
}