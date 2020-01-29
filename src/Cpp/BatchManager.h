#pragma once

#include <atomic>
#include <thread>

#include "BatchQueue.h"
#include "DataReader.h"
#include "DataTransformer.h"

namespace DeepCL
{
	namespace DataSystem
	{
		//class for controlling the asynchronous loading of data, storing the necessary objects, etc.
		template <size_t numArgs, typename... varT>
		class BatchManager
		{
		public:
			BatchManager(const size_t batchSize, const size_t numThreads, const size_t capacity, BaseDataReader<varT...>* reader, BaseDataTransformer<varT...>* transformer) :
				batchSize(batchSize), numData(0), baseReader(reader), baseTransformer(transformer), numThreads(numThreads), queue(capacity, batchSize), finished(false), lastIdx(DeepCL::MAX_UNSIGNED_INT), threads()
			{
				//Queries the number of elements in the dataset
				numData = reader->GetNumData();

				//Creates numThreads threads which run the ThreadRun function. Each thread has a different index.
				for (size_t i = 0; i < numThreads; ++i)
					threads.push_back(new std::thread(&BatchManager::ThreadRun, this, i));
			}

			//Deletes all threads and signals to the threads that they should stop.
			virtual ~BatchManager()
			{ 
				finished = true;
				for (size_t i = 0; i < numThreads; ++i)
					delete threads[i];
			}

			//Basic getter functions
			inline size_t GetBatchSize() const{ return batchSize; }
			inline size_t GetDataSize() const { return numData; }

			Batch<varT...>* GetBatch();
			//virtual Batch<T1, T2> GetBatch();

		protected:
			size_t batchSize;

			//number of data samples in the dataset
			size_t numData;
			
			//Number of threads used for loading data
			size_t numThreads;

			//Pointer onto template of object used for reading data
			BaseDataReader<varT...>* baseReader;

			//Pointer onto template of object used for transforming data
			BaseDataTransformer<varT...>* baseTransformer;

			//Vector contains all threads used for loading data
			std::vector<std::thread*> threads;

			//informs the threads that they should stop
			std::atomic<bool> finished;

			//Object to synchronize access to the batch objects
			BatchQueue<varT...> queue;

			size_t lastIdx;

			//This function is run by each created thread and loads the data
			void ThreadRun(const size_t threadIdx);
		};

		//Function returns pointer onto a batch element that can be used for training
		template<size_t numArgs, typename... varT>
		Batch<varT...>* BatchManager<numArgs, varT...>::GetBatch()
		{
			//The index of the retrieved element is stored and the last retrieved batch is returned to the batchQueue using the last stored index.
			size_t newIdx;
			Batch<varT...>* batch = queue.GetBatch(newIdx);
			if (lastIdx != DeepCL::MAX_UNSIGNED_INT)
				queue.ReturnBatch(lastIdx);
			lastIdx = newIdx;
			return batch;
		}

		//This function contains the main loop that each thread executes to load data
		template<size_t numArgs, typename... varT>
		void BatchManager<numArgs, varT...>::ThreadRun(const size_t threadIdx)
		{
			//Tuple object which will contain the data after loading but before transforamtion
			BackendSystem::Tuple<std::vector<varT>...> resultTuple;

			//The offsets of all data points in the batch, since before transformation different batch elements may be of different size.
			std::vector<std::vector<size_t>> offsets;

			//The size of each data point before the batch is transformed.
			std::vector<std::vector<NNSystem::SizeVec>> sizes;
			for (size_t i = 0; i < numArgs; ++i)
			{
				offsets.push_back(std::vector<size_t>());
				sizes.push_back(std::vector<NNSystem::SizeVec>());
			}

			//Every thread needs its own custom copy of the reader class because the loader might use files etc. also class members are changed
			BaseDataReader<varT...>* reader = baseReader->AllocateCopy(); 
			//Every thread needs its one custom copy of the transformer because class members are changed and they should be independent of any threading.
			//All threads should start at different positions in the data
			BaseDataTransformer<varT...>* transformer = baseTransformer->AllocateCopy();reader->AddOffset(threadIdx * 1000);

			//Main loop of each thread
			size_t idx;
			while (!finished)
			{
				//retrieve an batch that must be created from the BatchQueue
				Batch<varT...>* batch = queue.GetEmptyBatch(idx);
				//Clear all elements in the batch since could have been used before.
				batch->sizes.clear();

				//The tuple elements in the temporary tuple must be cleared by looping over them
				TupleLoop<0, numArgs-1 , function, std::vector<varT>...>::apply(resultTuple);

				//The data in the batch that should be filled must be cleared (It is a tuple)
				TupleLoop<0, numArgs - 1, function, std::vector<varT>...>::apply(batch->data);

				//Query a new element from the dataset. The loaded data is stored temporary in resultTuple
				reader->GetNextData(resultTuple, offsets, sizes);
				//transform the queried data and store it in the batch object retrieved from the batch queue
				transformer->Transform(batch->data, batch->sizes, resultTuple, offsets, sizes);

				for (size_t i = 0; i < numArgs; ++i)
				{
					offsets[i].clear(); //Okay because compiler in general does not reduce the size of a vector! Standard does not mandate this behaviour!
					sizes[i].clear();
				}
				queue.AddBatch(idx); //Batch is returned to the queue to signal, that it can be used for training now
			}

			delete reader;
			delete transformer;
		}

		//Functional to perform the clear operation on the element. Is used to clear each vector in the tuple.
		class function
		{
		public:
			template<typename T>
			inline static void func(T& arg)//Only vectors must be cleared
			{}

			template<typename T>
			inline static void func(std::vector<T>& arg)
			{
				arg.clear();
			}
		};

		//Function iterates over all tuple elements with indices from from too to and executes the func class in function.
		template<size_t from, size_t to, class functionObj, class... Ts>
		struct TupleLoop
		{
		public:
			inline static void apply(BackendSystem::Tuple<Ts...>& tuple)
			{
				functionObj::func(BackendSystem::get<from>(tuple));
				TupleLoop<from + 1, to, functionObj, Ts...>::apply(tuple);
			}
		};

		// terminal case
		template<size_t from, class functionObj, class... Ts>
		struct TupleLoop<from, from, functionObj, Ts...> {
		public:
			inline static void apply(BackendSystem::Tuple<Ts...>& tuple)
			{
				functionObj::func(BackendSystem::get<from>(tuple));
			}
		};
	}
}
