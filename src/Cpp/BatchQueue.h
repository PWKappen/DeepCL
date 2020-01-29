#pragma once

#include <vector>
#include <list>
#include <queue>
#include <mutex>
#include "VariadicTuple.h"
#include "NNOperations.h"

namespace DeepCL
{
	namespace DataSystem
	{
		//Object for storing the information of a batch

		template<typename... varT>
		struct Batch
		{
		public:
			//the container for the data, may contain multiple different types at the same time. It would also be possible to use pointers/arrays instead of vector objects.
			BackendSystem::Tuple<std::vector<varT>...> data;

			//size of a batch element
			size_t batchSize;

			//Size of an element in each vector of the tuple. sizeW is used to denote the sequenceSize if the data is sequential
			std::vector<NNSystem::SizeVec> sizes;

			Batch() : batchSize(0), data(), sizes() {}
			Batch(const size_t batchSize) : batchSize(batchSize), data(), sizes()
			{}

			Batch(const Batch& other) :
				data(other.data), batchSize(other.batchSize), sizes(other.sizes)
			{}

			const Batch& operator=(const Batch& other)
			{
				data = other.data;
				batchSize = other.batchSize;
				sizer = other.sizes;

				return *this;
			}
		};

		//Object for storing batches in a thread safe fashion, it contains a pool of batches
		template<typename... varT>
		class BatchQueue
		{
		public:
			BatchQueue(const size_t numBatchesToSave, const size_t batchSize);
			~BatchQueue();

			//Returns a batch object with data loaded into it
			Batch<varT...>* GetBatch(size_t& idx);

			//Retruns a batch object which needs to be filled by the calling function
			Batch<varT...>* GetEmptyBatch(size_t& idx);

			//Returns a batch to the bool after usage
			void ReturnBatch(const size_t idx);

			//Returns a filled batch to the queue
			void AddBatch(const size_t idx);

			bool EmptyCreated();
			bool EmptyToCreate();

		private:
			//Pool of fully constructed batch elements. The batch elements contained in this vector are reused all the time and pointer on them are returned to the user
			std::vector<Batch<varT...>> data;

			//list of indices of batches that can be used for training
			std::list<size_t> fullyCreated;
			
			//queue of indices of batches that are empty and need to be filled
			std::queue<size_t> toCreate;

			//number of batches in the data pool
			size_t numBatches;

			//size of each batch
			size_t batchSize;

			//mutex to prevent race conditions on the fullyCreated list. It is looked when the fullycreated vector is accessed
			std::mutex fullyCreatedMutex;

			//mute to prevent race conditions on the toCreate queue. It is looked when the toCreate list vector is accessed
			std::mutex toCreateMutex;
		};


		//Basic constructor to initalize the object with the necessary information and create the pool of batches
		template<typename... varT>
		BatchQueue<varT...>::BatchQueue(const size_t numBatchesToSave, const size_t batchSize) :
			data(), numBatches(numBatchesToSave), batchSize(batchSize), fullyCreated(), fullyCreatedMutex(), toCreate(), toCreateMutex()
		{
			toCreateMutex.lock();
			for (size_t i = 0; i < numBatches; ++i)
			{
				//Creates the batches in the memory pool and adds them to the toCreate vector
				data.push_back(Batch<varT...>(batchSize));
				toCreate.push(i);
			}
			toCreateMutex.unlock();
		}

		template<typename... varT>
		BatchQueue<varT...>::~BatchQueue()
		{
		}

		//Returns a pointer onto an batch element that was fully created. Used to retrieve batches for trainig. Idx is necessary to be able to return the batch later.
		template<typename... varT>
		Batch<varT...>* BatchQueue<varT...>::GetBatch(size_t& idx)
		{
			bool wait = true;
			Batch<varT...>* resultBatch = nullptr;
			
			//Try retrieving a batch from the fullyCreated list until an element is availabel.
			while (wait)
			{
				fullyCreatedMutex.lock();
				if (fullyCreated.size() != 0)
				{
					idx = fullyCreated.front();
					resultBatch = &data[idx];
					fullyCreated.pop_front();
					wait = false;
				}
				fullyCreatedMutex.unlock();
				
				//sleep if no element is available
				if (wait)
					std::this_thread::sleep_for(std::chrono::microseconds(1));
			}

			//A pointer onto the batch element is returned to the caller
			return resultBatch;
		}

		//Returns a batch element that must be created.
		template<typename... varT>
		Batch<varT...>* BatchQueue<varT...>::GetEmptyBatch(size_t& idx)
		{
			bool wait = true;
			Batch<varT...>* resultBatch = nullptr;

			//Try to retrieve a batch element from toCreate until one is available.
			while (wait)
			{
				toCreateMutex.lock();
				if (toCreate.size() != 0)
				{
					idx = toCreate.front();
					resultBatch = &data[idx];
					toCreate.pop();
					wait = false;
				}
				toCreateMutex.unlock();

				//Wait for a short time when no element is available
				if (wait)
					std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
			
			//return pointer onto batch which needs to be filled.
			return resultBatch;
		}

		//Return batch after usage
		template<typename... varT>
		void BatchQueue<varT...>::ReturnBatch(const size_t idx)
		{
			toCreateMutex.lock();
			toCreate.push(idx);
			toCreateMutex.unlock();
			
		}

		//Add batch after it was constructed to the fullyCreated list.
		template<typename... varT>
		void BatchQueue<varT...>::AddBatch(const size_t idx)
		{
			fullyCreatedMutex.lock();
			fullyCreated.push_back(idx);
			fullyCreatedMutex.unlock();
		}
	}
}

