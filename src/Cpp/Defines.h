#pragma once

#include <climits>

namespace DeepCL
{
	//This file contains some defines and namespaces used by the system. 
//#define PROFILING_ENABLED
	typedef int DeepCLError;

	namespace BackendSystem
	{
		//Defines for indices of operations, buffers and kernels in the backend.
		typedef unsigned int KernelIdx;
		typedef unsigned int OperationIdx;
		typedef unsigned int BufferIdx;
		
		//Flags for how the memory should be created.
		enum MEM_FLAG
		{
			READ_ONLY,
			WRITE_ONLY,
			READ_WRITE
		};
	}

	//Used to initalize indices. Index was not initalized.
	const unsigned int MAX_UNSIGNED_INT = UINT_MAX;

	namespace NNSystem
	{
		using namespace BackendSystem;
	}

	//Defines of indices of operations and buffer in the Neural Network system
	typedef unsigned int NNBufferIdx;
	typedef unsigned int NNOperationIdx;

	//Used for passing kernel arguments
	typedef std::pair<size_t, int> dataPair;

	//Error codes of the Neural Network System. Propably more are necessary.
	const DeepCLError NN_DOES_NOT_EXIST = 9999999;
	const DeepCLError NN_SYSTEM_NOT_INITIALIZED = -251;
	const DeepCLError NN_GRAPH_NOT_INITIALIZED = -252;
	const DeepCLError NN_BATCH_MANAGER_NOT_INITIALIZED = -253;

}