#pragma once

#include <CL\cl.hpp>
#include <iostream>

#include "VariadicTuple.h"
#include "Defines.h"

namespace DeepCL
{
	namespace BackendSystem
	{
		//Class for stroring opencl kernels and all necessary information to perform it.
		class BaseOperation
		{
		public:
			//Function to enque(execute) the kernel of the operation
			virtual void Run(cl::CommandQueue& queue, cl::Event* event, const std::vector<cl::Buffer>& bufferList) = 0;
		};


		//derived class which contains all necessary operations to execute the kernel.
		template<size_t Tsize, class... Ts>
		class Operation : public BaseOperation
		{
		public:
			Operation(cl::Kernel* kernel, const Tuple<Ts...> parameter, const cl::NDRange offset,
				const cl::NDRange globalSize,
				const cl::NDRange localSize);
			~Operation();

			virtual void Run(cl::CommandQueue& queue, cl::Event* event, const std::vector<cl::Buffer>& bufferList);

		protected:
			Operation();
			//Kernel to be executed
			cl::Kernel* kernel;
			
			//Parameters, which should be used to execute the kernel
			Tuple<Ts...> parameter;

			//Work sizes parameters of the kernel
			cl::NDRange offset;
			cl::NDRange globalSize;
			cl::NDRange localSize;

		};

		//Operation to increment an element of the tuple. The index of the element is specified by the template argument called tuple.
		//The element that should be incremented is assumed to be a pair where the second element is incremented.
		template<size_t idx, size_t Tsize, class... Ts>
		class IncrementOperation : public BaseOperation
		{
		public:
			IncrementOperation(cl::Kernel* kernel, const Tuple<Ts...> parameter, const cl::NDRange offset,
				const cl::NDRange globalSize,
				const cl::NDRange localSize);
			~IncrementOperation();

			virtual void Run(cl::CommandQueue& queue, cl::Event* event, const std::vector<cl::Buffer>& bufferList);

		protected:
			IncrementOperation();
			cl::Kernel* kernel;
			Tuple<Ts...> parameter;
			cl::NDRange offset;
			cl::NDRange globalSize;
			cl::NDRange localSize;

		};

		
		//Function to set an aribtrary object to be used in the kernel. The func function is called in a tuple loop.
		template<class T> class SetArgument
		{
		public:
			inline static void func(T arg, const size_t i, cl::Kernel* kernel, const std::vector<cl::Buffer>& bufferList)
			{
#ifdef _DEBUG
				cl_int err = kernel->setArg(i, arg);
				if (err != CL_SUCCESS)
					std::cout << "Error setArg: " << err << std::endl;
#else
				kernel->setArg(i, arg);
#endif // DEBUG	
			}
		};

		//Specialization to set an buffer object to be used in the kernel.
		template<> class SetArgument<BufferIdx>
		{
		public:
			inline static void func(BufferIdx arg, const size_t i, cl::Kernel* kernel, const std::vector<cl::Buffer>& bufferList)
			{
#ifdef _DEBUG
				cl_int err = kernel->setArg(i, bufferList[arg]);
				if (err != CL_SUCCESS)
					std::cout << "Error setArg: " << err << std::endl;
#else
				kernel->setArg(i, bufferList[arg]);
#endif // DEBUG	
			}
		};

		//Specialization to set an simple data type to the kernel. The first part of the pair contains the size of the second element. The second element is set as parameter to the kernel.
		template <class T> class SetArgument<std::pair<size_t, T>>
		{
		public:
			inline static void func(std::pair<size_t, T> arg, const size_t i, cl::Kernel* kernel, const std::vector<cl::Buffer>& bufferList)
			{

#ifdef _DEBUG
				cl_int err = kernel->setArg(i, arg.first, &arg.second);
				if (err != CL_SUCCESS)
					std::cout << "Error setArg: " << err << std::endl;
#else
				kernel->setArg(i, arg.first, &arg.second);
#endif // DEBUG	
			}
		};

		//Specialization to set an simple data type to the kernel. The first part of the pair contains the size of the data object the second element points onto. The data element the second element points onto is set as parameter to the kernel.
		template <class T> class SetArgument < std::pair<size_t, T*> >
		{
		public:
			inline static void func(std::pair<size_t, T*> arg, const size_t i, cl::Kernel* kernel, const std::vector<cl::Buffer>& bufferList)
			{
#ifdef _DEBUG
				cl_int err = kernel->setArg(i, arg.first, arg.second);
				if (err != CL_SUCCESS)
					std::cout << "Error setArg: " << err << std::endl;
#else
				kernel->setArg(i, arg.first, &arg.second);
#endif // DEBUG	
			}
		};

		//Constructors and descructors for the different operations. All of them set the member objects to the corresponding function parameters.
		template<size_t Tsize, class... Ts>
		Operation<Tsize, Ts...>::Operation(cl::Kernel* kernel, const Tuple<Ts...> tuple, const cl::NDRange offset,
			const cl::NDRange globalSize, const cl::NDRange localSize) :
			kernel(kernel), parameter(tuple), offset(offset), globalSize(globalSize), localSize(localSize)
		{

		}

		template<size_t Tsize, class... Ts>
		Operation<Tsize, Ts...>::~Operation()
		{

		}

		template<size_t Tsize, class... Ts>
		Operation<Tsize, Ts...>::Operation()
		{

		}

		template<size_t idx, size_t Tsize, class... Ts>
		IncrementOperation<idx, Tsize, Ts...>::IncrementOperation(cl::Kernel* kernel, const Tuple<Ts...> tuple, const cl::NDRange offset,
			const cl::NDRange globalSize, const cl::NDRange localSize) :
			kernel(kernel), parameter(tuple), offset(offset), globalSize(globalSize), localSize(localSize)
		{

		}


		template<size_t idx, size_t Tsize, class... Ts>
		inline void IncrementOperation<idx, Tsize, Ts...>::Run(cl::CommandQueue& queue, cl::Event* clEvent, const std::vector<cl::Buffer>& bufferList)
		{
			//Apply the SetArgument func functions recursively on the parameter
			SetArgumentLoop<0, Tsize - 1, Ts...>::apply(parameter, kernel, bufferList);

			//Enqueue the kernel to the OpenCL queue
#ifdef _DEBUG
			cl_int err = queue.enqueueNDRangeKernel(*kernel, offset, globalSize, localSize, nullptr, clEvent);
			if (err != CL_SUCCESS)
				std::cout << "Error enqueueNDRangeKernel: " << err << std::endl;

#else
			queue.enqueueNDRangeKernel(*kernel, offset, globalSize, localSize, nullptr, clEvent);
#endif // DEBUG

			//increment the specific parameter after each execution (used for the adam optimizer)
			++get<1>(get<idx>(parameter));
		}

		template<size_t idx, size_t Tsize, class... Ts>
		IncrementOperation<idx, Tsize, Ts...>::~IncrementOperation()
		{

		}

		template<size_t idx, size_t Tsize, class... Ts>
		IncrementOperation<idx, Tsize, Ts...>::IncrementOperation()
		{

		}
		
		//Call the func function on each tuple element recursivly.
		//From is the first element of the tuple that is set and to is the last element that is set (for(from<=to))
		template<size_t from, size_t to, class... Ts>
		struct SetArgumentLoop
		{
		public:
			inline static void apply(Tuple<Ts...> tuple, cl::Kernel* kernel, const std::vector<cl::Buffer>& bufferList)
			{
				//Call the setargument function using get and elemholder to specify the template argument
				SetArgument<ElemHolder<from, Tuple<Ts...>>::type>::func(get<from>(tuple), from, kernel, bufferList);
				//Call the SetARgumentLoop apply function with from increased by one.
				SetArgumentLoop<from + 1, to, Ts...>::apply(tuple, kernel, bufferList);
			}
		};

		// Terminal case when from equals to. No more recursion takes place and the last call to SetArgument is performed.
		template<size_t from, class... Ts>
		struct SetArgumentLoop<from, from, Ts...> {
		public:
			inline static void apply(Tuple<Ts...> tuple, cl::Kernel* kernel, const std::vector<cl::Buffer>& bufferList)
			{
				SetArgument<ElemHolder<from, Tuple<Ts...>>::type>::func(get<from>(tuple), from, kernel, bufferList);
			}
		};
		
		//The same as the increment operation run fucntion but without incrementing anything.
		template<size_t Tsize, class... Ts>
		inline void Operation<Tsize, Ts...>::Run(cl::CommandQueue& queue, cl::Event* clEvent, const std::vector<cl::Buffer>& bufferList)
		{
			SetArgumentLoop<0, Tsize - 1, Ts...>::apply(parameter, kernel, bufferList);

#ifdef _DEBUG
			cl_int err = queue.enqueueNDRangeKernel(*kernel, offset, globalSize, localSize, nullptr, clEvent);
			if (err != CL_SUCCESS)
				std::cout << "Error enqueueNDRangeKernel: " << err << std::endl;
			
#else
			queue.enqueueNDRangeKernel(*kernel, offset, globalSize, localSize, nullptr, clEvent);
#endif // DEBUG

		}
	}
}