#pragma once

#include<string>
#include<vector>

namespace DeepCL
{
	namespace BackendSystem
	{
		class KernelLoader
		{
		public:
			//Function to read one kernel file(The kernels are extracted in the backend).
			static std::string* LoadKernel(const std::string& fileName);
			//Read all kernel file names contained in the config file.
			static std::vector<std::string>* LoadKernelConfig(const std::string& fileName);
		};
	}
}

