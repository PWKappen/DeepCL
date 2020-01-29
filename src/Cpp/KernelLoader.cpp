#include "KernelLoader.h"

#include<iostream>
#include<fstream>

namespace DeepCL
{
	namespace BackendSystem
	{
		std::string* KernelLoader::LoadKernel(const std::string& fileName)
		{
			std::ifstream kernelFile(fileName, std::ios::in);

			std::string* kernelSource;
			//Read the complete file into the string kernelSource.
			if (kernelFile.is_open())
			{
				kernelFile.seekg(0, std::ios::end);
				unsigned int size = static_cast<unsigned int>(kernelFile.tellg());
				kernelSource = new std::string(size, ' ');
				kernelFile.seekg(0, std::ios::beg);
				kernelFile.read(&(*kernelSource)[0], size);
			}
			else
			{
				//If the file could't be opened return an error.
				std::cout << "ERROR: File " << fileName << " could not be opened!" << std::endl;
				return kernelSource = new std::string("ERROR");
			}

			kernelFile.close();
			//Return the read kernel file content.
			return kernelSource;
		}

		//Each line of the file should contain only one kernel file name without the ".cl" at the end. It is assumed that all kernel files
		//end with ".cl" . The names should not be longer than 256 characters.
		std::vector<std::string>* KernelLoader::LoadKernelConfig(const std::string& fileName)
		{
			//This vector will contain the names of all files at the end.
			std::vector<std::string>* kernelFiles = new std::vector<std::string>();

			std::ifstream kernelConfigFile(fileName, std::ios::in);

			if (kernelConfigFile.is_open())
			{
				//Read the file line for line and store the read kernelFile name in kernelFiles.
				std::string tmp;
				char tmpChar[256];
				while (!kernelConfigFile.eof())
				{
					kernelConfigFile.getline(tmpChar, 256);
					tmp = tmpChar;
					//Add the .cl file extension to the read file name.
					tmp += ".cl";
					kernelFiles->push_back(tmp);
				}
			}
			//Uter an error if the file could not be read.
			else
				std::cout << "ERROR: File " << fileName << "could not be opened!" << std::endl;

			return kernelFiles;
		}
	}
}