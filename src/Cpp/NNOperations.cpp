#include "NNOperations.h"

namespace DeepCL
{
	namespace NNSystem
	{

		void NNOp::SetBuffer(std::vector<NNBuffer*>& bufferList, OperationIdx op)
		{
			size_t size = input.size();
			size_t i;
			
			//Set the to variable in each input buffer to point on this operation
			for (i = 0; i < size; ++i)
			{
				NNBuffer* localBuffer = bufferList[input[i]];
				localBuffer->to = op;
				bufferList[input[i]] = localBuffer;
			}

			size = output.size();

			//Set the from variable in each output buffer to point on this operation
			for (i = 0; i < size; ++i)
			{
				NNBuffer* localBuffer = bufferList[output[i]];
				localBuffer->from = op;
				bufferList[output[i]] = localBuffer;
			}
		}

		//In most cases the default time transformation does not change the sequence length. The output has the same length as the input.
		size_t NNOp::GetTimeTransform(std::vector<NNBuffer*>& bufferList)
		{
			size_t maxTime = bufferList[input[0]]->sequenceSize;
			const size_t numinputs = input.size() - timeOffset;
			for (size_t i = 1; i < numinputs; ++i)
			{
				if (bufferList[input[i]]->sequenceSize > maxTime)
					maxTime = bufferList[input[i]]->sequenceSize - timeOffset + bufferList[input[i]]->timeOffset;
			}

			timeSteps = maxTime;
			return maxTime;
		}

		void NNMatMulOp::Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList)
		{
			//The size of the work group that should be used by this function. In future a more general work group assigned should be employed.
			const int WORK_GROUP_SIZE_X = 8;
			const int WORK_GROUP_SIZE_Y = 8;

			//Work per thread in x/y direction that each thread should perform.
			const int WPTX = 1;
			const int WPTY = 1;

			//Store all input output buffers.
			NNBuffer bufferA = *bufferList[input[0]];
			NNBuffer bufferB = *bufferList[input[1]];
			NNBuffer bufferC = *bufferList[output[0]];

			NNBuffer tmpBufferOBj = *bufferList[tmpBuffer[0]];

			//Get the indices for the matrix multiplication kernel, matrix multiplication addition kernel (The result is added to the current buffer value) and the transpose kernel. 
			KernelIdx matrixKernel = backend.GetKernelIdx("MatrixMul");
			KernelIdx matrixKernelAdd = backend.GetKernelIdx("MatrixMulAdd");

			KernelIdx transpose = backend.GetKernelIdx("Transpose");

			//Create the tuple for the forward operation
			//The ForwardBuffer function returns the sub buffer indice of the current time step that needs to be initalized. The backward behaves equivalent.
			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair> tuple(bufferA.ForwardBuffer(), bufferB.ForwardBuffer(), bufferC.ForwardBuffer(),
				dataPair(sizeof(int), bufferA.size.sizeW), dataPair(sizeof(int), bufferB.size.sizeX), dataPair(sizeof(int), bufferA.size.sizeX));
	
			//Each gradient calculation requries an transposition which will be stored in the temporary buffer.
			Tuple<BufferIdx, BufferIdx, dataPair, dataPair> tupleXTranspose(bufferA.ForwardBuffer(), tmpBufferOBj.ForwardBuffer(),
				dataPair(sizeof(int), bufferA.size.sizeX), dataPair(sizeof(int), bufferA.size.sizeW));
			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair> tupleXMatMul(tmpBufferOBj.ForwardBuffer(), bufferC.BackwardBuffer(), bufferB.BackwardBuffer(),
				dataPair(sizeof(int), bufferA.size.sizeX), dataPair(sizeof(int), bufferC.size.sizeX), dataPair(sizeof(int), bufferA.size.sizeW));

			const int bX = (bufferB.size.sizeX + WPTX - 1) / WPTX;
			const int aY = (bufferA.size.sizeW + WPTY - 1) / WPTY;

			//Add operations to the graph.
			OperationIdx  matOp = backend.AddOperation<6, BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair>(matrixKernel, tuple, cl::NullRange, cl::NDRange(((bX) + (WORK_GROUP_SIZE_X - (bX) % WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X), (aY + (WORK_GROUP_SIZE_Y - (aY%WORK_GROUP_SIZE_Y)) % WORK_GROUP_SIZE_Y)), cl::NDRange(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y), BackendSystem::OpenCLBackend::OperationType::FORWARD);
			forwardOpIdx.push_back(matOp);

			const int cX = (bufferC.size.sizeX + WPTX - 1) / WPTX;
			const int aXY = (bufferA.size.sizeX + WPTY - 1) / WPTY;
			
			//The backward operations must be added in oposite order, in which they should be executed, since the vector will perform the operations starting at the last operation.
			matOp = backend.AddOperation<6, BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair>(matrixKernelAdd, tupleXMatMul, cl::NullRange, cl::NDRange(((cX)+(WORK_GROUP_SIZE_X - (cX) % WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X), (aXY + (WORK_GROUP_SIZE_Y - (aXY%WORK_GROUP_SIZE_Y)) % WORK_GROUP_SIZE_Y)), cl::NDRange(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
			matOp = backend.AddOperation<4, BufferIdx, BufferIdx, dataPair, dataPair>(transpose, tupleXTranspose, cl::NullRange, cl::NDRange(((bufferA.size.sizeX) + (WORK_GROUP_SIZE_X - (bufferA.size.sizeX) % WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X), (bufferA.size.sizeW + (WORK_GROUP_SIZE_Y - (bufferA.size.sizeW%WORK_GROUP_SIZE_Y)) % WORK_GROUP_SIZE_Y)), cl::NDRange(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);

			Tuple<BufferIdx, BufferIdx, dataPair, dataPair> tupleYTranspose(bufferB.ForwardBuffer(), tmpBufferOBj.ForwardBuffer(),
				dataPair(sizeof(int), bufferB.size.sizeX), dataPair(sizeof(int), bufferB.size.sizeY));
			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair> tupleYMatMul(bufferC.BackwardBuffer(), tmpBufferOBj.ForwardBuffer(), bufferA.BackwardBuffer(),
				dataPair(sizeof(int), bufferC.size.sizeW), dataPair(sizeof(int), bufferB.size.sizeY), dataPair(sizeof(int), bufferC.size.sizeX));

			const int bYX = (bufferB.size.sizeY + WPTX - 1) / WPTX;
			const int cY = (bufferC.size.sizeW + WPTY - 1) / WPTY;

			matOp = backend.AddOperation<6, BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair>(matrixKernelAdd, tupleYMatMul, cl::NullRange, cl::NDRange(((bYX) + (WORK_GROUP_SIZE_X - (bYX) % WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X), (cY + (WORK_GROUP_SIZE_Y - (cY%WORK_GROUP_SIZE_Y)) % WORK_GROUP_SIZE_Y)), cl::NDRange(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
			matOp = backend.AddOperation<4, BufferIdx, BufferIdx, dataPair, dataPair>(transpose, tupleYTranspose, cl::NullRange, cl::NDRange(((bufferB.size.sizeX) + (WORK_GROUP_SIZE_X - (bufferB.size.sizeX) % WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X), (bufferB.size.sizeY + (WORK_GROUP_SIZE_Y - (bufferB.size.sizeY%WORK_GROUP_SIZE_Y)) % WORK_GROUP_SIZE_Y)), cl::NDRange(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
		}

		SizeVec NNMatMulOp::GetOutputType(std::vector<NNBuffer*>& bufferList)
		{
			//Calculate the output sizes of the matrix multiplication given the input
			return SizeVec(bufferList[input[1]]->size.sizeX, bufferList[input[0]]->size.sizeW);
		}

		void NNMatMulOp::SetTmpBuffer(std::vector<NNBuffer*>& bufferList, OperationIdx op)
		{
			//The matrix multiplication needs a temporary buffer to store the results of the transposition.
			size_t input0 = bufferList[input[0]]->size.sizeX * bufferList[input[0]]->size.sizeY *bufferList[input[0]]->size.sizeZ*bufferList[input[0]]->size.sizeW;
			size_t input1 = bufferList[input[1]]->size.sizeX * bufferList[input[1]]->size.sizeY *bufferList[input[1]]->size.sizeZ*bufferList[input[1]]->size.sizeW;

			tmpSizes.push_back(SizeVec(input0 > input1 ? input0 : input1));
		}

		void NNMatMulFlatOp::Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList)
		{
			const int WORK_GROUP_SIZE_X = 8;
			const int WORK_GROUP_SIZE_Y = 8;

			const int WPTX = 1;
			const int WPTY = 1;

			NNBuffer bufferA = *bufferList[input[0]];
			NNBuffer bufferB = *bufferList[input[1]];
			NNBuffer bufferC = *bufferList[output[0]];

			NNBuffer tmpBufferOBj = *bufferList[tmpBuffer[0]];

			KernelIdx matrixKernel = backend.GetKernelIdx("MatrixMul");
			KernelIdx matrixKernelAdd = backend.GetKernelIdx("MatrixMulAdd");

			KernelIdx transpose = backend.GetKernelIdx("Transpose");

			const int flattenedSize = bufferA.size.sizeX * bufferA.size.sizeY * bufferA.size.sizeZ;

			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair> tuple(bufferA.ForwardBuffer(), bufferB.ForwardBuffer(), bufferC.ForwardBuffer(),
				dataPair(sizeof(int), bufferA.size.sizeW), dataPair(sizeof(int), bufferB.size.sizeX), dataPair(sizeof(int), flattenedSize));
		

			Tuple<BufferIdx, BufferIdx, dataPair, dataPair> tupleXTranspose(bufferA.ForwardBuffer(), tmpBufferOBj.ForwardBuffer(),
				dataPair(sizeof(int), flattenedSize), dataPair(sizeof(int), bufferA.size.sizeW));
			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair> tupleXMatMul(tmpBufferOBj.ForwardBuffer(), bufferC.BackwardBuffer(), bufferB.BackwardBuffer(),
 				dataPair(sizeof(int), flattenedSize), dataPair(sizeof(int), bufferC.size.sizeX), dataPair(sizeof(int), bufferA.size.sizeW));

			const int bX = (bufferB.size.sizeX + WPTX - 1) / WPTX;
			const int aW = (bufferA.size.sizeW + WPTY - 1) / WPTY;

			OperationIdx  matOp = backend.AddOperation<6, BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair>(matrixKernel, tuple, cl::NullRange, cl::NDRange(((bX) + (WORK_GROUP_SIZE_X - (bX) % WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X), (aW + (WORK_GROUP_SIZE_Y - (aW%WORK_GROUP_SIZE_Y)) % WORK_GROUP_SIZE_Y)), cl::NDRange(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y), BackendSystem::OpenCLBackend::OperationType::FORWARD);
			forwardOpIdx.push_back(matOp);

			const int cX = (bufferC.size.sizeW + WPTX - 1) / WPTX;
			const int fY = (flattenedSize + WPTY - 1) / WPTY;

			matOp = backend.AddOperation<6, BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair>(matrixKernelAdd, tupleXMatMul, cl::NullRange, cl::NDRange(((cX) + (WORK_GROUP_SIZE_X - (cX) % WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X), (fY + (WORK_GROUP_SIZE_Y - (fY%WORK_GROUP_SIZE_Y)) % WORK_GROUP_SIZE_Y)), cl::NDRange(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
			matOp = backend.AddOperation<4, BufferIdx, BufferIdx, dataPair, dataPair>(transpose, tupleXTranspose, cl::NullRange, cl::NDRange(((flattenedSize)+(WORK_GROUP_SIZE_X - (flattenedSize) % WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X), (bufferA.size.sizeW + (WORK_GROUP_SIZE_Y - (bufferA.size.sizeW%WORK_GROUP_SIZE_Y)) % WORK_GROUP_SIZE_Y)), cl::NDRange(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);

			const int bYX = (bufferB.size.sizeY + WPTX - 1) / WPTX;
			const int cY = (bufferC.size.sizeW + WPTY - 1) / WPTY;

			Tuple<BufferIdx, BufferIdx, dataPair, dataPair> tupleYTranspose(bufferB.ForwardBuffer(), tmpBufferOBj.ForwardBuffer(),
				dataPair(sizeof(int), bufferB.size.sizeX), dataPair(sizeof(int), bufferB.size.sizeY));
			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair> tupleYMatMul(bufferC.BackwardBuffer(), tmpBufferOBj.ForwardBuffer(), bufferA.BackwardBuffer(),
				dataPair(sizeof(int), bufferC.size.sizeW), dataPair(sizeof(int), bufferB.size.sizeY), dataPair(sizeof(int), bufferC.size.sizeX * bufferC.size.sizeY * bufferC.size.sizeZ));

			matOp = backend.AddOperation<6, BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair>(matrixKernelAdd, tupleYMatMul, cl::NullRange, cl::NDRange(((bYX)+(WORK_GROUP_SIZE_X - (bYX) % WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X), (cY + (WORK_GROUP_SIZE_Y - (cY%WORK_GROUP_SIZE_Y)) % WORK_GROUP_SIZE_Y)), cl::NDRange(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
			matOp = backend.AddOperation<4, BufferIdx, BufferIdx, dataPair, dataPair>(transpose, tupleYTranspose, cl::NullRange, cl::NDRange(((bufferB.size.sizeX) + (WORK_GROUP_SIZE_X - (bufferB.size.sizeX) % WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X), (bufferB.size.sizeY + (WORK_GROUP_SIZE_Y - (bufferB.size.sizeY%WORK_GROUP_SIZE_Y)) % WORK_GROUP_SIZE_Y)), cl::NDRange(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
		}

		SizeVec NNMatMulFlatOp::GetOutputType(std::vector<NNBuffer*>& bufferList)
		{
			return SizeVec(bufferList[input[1]]->size.sizeX, bufferList[input[0]]->size.sizeW);
		}

		void NNMatMulFlatOp::SetTmpBuffer(std::vector<NNBuffer*>& bufferList, OperationIdx op)
		{
			size_t input0 = bufferList[input[0]]->size.sizeX * bufferList[input[0]]->size.sizeY *bufferList[input[0]]->size.sizeZ*bufferList[input[0]]->size.sizeW;
			size_t input1 = bufferList[input[1]]->size.sizeX * bufferList[input[1]]->size.sizeY *bufferList[input[1]]->size.sizeZ*bufferList[input[1]]->size.sizeW;

			tmpSizes.push_back(SizeVec(input0 > input1 ? input0 : input1));
		}

		void NNConvOp::Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList)
		{
			const int WORK_GROUP_SIZE_X = 8;
			const int WORK_GROUP_SIZE_Y = 8;

			//const int WORK_GROUP_SIZE_X = 8;
			//const int WORK_GROUP_SIZE_Y = 8;
			//const int WORK_GROUP_SIZE_Z = 4;


			NNBuffer bufferA = *bufferList[input[0]];
			NNBuffer bufferB = *bufferList[input[1]];
			NNBuffer bufferC = *bufferList[output[0]];

			NNBuffer tmpBufferOBj = *bufferList[tmpBuffer[0]];

			if (pad == -1)
				pad = ConvType::VALID == convType ? 0 : (convType == ConvType::SAME ? bufferB.size.sizeX >> 1 : bufferB.size.sizeX - 1);

			KernelIdx kernel = backend.GetKernelIdx("Convolution");//, "WIDTH_KERNEL=" + std::to_string(bufferB.size.sizeX) + " HEIGHT_KERNEL=" + std::to_string(bufferB.size.sizeY) + " STRIDE_X=" + std::to_string(strideX) + " STRIDE_Y=" + std::to_string(strideY) + " TILE_WIDTH=" + std::to_string(WORK_GROUP_SIZE_X) + " TILE_HEIGHT=" + std::to_string(WORK_GROUP_SIZE_Y)); // +" TILE_DEPTH=" + std::to_string(WORK_GROUP_SIZE_Z));
			KernelIdx kernelConvAdd = backend.GetKernelIdx("ConvolutionAdd");
			KernelIdx kernelGradWgt = backend.GetKernelIdx("ConvolutionWeightGrad");
			KernelIdx kernelReorder = backend.GetKernelIdx("RotateAndReorder");


			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair> tuple(bufferA.ForwardBuffer(), bufferB.ForwardBuffer(), bufferC.ForwardBuffer(),
				dataPair(sizeof(int), bufferA.size.sizeX), dataPair(sizeof(int), bufferA.size.sizeY), dataPair(sizeof(int), bufferB.size.sizeX), dataPair(sizeof(int), bufferB.size.sizeY), dataPair(sizeof(int), bufferB.size.sizeZ), dataPair(sizeof(int), bufferB.size.sizeW), dataPair(sizeof(int), pad), dataPair(sizeof(int), bufferA.size.sizeW));

			size_t numOutputs = ((bufferC.size.sizeX + WORK_GROUP_SIZE_X - 1) / WORK_GROUP_SIZE_X)*WORK_GROUP_SIZE_X * (bufferC.size.sizeY);

			//size_t sizeX = ((bufferC.size.sizeX + WORK_GROUP_SIZE_X - 1) / WORK_GROUP_SIZE_X)*WORK_GROUP_SIZE_X;
			//size_t sizeY = ((bufferC.size.sizeY + WORK_GROUP_SIZE_Y - 1) / WORK_GROUP_SIZE_Y)*WORK_GROUP_SIZE_Y;
			//size_t sizeZ = ((bufferC.size.sizeZ + WORK_GROUP_SIZE_Z - 1) / WORK_GROUP_SIZE_Z)*WORK_GROUP_SIZE_Z*bufferC.size.sizeW;

			OperationIdx  op = backend.AddOperation<11, BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair>(kernel, tuple, cl::NullRange, cl::NDRange(numOutputs, (bufferB.size.sizeW + (WORK_GROUP_SIZE_Y - (bufferB.size.sizeW %WORK_GROUP_SIZE_Y)) % WORK_GROUP_SIZE_Y), (bufferA.size.sizeW + (2 - (bufferA.size.sizeW % 2)) % 2)), cl::NDRange(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y, 1), BackendSystem::OpenCLBackend::OperationType::FORWARD);
			//OperationIdx  op = backend.AddOperation<11, BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair>(kernel, tuple, cl::NullRange, cl::NDRange(sizeX, sizeY, sizeZ), cl::NDRange(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y, WORK_GROUP_SIZE_Z), BackendSystem::OpenCLBackend::OperationType::FORWARD);


			forwardOpIdx.push_back(op);
			//size_t tmp = ((bufferC.size.sizeX + 2 * pad + WORK_GROUP_SIZE_X - 1) / WORK_GROUP_SIZE_X * (bufferC.size.sizeY + 2 * pad) * WORK_GROUP_SIZE_X) + (WORK_GROUP_SIZE_X - ((((bufferC.size.sizeX + 2 * pad + WORK_GROUP_SIZE_X - 1) / WORK_GROUP_SIZE_X * (bufferC.size.sizeY + 2 * pad) * WORK_GROUP_SIZE_X)) % WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X);
			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair> tupleGradWgt(bufferA.ForwardBuffer(), bufferC.BackwardBuffer(), bufferB.BackwardBuffer(),
				dataPair(sizeof(int), bufferA.size.sizeX), dataPair(sizeof(int), bufferA.size.sizeY), dataPair(sizeof(int), bufferC.size.sizeX), dataPair(sizeof(int), bufferC.size.sizeY), dataPair(sizeof(int), bufferA.size.sizeZ), dataPair(sizeof(int), bufferC.size.sizeZ), dataPair(sizeof(int), pad), dataPair(sizeof(int), bufferA.size.sizeW));
			op = backend.AddOperation<11, BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair>(kernelGradWgt, tupleGradWgt, cl::NullRange, cl::NDRange(((bufferB.size.sizeX * bufferB.size.sizeY * bufferA.size.sizeZ) + (WORK_GROUP_SIZE_X - (bufferB.size.sizeX * bufferB.size.sizeY * bufferA.size.sizeZ) % WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X), ((bufferB.size.sizeW) + (WORK_GROUP_SIZE_X - (bufferB.size.sizeW) % WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X)), cl::NDRange(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(op);

			const int gradPadding = bufferB.size.sizeX - 1 - pad;
			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair> tupleGradImg(bufferC.BackwardBuffer(), tmpBufferOBj.ForwardBuffer(), bufferA.BackwardBuffer(),
				dataPair(sizeof(int), bufferC.size.sizeX), dataPair(sizeof(int), bufferC.size.sizeY), dataPair(sizeof(int), bufferB.size.sizeX), dataPair(sizeof(int), bufferB.size.sizeY), dataPair(sizeof(int), bufferB.size.sizeW), dataPair(sizeof(int), bufferB.size.sizeZ), dataPair(sizeof(int), gradPadding), dataPair(sizeof(int), bufferA.size.sizeW));
			op = backend.AddOperation<11, BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair>(kernelConvAdd, tupleGradImg, cl::NullRange, cl::NDRange(((bufferA.size.sizeX + 2 * gradPadding + WORK_GROUP_SIZE_X - 1) / WORK_GROUP_SIZE_X * (bufferA.size.sizeY + 2 * gradPadding) * WORK_GROUP_SIZE_X) + (WORK_GROUP_SIZE_X - ((((bufferA.size.sizeX + 2 * gradPadding + WORK_GROUP_SIZE_X - 1) / WORK_GROUP_SIZE_X * (bufferA.size.sizeY + 2 * gradPadding) * WORK_GROUP_SIZE_X)) % WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X), (bufferB.size.sizeZ + (WORK_GROUP_SIZE_Y - (bufferB.size.sizeZ %WORK_GROUP_SIZE_Y)) % WORK_GROUP_SIZE_Y), (bufferA.size.sizeW + (2 - (bufferA.size.sizeW % 2)) % 2)), cl::NDRange(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y, 1), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(op);

			Tuple<BufferIdx, BufferIdx, dataPair, dataPair, dataPair, dataPair> tupleYTranspose(bufferB.ForwardBuffer(), tmpBufferOBj.ForwardBuffer(),
				dataPair(sizeof(int), bufferB.size.sizeX), dataPair(sizeof(int), bufferB.size.sizeY), dataPair(sizeof(int), bufferB.size.sizeZ), dataPair(sizeof(int), bufferB.size.sizeW));
			op = backend.AddOperation<6, BufferIdx, BufferIdx, dataPair, dataPair, dataPair, dataPair>(kernelReorder, tupleYTranspose, cl::NullRange, cl::NDRange(((bufferB.size.sizeX) + (WORK_GROUP_SIZE_X - (bufferB.size.sizeX) % WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X), (bufferB.size.sizeY + (WORK_GROUP_SIZE_Y - (bufferB.size.sizeY % WORK_GROUP_SIZE_Y)) % WORK_GROUP_SIZE_Y), ((bufferB.size.sizeZ * bufferB.size.sizeW) + (2 - ((bufferB.size.sizeZ * bufferB.size.sizeW) % 2)) % 2)), cl::NDRange(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y, 1), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(op);
		}

		SizeVec NNConvOp::GetOutputType(std::vector<NNBuffer*>& bufferList)
		{
			SizeVec buf0 = bufferList[input[0]]->size;
			SizeVec buf1 = bufferList[input[1]]->size;

			if (pad == -1)
			{
				if (convType == ConvType::FULL)
					return SizeVec(buf0.sizeX + 2 * ((buf1.sizeX >> 1)), buf0.sizeY + 2 * ((buf1.sizeY >> 1)), buf1.sizeW, buf0.sizeW);
				else if (convType == ConvType::VALID)
					return SizeVec(buf0.sizeX - 2 * (buf1.sizeX >> 1), buf0.sizeY - 2 * (buf1.sizeY >> 1), buf1.sizeW, buf0.sizeW);
				else
					return SizeVec(buf0.sizeX, buf0.sizeY, buf1.sizeW, buf0.sizeW);
			}
			else
				return SizeVec((buf0.sizeX - buf1.sizeX + 2 * pad + strideX) / strideX, (buf0.sizeY - buf1.sizeY + 2 * pad + strideY) / strideY, buf1.sizeW, buf0.sizeW);

		}

		void NNConvOp::SetTmpBuffer(std::vector<NNBuffer*>& bufferList, OperationIdx op)
		{
			tmpSizes.push_back(SizeVec(bufferList[input[1]]->size.sizeX * bufferList[input[1]]->size.sizeY *bufferList[input[1]]->size.sizeZ*bufferList[input[1]]->size.sizeW));
		}

		void NNCopyInitOp::Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList)
		{
			const int WORK_GROUP_SIZE_X = 64;

			NNBuffer bufferA = *bufferList[input[0]];
			NNBuffer bufferC = *bufferList[output[0]];

			size_t totalSize = bufferA.size.sizeX * bufferA.size.sizeY * bufferA.size.sizeZ * bufferA.size.sizeW;

			if (bufferC.GetCurTimeStep() == 0)
			{
				Tuple<BufferIdx, BufferIdx, dataPair> tuple(bufferA.ForwardBuffer(), bufferC.ForwardBuffer(),
					dataPair(sizeof(int), totalSize));
				Tuple<BufferIdx, BufferIdx, dataPair> tupleGrad(bufferC.BackwardBuffer(), bufferA.BackwardBuffer(),
					dataPair(sizeof(int), totalSize));

				KernelIdx reluKernel = backend.GetKernelIdx("Copy");
				KernelIdx reluKernelGrad = backend.GetKernelIdx("CopyAdd");

				OperationIdx  matOp = backend.AddOperation<3, BufferIdx, BufferIdx, dataPair>(reluKernel, tuple, cl::NullRange, cl::NDRange((totalSize + WORK_GROUP_SIZE_X - (totalSize%WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::FORWARD);
				forwardOpIdx.push_back(matOp);
				matOp = backend.AddOperation<3, BufferIdx, BufferIdx, dataPair>(reluKernelGrad, tupleGrad, cl::NullRange, cl::NDRange((totalSize + (WORK_GROUP_SIZE_X - (totalSize%WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
				backwardOpIdx.push_back(matOp);
			}
		}

		SizeVec NNCopyInitOp::GetOutputType(std::vector<NNBuffer*>& bufferList)
		{
			return bufferList[input[0]]->size;
		}

		void NNReLUOp::Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList)
		{
			const int WORK_GROUP_SIZE_X = 64;

			NNBuffer bufferA = *bufferList[input[0]];
			NNBuffer bufferC = *bufferList[output[0]];

			size_t totalSize = bufferA.size.sizeX * bufferA.size.sizeY * bufferA.size.sizeZ * bufferA.size.sizeW;

			Tuple<BufferIdx, BufferIdx, dataPair> tuple(bufferA.ForwardBuffer(), bufferC.ForwardBuffer(),
				dataPair(sizeof(int), totalSize));
			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair> tupleGrad(bufferA.ForwardBuffer(), bufferC.BackwardBuffer(), bufferA.BackwardBuffer(),
				dataPair(sizeof(int), totalSize));

			KernelIdx reluKernel = backend.GetKernelIdx("ReLU");
			KernelIdx reluKernelGrad = backend.GetKernelIdx("ReLUGrad");

			OperationIdx  matOp = backend.AddOperation<3, BufferIdx, BufferIdx, dataPair>(reluKernel, tuple, cl::NullRange, cl::NDRange((totalSize + WORK_GROUP_SIZE_X - (totalSize%WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::FORWARD);
			forwardOpIdx.push_back(matOp);
			matOp = backend.AddOperation<4, BufferIdx, BufferIdx, BufferIdx, dataPair>(reluKernelGrad, tupleGrad, cl::NullRange, cl::NDRange((totalSize + (WORK_GROUP_SIZE_X - (totalSize%WORK_GROUP_SIZE_X)%WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
		}

		SizeVec NNReLUOp::GetOutputType(std::vector<NNBuffer*>& bufferList)
		{
			return bufferList[input[0]]->size;
		}


		void NNTanhOp::Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList)
		{
			const int WORK_GROUP_SIZE_X = 64;

			NNBuffer bufferA = *bufferList[input[0]];
			NNBuffer bufferC = *bufferList[output[0]];

			size_t totalSize = bufferA.size.sizeX * bufferA.size.sizeY * bufferA.size.sizeZ * bufferA.size.sizeW;

			Tuple<BufferIdx, BufferIdx, dataPair> tuple(bufferA.ForwardBuffer(), bufferC.ForwardBuffer(),
				dataPair(sizeof(int), totalSize));
			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair> tupleGrad(bufferA.ForwardBuffer(), bufferC.BackwardBuffer(), bufferA.BackwardBuffer(),
				dataPair(sizeof(int), totalSize));

			KernelIdx reluKernel = backend.GetKernelIdx("Tanh");
			KernelIdx reluKernelGrad = backend.GetKernelIdx("TanhGrad");

			OperationIdx  matOp = backend.AddOperation<3, BufferIdx, BufferIdx, dataPair>(reluKernel, tuple, cl::NullRange, cl::NDRange((totalSize + WORK_GROUP_SIZE_X - (totalSize%WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::FORWARD);
			forwardOpIdx.push_back(matOp);
			matOp = backend.AddOperation<4, BufferIdx, BufferIdx, BufferIdx, dataPair>(reluKernelGrad, tupleGrad, cl::NullRange, cl::NDRange((totalSize + (WORK_GROUP_SIZE_X - (totalSize%WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
		}

		SizeVec NNTanhOp::GetOutputType(std::vector<NNBuffer*>& bufferList)
		{
			return bufferList[input[0]]->size;
		}

		void NNElemWiseProductOp::Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList)
		{
			const int WORK_GROUP_SIZE_X = 64;

			NNBuffer bufferA = *bufferList[input[0]];
			NNBuffer bufferB = *bufferList[input[1]];
			NNBuffer bufferC = *bufferList[output[0]];

			size_t totalSize = bufferA.size.sizeX * bufferA.size.sizeY * bufferA.size.sizeZ * bufferA.size.sizeW;

			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair> tuple(bufferA.ForwardBuffer(), bufferB.ForwardBuffer(), bufferC.ForwardBuffer(),
				dataPair(sizeof(int), totalSize));
			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair> tupleGrad(bufferA.ForwardBuffer(), bufferC.BackwardBuffer(), bufferB.BackwardBuffer(),
				dataPair(sizeof(int), totalSize));
			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair> tupleGrad2(bufferB.ForwardBuffer(), bufferC.BackwardBuffer(), bufferA.BackwardBuffer(),
				dataPair(sizeof(int), totalSize));

			KernelIdx reluKernel = backend.GetKernelIdx("ElemWiseProduct");
			KernelIdx reluKernelGrad = backend.GetKernelIdx("ElemWiseProductAdd");

			OperationIdx  matOp = backend.AddOperation<4, BufferIdx, BufferIdx, BufferIdx, dataPair>(reluKernel, tuple, cl::NullRange, cl::NDRange((totalSize + WORK_GROUP_SIZE_X - (totalSize%WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::FORWARD);
			forwardOpIdx.push_back(matOp);
			matOp = backend.AddOperation<4, BufferIdx, BufferIdx, BufferIdx, dataPair>(reluKernelGrad, tupleGrad, cl::NullRange, cl::NDRange((totalSize + (WORK_GROUP_SIZE_X - (totalSize%WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
			matOp = backend.AddOperation<4, BufferIdx, BufferIdx, BufferIdx, dataPair>(reluKernelGrad, tupleGrad2, cl::NullRange, cl::NDRange((totalSize + (WORK_GROUP_SIZE_X - (totalSize%WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
		}

		SizeVec NNElemWiseProductOp::GetOutputType(std::vector<NNBuffer*>& bufferList)
		{
			return bufferList[input[0]]->size;
		}

		void NNSplitOp::Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList)
		{
			const int WORK_GROUP_SIZE_X = 64;

			NNBuffer bufferA = *bufferList[input[0]];
			NNBuffer bufferC = *bufferList[output[0]];

			size_t totalSize = bufferC.size.sizeX * bufferC.size.sizeY;

			Tuple<BufferIdx, BufferIdx, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair ,dataPair> tuple(bufferA.ForwardBuffer(), bufferC.ForwardBuffer(),
				dataPair(sizeof(int), static_cast<int>(bufferC.GetCurTimeStep())), dataPair(sizeof(int), bufferC.size.sizeX), dataPair(sizeof(int), bufferC.size.sizeY), dataPair(sizeof(int), bufferA.size.sizeX), dataPair(sizeof(int), bufferA.size.sizeY), dataPair(sizeof(int), bufferA.size.sizeZ), dataPair(sizeof(int), bufferA.size.sizeW));
			Tuple<BufferIdx, BufferIdx, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair> tupleGrad(bufferA.BackwardBuffer(), bufferC.BackwardBuffer(),
				dataPair(sizeof(int), static_cast<int>(bufferC.GetCurTimeStep())), dataPair(sizeof(int), bufferC.size.sizeX), dataPair(sizeof(int), bufferC.size.sizeY), dataPair(sizeof(int), bufferA.size.sizeX), dataPair(sizeof(int), bufferA.size.sizeY), dataPair(sizeof(int), bufferA.size.sizeZ), dataPair(sizeof(int), bufferA.size.sizeW));

			KernelIdx reluKernel = backend.GetKernelIdx("SplitData");
			KernelIdx reluKernelGrad = backend.GetKernelIdx("SplitDataGrad");

			OperationIdx  matOp = backend.AddOperation<9, BufferIdx, BufferIdx, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair>(reluKernel, tuple, cl::NullRange, cl::NDRange((totalSize + WORK_GROUP_SIZE_X - (totalSize%WORK_GROUP_SIZE_X)), (bufferA.size.sizeZ + 2 - (bufferA.size.sizeZ%2)), (bufferA.size.sizeW + 2 - (bufferA.size.sizeW%2))), cl::NDRange(WORK_GROUP_SIZE_X, 1, 1), BackendSystem::OpenCLBackend::OperationType::FORWARD);
			forwardOpIdx.push_back(matOp);
			matOp = backend.AddOperation<9, BufferIdx, BufferIdx, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair>(reluKernelGrad, tupleGrad, cl::NullRange, cl::NDRange((totalSize + WORK_GROUP_SIZE_X - (totalSize%WORK_GROUP_SIZE_X)), (bufferA.size.sizeZ + 2 - (bufferA.size.sizeZ%2)), (bufferA.size.sizeW + 2 - (bufferA.size.sizeW%2))), cl::NDRange(WORK_GROUP_SIZE_X, 1, 1), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
		}

		SizeVec NNSplitOp::GetOutputType(std::vector<NNBuffer*>& bufferList)
		{
			return SizeVec(w, h, bufferList[input[0]]->size.sizeZ, bufferList[input[0]]->size.sizeW);
		}

		size_t NNSplitOp::GetTimeTransform(std::vector<NNBuffer*>& bufferList)
		{
			SizeVec size = bufferList[input[0]]->size;
			size_t time = ((size.sizeX + w - 1) / w) * ((size.sizeY + h - 1)/h);
			return time;
		}

		void NNMaxPoolingOp::Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList)
		{
			const int WORK_GROUP_SIZE_X = 8;

			NNBuffer bufferA = *bufferList[input[0]];
			NNBuffer bufferC = *bufferList[output[0]];

			Tuple<BufferIdx, BufferIdx, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair> tuple(bufferA.ForwardBuffer(), bufferC.ForwardBuffer(),
				dataPair(sizeof(int), bufferA.size.sizeX), dataPair(sizeof(int), bufferA.size.sizeY), dataPair(sizeof(int), bufferA.size.sizeZ * bufferA.size.sizeW), dataPair(sizeof(int), stride), dataPair(sizeof(int), padX), dataPair(sizeof(int), padY), dataPair(sizeof(int), size));
			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair> tupleGrad(bufferA.ForwardBuffer(), bufferC.BackwardBuffer(), bufferA.BackwardBuffer(),
				dataPair(sizeof(int), bufferA.size.sizeX), dataPair(sizeof(int), bufferA.size.sizeY), dataPair(sizeof(int), bufferA.size.sizeZ * bufferA.size.sizeW), dataPair(sizeof(int), stride), dataPair(sizeof(int), padX), dataPair(sizeof(int), padY), dataPair(sizeof(int), size));

			KernelIdx maxPoolingKernel = backend.GetKernelIdx("MaxPooling");
			KernelIdx maxPoolingKernelGrad = backend.GetKernelIdx("MaxPoolingGrad");

			OperationIdx  matOp = backend.AddOperation<9, BufferIdx, BufferIdx, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair>(maxPoolingKernel, tuple, cl::NullRange, cl::NDRange((bufferC.size.sizeX + WORK_GROUP_SIZE_X - (bufferC.size.sizeX%WORK_GROUP_SIZE_X)), (bufferC.size.sizeY + WORK_GROUP_SIZE_X - (bufferC.size.sizeY%WORK_GROUP_SIZE_X)), ((bufferC.size.sizeZ*bufferC.size.sizeW) + WORK_GROUP_SIZE_X - ((bufferC.size.sizeZ*bufferC.size.sizeW )%WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_X, 1), BackendSystem::OpenCLBackend::OperationType::FORWARD);
			forwardOpIdx.push_back(matOp);
			matOp = backend.AddOperation<10, BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair, dataPair>(maxPoolingKernelGrad, tupleGrad, cl::NullRange, cl::NDRange((bufferC.size.sizeX + WORK_GROUP_SIZE_X - (bufferC.size.sizeX%WORK_GROUP_SIZE_X)), (bufferC.size.sizeY + WORK_GROUP_SIZE_X - (bufferC.size.sizeY%WORK_GROUP_SIZE_X)), ((bufferC.size.sizeZ*bufferC.size.sizeW) + WORK_GROUP_SIZE_X - ((bufferC.size.sizeZ*bufferC.size.sizeW) % WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_X, 1), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
		}

		SizeVec NNMaxPoolingOp::GetOutputType(std::vector<NNBuffer*>& bufferList)
		{
			SizeVec tmpSize = bufferList[input[0]]->size;
			return SizeVec((tmpSize.sizeX - size + padX) / stride + 1, (tmpSize.sizeY - size + padY) / stride + 1, tmpSize.sizeZ, tmpSize.sizeW);
		}

		void NNMatTransposeOp::Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList)
		{
			const int WORK_GROUP_SIZE_X = 8;
			const int WORK_GROUP_SIZE_Y = 8;

			NNBuffer bufferA = *bufferList[input[0]];
			NNBuffer bufferC = *bufferList[output[0]];

			Tuple<BufferIdx, BufferIdx, dataPair, dataPair> tuple(bufferA.ForwardBuffer(), bufferC.ForwardBuffer(),
				dataPair(sizeof(int), bufferA.size.sizeX), dataPair(sizeof(int), bufferA.size.sizeW));
			Tuple<BufferIdx, BufferIdx, dataPair, dataPair> tupleGrad(bufferC.BackwardBuffer(), bufferA.BackwardBuffer(),
				dataPair(sizeof(int), bufferA.size.sizeX), dataPair(sizeof(int), bufferA.size.sizeW));

			KernelIdx reluKernel = backend.GetKernelIdx("Transpose");

			OperationIdx  matOp = backend.AddOperation<4, BufferIdx, BufferIdx, dataPair, dataPair>(reluKernel, tuple, cl::NullRange, cl::NDRange(((bufferA.size.sizeX) + (WORK_GROUP_SIZE_X - (bufferA.size.sizeX) % WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X), (bufferA.size.sizeW + (WORK_GROUP_SIZE_Y - (bufferA.size.sizeW%WORK_GROUP_SIZE_Y)) % WORK_GROUP_SIZE_Y)), cl::NDRange(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y), BackendSystem::OpenCLBackend::OperationType::FORWARD);
			forwardOpIdx.push_back(matOp);
		}

		SizeVec NNMatTransposeOp::GetOutputType(std::vector<NNBuffer*>& bufferList)
		{
			SizeVec sizeV = bufferList[0]->size;
			return SizeVec(sizeV.sizeY, sizeV.sizeX, sizeV.sizeZ, sizeV.sizeW);
		}

		void NNLeastSquaresOp::Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList)
		{
			const int WORK_GROUP_SIZE_X = 64;

			NNBuffer bufferA = *bufferList[input[0]];
			NNBuffer bufferC = *bufferList[output[0]];
			NNBuffer labelBuffer = *bufferList[input[1]];

			size_t totalSize = bufferA.size.sizeX * bufferA.size.sizeW;

			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair> tuple(bufferA.ForwardBuffer(), labelBuffer.ForwardBuffer(), bufferC.ForwardBuffer(),
				dataPair(sizeof(int), bufferA.size.sizeX), dataPair(sizeof(int), bufferA.size.sizeW));
			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair> tupleGrad(bufferA.ForwardBuffer(), labelBuffer.ForwardBuffer(), bufferA.BackwardBuffer(),
				dataPair(sizeof(int), totalSize), dataPair(sizeof(int), bufferA.size.sizeW));

			KernelIdx kernel = backend.GetKernelIdx("MeanSquaredError");
			KernelIdx kernelGrad = backend.GetKernelIdx("MeanSquaredErrorGrad");

			OperationIdx  matOp = backend.AddOperation<5, BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair>(kernel, tuple, cl::NullRange, cl::NDRange((bufferA.size.sizeW + WORK_GROUP_SIZE_X - (bufferA.size.sizeW%WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::FORWARD);
			forwardOpIdx.push_back(matOp);
			matOp = backend.AddOperation<5, BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair>(kernelGrad, tupleGrad, cl::NullRange, cl::NDRange((totalSize + (WORK_GROUP_SIZE_X - (totalSize %WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
		}

		SizeVec NNLeastSquaresOp::GetOutputType(std::vector<NNBuffer*>& bufferList)
		{
			return SizeVec(1);
		}

		void NNAddBiasOp::Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList)
		{
			const int WORK_GROUP_SIZE_X = 64;

			NNBuffer bufferA = *bufferList[input[0]];
			NNBuffer bufferC = *bufferList[output[0]];
			NNBuffer bufferB = *bufferList[input[1]];

			size_t totalSize = bufferA.size.sizeX * bufferA.size.sizeW;

			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair> tuple(bufferA.ForwardBuffer(), bufferB.ForwardBuffer(), bufferC.ForwardBuffer(),
				dataPair(sizeof(int), bufferA.size.sizeX), dataPair(sizeof(int), bufferA.size.sizeW));
			Tuple<BufferIdx, BufferIdx, dataPair, dataPair> tupleGradB(bufferC.BackwardBuffer(), bufferB.BackwardBuffer(),
				dataPair(sizeof(int), bufferC.size.sizeX), dataPair(sizeof(int), bufferC.size.sizeW));
			Tuple<BufferIdx, BufferIdx, dataPair> tupleGradA(bufferC.BackwardBuffer(), bufferA.BackwardBuffer(), 
				dataPair(sizeof(int), totalSize));

			KernelIdx kernel = backend.GetKernelIdx("AddToMatrix");
			KernelIdx kernelGradB = backend.GetKernelIdx("AddToMatrixGrad");
			KernelIdx kernelGradA = backend.GetKernelIdx("CopyAdd");

			OperationIdx  matOp = backend.AddOperation<5, BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair>(kernel, tuple, cl::NullRange, cl::NDRange((totalSize + WORK_GROUP_SIZE_X - (totalSize%WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::FORWARD);
			forwardOpIdx.push_back(matOp);
			matOp = backend.AddOperation<3, BufferIdx, BufferIdx, dataPair>(kernelGradA, tupleGradA, cl::NullRange, cl::NDRange((totalSize + (WORK_GROUP_SIZE_X - (totalSize %WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
			matOp = backend.AddOperation<4, BufferIdx, BufferIdx, dataPair, dataPair>(kernelGradB, tupleGradB, cl::NullRange, cl::NDRange((bufferC.size.sizeX + (WORK_GROUP_SIZE_X - (bufferC.size.sizeX %WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
		}

		SizeVec NNAddBiasOp::GetOutputType(std::vector<NNBuffer*>& bufferList)
		{
			return bufferList[input[0]]->size;
		}

		void NNAddOp::Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList)
		{
			const int WORK_GROUP_SIZE_X = 64;

			NNBuffer bufferA = *bufferList[input[0]];
			NNBuffer bufferC = *bufferList[output[0]];
			NNBuffer bufferB = *bufferList[input[1]];

			size_t totalSize = bufferA.size.sizeX * bufferA.size.sizeW * bufferA.size.sizeY * bufferA.size.sizeZ;

			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair> tuple(bufferA.ForwardBuffer(), bufferB.ForwardBuffer(), bufferC.ForwardBuffer(bufferC.GetCurTimeStep() + timeResult),
				dataPair(sizeof(int), totalSize));
			Tuple<BufferIdx, BufferIdx, dataPair> tupleGradB(bufferC.BackwardBuffer(bufferC.GetCurTimeStep() + timeResult), bufferB.BackwardBuffer(),
				dataPair(sizeof(int), totalSize));
			Tuple<BufferIdx, BufferIdx, dataPair> tupleGradA(bufferC.BackwardBuffer(bufferC.GetCurTimeStep() + timeResult), bufferA.BackwardBuffer(),
				dataPair(sizeof(int), totalSize));

			KernelIdx kernel = backend.GetKernelIdx("Add");
			KernelIdx kernelGradA = backend.GetKernelIdx("CopyAdd");

			OperationIdx  matOp = backend.AddOperation<4, BufferIdx, BufferIdx, BufferIdx, dataPair>(kernel, tuple, cl::NullRange, cl::NDRange((totalSize + WORK_GROUP_SIZE_X - (totalSize%WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::FORWARD);
			forwardOpIdx.push_back(matOp);
			matOp = backend.AddOperation<3, BufferIdx, BufferIdx, dataPair>(kernelGradA, tupleGradA, cl::NullRange, cl::NDRange((totalSize + (WORK_GROUP_SIZE_X - (totalSize %WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
			matOp = backend.AddOperation<3, BufferIdx, BufferIdx, dataPair>(kernelGradA, tupleGradB, cl::NullRange, cl::NDRange((totalSize + (WORK_GROUP_SIZE_X - (totalSize %WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
		}

		SizeVec NNAddOp::GetOutputType(std::vector<NNBuffer*>& bufferList)
		{
			return bufferList[input[0]]->size;
		}

		void NNCopyOp::Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList)
		{
			const int WORK_GROUP_SIZE_X = 64;

			NNBuffer bufferA = *bufferList[input[0]];
			NNBuffer bufferC = *bufferList[output[0]];

			size_t totalSize = bufferA.size.sizeX * bufferA.size.sizeW * bufferA.size.sizeY * bufferA.size.sizeZ;

			Tuple<BufferIdx, BufferIdx, dataPair> tuple(bufferA.ForwardBuffer(), bufferC.ForwardBuffer(bufferC.GetCurTimeStep() + timeResult),
				dataPair(sizeof(int), totalSize));
			Tuple<BufferIdx, BufferIdx, dataPair> tupleGradB(bufferC.BackwardBuffer(bufferC.GetCurTimeStep() + timeResult), bufferA.BackwardBuffer(),
				dataPair(sizeof(int), totalSize));

			KernelIdx kernel = backend.GetKernelIdx("Copy");
			KernelIdx kernelGradA = backend.GetKernelIdx("CopyAdd");

			OperationIdx  matOp = backend.AddOperation<3, BufferIdx, BufferIdx, dataPair>(kernel, tuple, cl::NullRange, cl::NDRange((totalSize + WORK_GROUP_SIZE_X - (totalSize%WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::FORWARD);
			forwardOpIdx.push_back(matOp);
			matOp = backend.AddOperation<3, BufferIdx, BufferIdx, dataPair>(kernelGradA, tupleGradB, cl::NullRange, cl::NDRange((totalSize + (WORK_GROUP_SIZE_X - (totalSize %WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
		}

		SizeVec NNCopyOp::GetOutputType(std::vector<NNBuffer*>& bufferList)
		{
			return bufferList[input[0]]->size;
		}

		void NNSubConstOp::Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList)
		{
			const int WORK_GROUP_SIZE_X = 64;

			NNBuffer bufferA = *bufferList[input[0]];
			NNBuffer bufferC = *bufferList[output[0]];


			size_t totalSize = bufferA.size.sizeX * bufferA.size.sizeW * bufferA.size.sizeY * bufferA.size.sizeZ;

			Tuple<BufferIdx, BufferIdx, std::pair<size_t, float>, dataPair> tuple(bufferA.ForwardBuffer(), bufferC.ForwardBuffer(),
				std::pair<size_t, float>(sizeof(float), co), dataPair(sizeof(int), totalSize));
			Tuple<BufferIdx, BufferIdx, dataPair> tupleGradA(bufferC.BackwardBuffer(), bufferA.BackwardBuffer(),
				dataPair(sizeof(int), totalSize));

			KernelIdx kernel = backend.GetKernelIdx("SubtractFromConst");
			KernelIdx kernelGradA = backend.GetKernelIdx("SubtractFromConstGrad");

			OperationIdx  matOp = backend.AddOperation<4, BufferIdx, BufferIdx, std::pair<size_t, float>, dataPair>(kernel, tuple, cl::NullRange, cl::NDRange((totalSize + WORK_GROUP_SIZE_X - (totalSize%WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::FORWARD);
			forwardOpIdx.push_back(matOp);
			matOp = backend.AddOperation<3, BufferIdx, BufferIdx, dataPair>(kernelGradA, tupleGradA, cl::NullRange, cl::NDRange((totalSize + (WORK_GROUP_SIZE_X - (totalSize %WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
		}

		SizeVec NNSubConstOp::GetOutputType(std::vector<NNBuffer*>& bufferList)
		{
			return bufferList[input[0]]->size;
		}

		void NNAddBiasConvOp::Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList)
		{
			const int WORK_GROUP_SIZE_X = 64;

			NNBuffer bufferA = *bufferList[input[0]];
			NNBuffer bufferC = *bufferList[output[0]];
			NNBuffer bufferB = *bufferList[input[1]];

			size_t totalImageSize = bufferA.size.sizeX * bufferA.size.sizeY;
			size_t totalSize = bufferA.size.sizeX * bufferA.size.sizeY * bufferA.size.sizeZ * bufferA.size.sizeW;

			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair> tuple(bufferA.ForwardBuffer(), bufferB.ForwardBuffer(), bufferC.ForwardBuffer(),
				dataPair(sizeof(int), bufferA.size.sizeX * bufferA.size.sizeY), dataPair(sizeof(int), bufferA.size.sizeZ), dataPair(sizeof(int), bufferA.size.sizeW));
			Tuple<BufferIdx, BufferIdx, dataPair, dataPair, dataPair> tupleGradB(bufferC.BackwardBuffer(), bufferB.BackwardBuffer(),
				dataPair(sizeof(int), bufferC.size.sizeX * bufferC.size.sizeY), dataPair(sizeof(int), bufferC.size.sizeZ), dataPair(sizeof(int), bufferC.size.sizeW));
			Tuple<BufferIdx, BufferIdx, dataPair> tupleGradA(bufferC.BackwardBuffer(), bufferA.BackwardBuffer()
				, dataPair(sizeof(int), totalSize));

			KernelIdx kernel = backend.GetKernelIdx("AddToImageTensor");
			KernelIdx kernelGradB = backend.GetKernelIdx("AddToImageTensorGrad");
			KernelIdx kernelGradA = backend.GetKernelIdx("CopyAdd");

			OperationIdx  matOp = backend.AddOperation<6, BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair>(kernel, tuple, cl::NullRange, cl::NDRange((totalSize + WORK_GROUP_SIZE_X - (totalSize%WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::FORWARD);
			forwardOpIdx.push_back(matOp);
			matOp = backend.AddOperation<3, BufferIdx, BufferIdx, dataPair>(kernelGradA, tupleGradA, cl::NullRange, cl::NDRange((totalSize + (WORK_GROUP_SIZE_X - (totalSize %WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
			matOp = backend.AddOperation<5, BufferIdx, BufferIdx, dataPair, dataPair, dataPair>(kernelGradB, tupleGradB, cl::NullRange, cl::NDRange((bufferC.size.sizeZ + (WORK_GROUP_SIZE_X - (bufferC.size.sizeZ %WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
		}

		SizeVec NNAddBiasConvOp::GetOutputType(std::vector<NNBuffer*>& bufferList)
		{
			return bufferList[input[0]]->size;
		}

		void NNCrossEntropyOp::Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList)
		{
			const int WORK_GROUP_SIZE_X = 8;

			NNBuffer bufferA = *bufferList[input[0]];
			NNBuffer bufferC = *bufferList[output[0]];
			NNBuffer labelBuffer = *bufferList[input[1]];

			size_t totalSize = bufferA.size.sizeX * bufferA.size.sizeW;

			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair> tuple(bufferA.ForwardBuffer(), labelBuffer.ForwardBuffer(), bufferC.ForwardBuffer(),
				dataPair(sizeof(int), bufferA.size.sizeX), dataPair(sizeof(int), bufferA.size.sizeW));
			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair> tupleGrad(bufferA.ForwardBuffer(), labelBuffer.ForwardBuffer(), bufferA.BackwardBuffer(),
				dataPair(sizeof(int), bufferA.size.sizeX), dataPair(sizeof(int), bufferA.size.sizeW), dataPair(sizeof(int), bufferA.size.sizeW));

			KernelIdx kernel = backend.GetKernelIdx("CrossEntropy");
			KernelIdx kernelGrad = backend.GetKernelIdx("CrossEntropyGrad");

			OperationIdx  matOp = backend.AddOperation<5, BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair>(kernel, tuple, cl::NullRange, cl::NDRange((bufferA.size.sizeW + WORK_GROUP_SIZE_X - (bufferA.size.sizeW%WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::FORWARD);
			forwardOpIdx.push_back(matOp);
			matOp = backend.AddOperation<6, BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair, dataPair>(kernelGrad, tupleGrad, cl::NullRange, cl::NDRange((bufferA.size.sizeX + (WORK_GROUP_SIZE_X - (bufferA.size.sizeX %WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X)), (bufferA.size.sizeW + (WORK_GROUP_SIZE_X - (bufferA.size.sizeW %WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
		}

		SizeVec NNCrossEntropyOp::GetOutputType(std::vector<NNBuffer*>& bufferList)
		{
			return SizeVec(1);
		}

		void NNClassificationRewardOp::Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList)
		{
			const int WORK_GROUP_SIZE_X = 8;

			NNBuffer bufferA = *bufferList[input[0]];
			NNBuffer bufferC = *bufferList[output[0]];
			NNBuffer labelBuffer = *bufferList[input[1]];

			size_t totalSize = bufferA.size.sizeX * bufferA.size.sizeW;

			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair> tuple(bufferA.ForwardBuffer(), labelBuffer.ForwardBuffer(), bufferC.ForwardBuffer(),
				dataPair(sizeof(int), bufferA.size.sizeX), dataPair(sizeof(int), bufferA.size.sizeW));

			KernelIdx kernel = backend.GetKernelIdx("CalcR");

			OperationIdx  matOp = backend.AddOperation<5, BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair>(kernel, tuple, cl::NullRange, cl::NDRange((bufferA.size.sizeW + WORK_GROUP_SIZE_X - (bufferA.size.sizeW%WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::FORWARD);
			forwardOpIdx.push_back(matOp);
		}

		SizeVec NNClassificationRewardOp::GetOutputType(std::vector<NNBuffer*>& bufferList)
		{
			return SizeVec(1);
		}

		void NNSoftMaxOp::Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList)
		{
			const int WORK_GROUP_SIZE_X = 8;

			NNBuffer bufferA = *bufferList[input[0]];
			NNBuffer bufferC = *bufferList[output[0]];

			size_t totalSize = bufferA.size.sizeX * bufferA.size.sizeW;

			Tuple<BufferIdx, BufferIdx, dataPair, dataPair> tuple(bufferA.ForwardBuffer(), bufferC.ForwardBuffer(),
				dataPair(sizeof(int), bufferA.size.sizeX), dataPair(sizeof(int), bufferA.size.sizeW));
			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair> tupleGrad(bufferC.BackwardBuffer(), bufferC.ForwardBuffer(), bufferA.BackwardBuffer(),
				dataPair(sizeof(int), bufferA.size.sizeX), dataPair(sizeof(int), bufferA.size.sizeW));

			KernelIdx kernel = backend.GetKernelIdx("Softmax");
			KernelIdx kernelGrad = backend.GetKernelIdx("SoftmaxGrad");

			OperationIdx  matOp = backend.AddOperation<4, BufferIdx, BufferIdx, dataPair, dataPair>(kernel, tuple, cl::NullRange, cl::NDRange((bufferA.size.sizeW + WORK_GROUP_SIZE_X - (bufferA.size.sizeW%WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::FORWARD);
			forwardOpIdx.push_back(matOp);
			matOp = backend.AddOperation<5, BufferIdx, BufferIdx, BufferIdx, dataPair, dataPair>(kernelGrad, tupleGrad, cl::NullRange, cl::NDRange((bufferA.size.sizeX + (WORK_GROUP_SIZE_X - (bufferA.size.sizeX %WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X)), (bufferA.size.sizeW + (WORK_GROUP_SIZE_X - (bufferA.size.sizeW %WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
		}

		SizeVec NNSoftMaxOp::GetOutputType(std::vector<NNBuffer*>& bufferList)
		{
			return bufferList[input[0]]->size;
		}

		void NNSigmoidOp::Instantiate(BackendSystem::OpenCLBackend& backend, std::vector<NNBuffer*>& bufferList)
		{
			const int WORK_GROUP_SIZE_X = 64;

			NNBuffer bufferA = *bufferList[input[0]];
			NNBuffer bufferC = *bufferList[output[0]];

			size_t totalSize = bufferA.size.sizeX * bufferA.size.sizeY * bufferA.size.sizeZ * bufferA.size.sizeW;

			Tuple<BufferIdx, BufferIdx, dataPair> tuple(bufferA.ForwardBuffer(), bufferC.ForwardBuffer(),
				dataPair(sizeof(int), totalSize));
			Tuple<BufferIdx, BufferIdx, BufferIdx, dataPair> tupleGrad(bufferC.ForwardBuffer(), bufferC.BackwardBuffer(), bufferA.BackwardBuffer(),
				dataPair(sizeof(int), totalSize));

			KernelIdx reluKernel = backend.GetKernelIdx("Sigmoid");
			KernelIdx reluKernelGrad = backend.GetKernelIdx("SigmoidGrad");

			OperationIdx  matOp = backend.AddOperation<3, BufferIdx, BufferIdx, dataPair>(reluKernel, tuple, cl::NullRange, cl::NDRange((totalSize + WORK_GROUP_SIZE_X - (totalSize%WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::FORWARD);
			forwardOpIdx.push_back(matOp);
			matOp = backend.AddOperation<4, BufferIdx, BufferIdx, BufferIdx, dataPair>(reluKernelGrad, tupleGrad, cl::NullRange, cl::NDRange((totalSize + (WORK_GROUP_SIZE_X - (totalSize%WORK_GROUP_SIZE_X) % WORK_GROUP_SIZE_X))), cl::NDRange(WORK_GROUP_SIZE_X), BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			backwardOpIdx.push_back(matOp);
		}

		SizeVec NNSigmoidOp::GetOutputType(std::vector<NNBuffer*>& bufferList)
		{
			return bufferList[input[0]]->size;
		}
	}
}