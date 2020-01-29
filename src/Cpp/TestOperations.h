//Contains functions for testing the implementations of different operations.
//Operation for testing the convolution operation
void Convolution(const float* data, const float* kernel, float* result, const size_t wA, const size_t hA, const size_t wK, const size_t hK, const size_t dK, const size_t numK, const int pad, const size_t batchSize)
{
	const size_t outputSizeX = (wA - wK + 2 * pad) + 1;
	const size_t outputSizeY = (hA - hK + 2 * pad) + 1;

	float sum;

	for (size_t b = 0; b < batchSize; ++b)
	{
		for (size_t i = 0; i < outputSizeY; ++i)
		{
			for (size_t j = 0; j < outputSizeX; ++j)
			{
				for (size_t k = 0; k < numK; ++k)
				{
					//Perform the convolution for a specific position in the volume of a batch element.
					//It essentially computes an weighted average.
					sum = 0;
					for (size_t z = 0; z < dK; ++z)
					{
						for (int y = i - pad; y < (static_cast<int>(i+hK) - pad); ++y)
						{
							for (int x = j - pad; x < (static_cast<int>(j+wK) - pad); ++x)
							{
								if (!(x < 0 || x >= static_cast<int>(wA) || y < 0 || y >= static_cast<int>(hA)))
								{
									sum += data[x + y * wA + z * wA*hA + b * wA*hA*dK] * kernel[x - static_cast<int>(j) + pad + (y - static_cast<int>(i) + pad)*wK + wK * hK * z + k * wK*hK*dK];
								}
							}
						}
					}
					result[j + i * outputSizeX + k * outputSizeX * outputSizeY + b * outputSizeX * outputSizeY * numK] = sum;
				}
			}
		}
	}
}

//Convolution that adds all batches. Used for checking the backward pass.
void ConvolutionAddBatch(const float* data, const float* kernel, float* result, const size_t wA, const size_t hA, const size_t wK, const size_t hK, const size_t dK, const size_t numK, const int pad, const size_t batchSize)
{
	const size_t outputSizeX = (wA - wK + 2 * pad) + 1;
	const size_t outputSizeY = (hA - hK + 2 * pad) + 1;

	float sum;

	for (size_t i = 0; i < outputSizeY; ++i)
	{
		for (size_t j = 0; j < outputSizeX; ++j)
		{
			for (size_t k = 0; k < numK; ++k)
			{
				for (size_t z = 0; z < dK; ++z)
				{
					//Perform the convolution for a specific, position in the volume.
					sum = 0;
					for (size_t b = 0; b < batchSize; ++b)
					{
						for (int y = i - pad; y < (static_cast<int>(i + hK) - pad); ++y)
						{
							for (int x = j - pad; x < (static_cast<int>(j + wK) - pad); ++x)
							{
								if (!(x < 0 || x >= static_cast<int>(wA) || y < 0 || y >= static_cast<int>(hA)))
								{
									sum += data[x + y * wA + z * wA*hA + b * wA*hA*dK] * kernel[x - static_cast<int>(j)+pad + (y - static_cast<int>(i)+pad)*wK + k * wK*hK + b*wK*hK*numK];
								}
							}
						}
					}
					result[j + i * outputSizeX + k * outputSizeX * outputSizeY * dK + z * outputSizeX * outputSizeY] = sum;
				}
			}
		}
	}
		
}

//Function for testing the matrix multiplication.
void MatrixMulitplication(const float* A, const float* B, float* C, const size_t hA, const size_t wB, const size_t wA)
{
	float sum = 0;
	for (size_t i = 0; i < hA; ++i)
	{
		for (size_t j = 0; j < wB; ++j)
		{
			sum = 0;
			for (size_t k = 0; k < wA; ++k)
			{
				sum += A[k + wA * i] * B[k * wB + j];
			}
			C[i * wB + j] = sum;
		}
	}
}

//Function for testing the transpose operation
void Transpose(const float* A, float* C, const size_t hA, const size_t wA)
{
	if (A == C)
	{
		std::cout << "ERROR: A can't be the same object as C" << std::endl;
		return;
	}

	for (size_t i = 0; i < hA; ++i)
	{
		for (size_t j = 0; j < wA; ++j)
		{
			C[i + j * hA] = A[i * wA + j];
		}
	}
}

//Function to set all values in A to zero.
void SetZero(float* A, size_t size)
{
	for (size_t i = 0; i < size; ++i)
		A[i] = 0;
}

//Function to perform the element wise product
void ElemWiseProduct(const float* A, const float* B, float* C, const size_t size)
{
	for (size_t i = 0; i < size; ++i)
		C[i] = A[i] * B[i];
}


//Function to perform the element wise addition
void ElemWiseAdd(const float* A, const float* B, float* C, const size_t size)
{
	for (size_t i = 0; i < size; ++i)
		C[i] = A[i] + B[i];
}

//Function to substract a constant from the buffer.
void SubtractFromConst(const float* A, const float co, float* C, const size_t size)
{
	for (size_t i = 0; i < size; ++i)
		C[i] = co - A[i];
}

//Function to add a constant to the buffer.
void AddToConst(const float* A, const float co, float* C, const size_t size)
{
	for (size_t i = 0; i < size; ++i)
		C[i] = co + A[i];
}

//Function to multiply a constant with the buffer A element wise.
void MultiplyConst(const float* A, const float co, float* C, const size_t size)
{
	for (size_t i = 0; i < size; ++i)
		C[i] = co * A[i];
}

//Perform a flip of the operation. Necessary for a convolution in the backward pass.
void Flip(const float* data, float* result, const size_t wK, const size_t hK, const size_t dK, const size_t numK)
{
	for (size_t i = 0; i < numK; ++i)
	{
		for (size_t j = 0; j < dK; ++j)
		{
			for (size_t k = 0; k < hK; ++k)
			{
				for (size_t l = 0; l < wK; ++l)
				{
					result[wK * hK * i + wK * hK * numK * j + wK - l  -1+ (hK - k - 1)*wK] = data[wK*hK*dK*i + wK*hK*j + l + k * wK];
				}
			}
		}
	}
}

//Calculates the absolute difference between the elements of two buffers.
void CalculateError(const float* data, const float* data2, float* error, const size_t size)
{
	for (size_t i = 0; i < size; ++i)
		error[i] = abs(data[i] - data2[i]);
}

//Calculates the accumulated absolut difference between two vectors
float CalculateErrorAccumulated(const float* data, const float* data2, const size_t size)
{
	float result = 0;
	for (size_t i = 0; i < size; ++i)
		result += abs(data[i] - data2[i]);
	return result;
}