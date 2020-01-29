void kernel AddToMatrix(global read_only const float* restrict A, global read_only const float* restrict Y, global write_only float* restrict L, const int n, const int m)
{
	const int i = get_global_id(0);
		
	if(i >= n * m)
		return;
		
	float y = Y[i%n];
	float a = A[i];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	L[i] = a + y;
}


void kernel AddToMatrixGrad(global read_only const float* restrict A, global float* restrict gradL, const int n, const int m)
{
	const int i = get_global_id(0);
	
	float sum = 0;
	
	if(i >= n)
		return;
	
	float a;
	
	for(int j = 0; j < m; ++j)
	{
		a = A[i + j * n];
		
		barrier(CLK_LOCAL_MEM_FENCE);
		sum += a;
	}
	
	gradL[i] += sum;
}

void kernel AddToImageTensor(global read_only const float* restrict A, global read_only const float* restrict Y, global write_only float* restrict L, const int n, const int m, const int batchSize)
{
	const int i = get_global_id(0);

	if (i >= n * m * batchSize)
		return;
	int featureMap = (int)(i / n) % m;
	float y = Y[featureMap];
	float a = A[i];

	barrier(CLK_LOCAL_MEM_FENCE);

	L[i] = a + y;
}

void kernel AddToImageTensorGrad(global read_only const float* restrict A, global float* restrict gradL, const int n, const int m, const int batchSize)
{
	const int i = get_global_id(0);

	float sum = 0;

	if (i >= m)
		return;

	float a;

	for (int k = 0; k < batchSize; ++k)
	{
		for (int j = 0; j < n; ++j)
		{
			a = A[(k*m+i) * n + j];

			sum += a;
		}
	}
	gradL[i] += sum;
}

void kernel CopyAdd(global read_only const float* restrict A, global float* restrict Y, const int n)
{
	const int i = get_global_id(0);
	
	if(i >= n)
		return;
	
	Y[i] += A[i];
}

void kernel Copy(global read_only const float* restrict A, global write_only float* restrict Y, const int n)
{
	const int i = get_global_id(0);

	if (i >= n)
		return;

	Y[i] = A[i];
}