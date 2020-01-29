void kernel Softmax(global read_only const float* restrict A, global write_only float* restrict B, const int m, const int n)
{
	const int i = get_global_id(0);
	
	if(i >= n)
		return;
	
	float sum = 0;
	float result = 0;
	
	//Calculate the normalization factor.
	//At the moment each thread has to do this.
	for(int j = 0; j < m; ++j)
	{
		sum += exp(A[j + i*m]);
	}
	
	//Calculate the result of this operation.
	//Use the calculated normalization constant multiple times.
	for(int j = 0; j < m; ++j)
	{
		result = exp(A[j + i * m]) / (sum > 0 ? sum : 1);
		B[j + i * m] = result;
	}
}

void kernel SoftmaxGrad(global read_only const float* restrict A, global read_only const float* restrict B, global float* restrict derivative, const int m, const int n)
{
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int lSizeX = get_local_size(0);
	const int lSizeY = get_local_size(1);
	
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	
	float sum = 0.0f;
	
	if (i >= m || j >= n)
		return;
	
	for(int k = 0; k < m; ++k)
	{
		sum += A[j * m + k] * B[j * m + k] * ((i == k ? 1 : 0) - B[j * m + i]);
	}
	
	derivative[j * m + i] += sum;
}