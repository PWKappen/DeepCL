void kernel MeanSquaredError(global read_only const float* restrict A, global read_only const float* restrict Y, global write_only float* restrict L, const int n, const int m)
{
	const int i = get_global_id(0);
		
	float sum = 0;
	
	if(i >= m)
		return;
	
	float y;
	float a;
	
	for(int j = 0; j < n; ++j)
	{
		y = Y[i * n + j];
		a = A[i * n + j];
		sum += (y-a)*(y-a);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(i < m)
		L[i] = sum;
}

void kernel MeanSquaredErrorGrad(global read_only const float* restrict A, global read_only const float* restrict Y, global float* restrict gradL, const int n, const int m)
{
	const int tx = get_local_id(0);
	
	const int i = get_global_id(0);
	
	if (i < n)
		gradL[i] += (A[i] - Y[i]) / (2 * m);
}
