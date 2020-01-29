void kernel GradientDecent(global read_only float* restrict A, global float* restrict L, const float alpha, const int size)
{
	const int i = get_global_id(0);
		
	if(i >= size)
		return;
		
	float a = A[i];
	float l = L[i];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	L[i] = l - alpha * a;
}