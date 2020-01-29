void kernel Sigmoid(global read_only const float* restrict A, global write_only float* restrict B, const int sizeA)
{
	#define GS 64
	
	const int i = get_global_id(0);
	const int tx = get_local_id(0);
	const int sizeX = get_local_size(0);
	const int groupId = get_group_id(0);
	
	if(i>= sizeA)
		return;
	
	__local float buffer[GS];
	buffer[tx] = A[groupId *sizeX + tx];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	B[groupId * sizeX + tx] = 1.f/(1.f+exp(-buffer[tx]));
}

void kernel SigmoidGrad(global read_only const float* restrict A, global read_only const float* restrict B, global float* restrict derivative, const int sizeA)
{
	#define GS 64
	
	const int i = get_global_id(0);
	const int tx = get_local_id(0);
	const int sizeX = get_local_size(0);
	const int groupId = get_group_id(0);
	
	if(i >= sizeA)
		return;
	
	int offset;	
	__local float buffer[GS];
	__local float buffer2[GS];
	buffer[tx] = A[groupId *sizeX + tx];
	buffer2[tx] = B[groupId *sizeX + tx];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	offset = tx;
	derivative[groupId * sizeX + offset] += (buffer[offset]*(1.f-buffer[offset]) * buffer2[offset]);
}