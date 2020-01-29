void kernel ReLU(global read_only const float* restrict A, global write_only float* restrict B, const int sizeA)
{
	#define GS 64
	
	//Always start by quering necessary information like global index etc.
	const int i = get_global_id(0);
	const int tx = get_local_id(0);
	const int sizeX = get_local_size(0);
	const int groupId = get_group_id(0);
	
	//Check if thread is in bounds
	if(i>= sizeA)
		return;
	
	//Load data into local memory
	__local float buffer[GS];
	buffer[tx] = A[groupId *sizeX + tx];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	B[groupId * sizeX + tx] = fmax(buffer[tx],0);
}

void kernel ReLUGrad(global read_only const float* restrict A, global read_only const float* restrict B, global float* restrict derivative, const int sizeA)
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
	derivative[groupId * sizeX + offset] += (buffer[offset] > 0 ? buffer2[offset] : 0);
}