void kernel SplitData(global read_only const float* restrict input, global write_only float* restrict output, const int timeStep, const int wOut, const int hOut, const int w, const int h, const int d, const int batchSize)
{

	const int i = get_global_id(0);
	const int j = get_global_id(1);
	const int k = get_global_id(2);

	const int numFieldsX = (w + wOut - 1) / wOut;
	const int numFieldsY = (h + hOut - 1) / hOut;

	if (i >= wOut * hOut || j >= d || k >= batchSize)
		return;
	
	const int currY = timeStep / numFieldsX;
	const int currX = timeStep - currY * numFieldsX;

	const int yOut = i / wOut;
	const int xOut = i - yOut * wOut;
	const int x = xOut + wOut * currX;
	const int y = yOut + hOut * currY;

	if (x < w && y < h)
		output[xOut + wOut * (yOut + hOut * (j + d * k))] = input[x + w * (y + h * (j + d * k))];
	else
		output[xOut + wOut * (yOut + hOut * (j + d * k))] = 0;
}

void kernel SplitDataGrad(global read_only float* restrict input, const global float* restrict grad, const int timeStep, const int wOut, const int hOut, const int w, const int h, const int d, const int batchSize)
{
	
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	const int k = get_global_id(2);

	const int numFieldsX = (w + wOut - 1) / wOut;
	const int numFieldsY = (h + hOut - 1) / hOut;

	if (i >= wOut * hOut || j >= d || k >= batchSize)
		return;
	
	const int currY = timeStep / numFieldsX;
	const int currX = timeStep - currY * numFieldsX;

	const int yOut = i / wOut;
	const int xOut = i - yOut * wOut;
	const int x = xOut + wOut * currX;
	const int y = yOut + hOut * currY;

	if (x < w && y < h)
		input[x + w * (y + h * (j + d * k))] += grad[xOut + wOut * (yOut + hOut * (j + d * k))];
}