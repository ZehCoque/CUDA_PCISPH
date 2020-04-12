
#include "common.cuh"
#include "initialize.cuh"

int main(void)
{
	initialize();

	cudaDeviceReset();

	return 0;
}