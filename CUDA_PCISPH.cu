#include "common.cuh"
#include "global_variables.cuh"

int initialize();

int mainLoop();

int main(void)
{
	initialize();

	mainLoop();

	cudaDeviceReset();

	return 0;
}