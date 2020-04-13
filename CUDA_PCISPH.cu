#include "common.cuh"
#include "global_variables.cuh"

int initialize();

int mainLoop();

int main(void)
{
	initialize();

	while (simulation_time < final_time)
	{
		mainLoop();
		simulation_time = final_time;
	}
	

	cudaDeviceReset();

	return 0;
}