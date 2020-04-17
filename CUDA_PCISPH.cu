#include "common.cuh"
#include "global_variables.cuh"

int initialize();

int mainLoop();

void multiprocessor_writer();

int main(void)
{
	int init = initialize();

	if (init != 0) {
		printf("\n\nINITIALIZATION ERROR\n\n");
		return 1;
	}

	while (simulation_time < final_time)
	{
		int main_loop = mainLoop();

		if (main_loop != 0) {
			printf("\n\nMAIN LOOP ERROR\n\n");
			return 1;
		}

		multiprocessor_writer();
		simulation_time = final_time;
	}
	

	cudaDeviceReset();

	return 0;
}