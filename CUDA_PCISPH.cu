#include "common.cuh"
#include "global_variables.cuh"
#include "utilities.cuh"
#include <chrono>

int initialize();

int mainLoop();

void multiprocessor_writer();

int main(void)
{
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	std::cout << "INITIALIZING...\n";
	int init = initialize();
	//system("CLS");
	if (init != 0) {
		printf("\n\nINITIALIZATION ERROR\n\n");
		return 1;
	}

	while (simulation_time < final_time)
	{
		std::cout << "PROGRESS:" << std::endl;
		displayProgress(start);
		int main_loop = mainLoop();

		if (main_loop != 0) {
			printf("\n\nMAIN LOOP ERROR\n\n");
			return 1;
		}

		multiprocessor_writer();
	}

	//for (int i = 0; i < 11; i++) {
	//	testFunc();
	//}
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	cudaDeviceReset();

	return 0;
}