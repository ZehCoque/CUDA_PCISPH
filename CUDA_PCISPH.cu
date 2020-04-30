#include "common.cuh"
#include "global_variables.cuh"
#include "utilities.cuh"
#include "chrono"

int initialize();

int mainLoop();

void multiprocessor_writer();

int main(void)
{
	auto started = std::chrono::high_resolution_clock::now();
	std::cout << "INITIALIZING\n";
	int init = initialize();
	//system("CLS");
	if (init != 0) {
		printf("\n\nINITIALIZATION ERROR\n\n");
		return 1;
	}

	while (simulation_time < final_time)
	{
		float progress = simulation_time / final_time;
		int barWidth = 70;
		std::cout << "[";
		int pos = barWidth * progress;
		for (int i = 0; i < barWidth; ++i) {
			if (i < pos) std::cout << "=";
			else if (i == pos) std::cout << ">";
			else std::cout << " ";
		}
		std::cout << "] " << dround(progress * 100.0 , 2) << " %\r";
		std::cout.flush();

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
	auto end = std::chrono::high_resolution_clock::now();
	cudaDeviceReset();

	return 0;
}