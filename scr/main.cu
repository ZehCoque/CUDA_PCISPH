#include "common.cuh"
#include "global_variables.cuh"
#include "utilities.cuh"
#include <conio.h>
#include <chrono>

int fileReader();

int initialize();

int mainLoop();

void multiprocessor_writer();

int main(void)
{
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	std::cout << "INITIALIZING...\n";

	int rdr = fileReader();

	int init = initialize();
	std::chrono::high_resolution_clock::time_point init_end = std::chrono::high_resolution_clock::now();
	auto init_time = std::chrono::duration_cast<std::chrono::seconds>(init_end - start).count();
	std::cout << "It took " << init_time << " s to initialize\n"
	<< "----------------------------------------------------------------\n\n";
	//system("CLS");
	if (init != 0) {
		printf("\n\nINITIALIZATION ERROR\n\n");
		return 1;
	} 

	std::cout << "MAIN LOOP:\n" << "Progress:" << std::endl;
	while (simulation_time < final_time)
	{
		
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
	_getch();
	return 0;
}