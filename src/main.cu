//// This file contains the description of the "main" function and the declaration of the functions inside the "main_functions" file.
//// In C++, the function called "main" is mandatory. The compiler starts by readind this function.
//
//#include "common.cuh"
//#include "global_variables.cuh"
//#include "utilities.cuh"
//#include <conio.h>
//#include <chrono>
//
//// Declarations of the functions of the "main_functions.cu" file. I declared them here because these functions are only used in this file.
//
//int fileReader(); // This function reads the file in the /props directoty
//
//int initialize(); // This function makes everything necessary to initialize the simulation
//
//int mainLoop(); // This is the main loop, calculates forces, positions and more.
//
//void multiprocessor_writer(); // This function writes the .vtu files using multiple CPU cores
//
//// The main function description starts here
//int main(void)
//{
//
//	float save_count = 0; // this variable is counting the time that passes on every iteration.
//	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now(); // this variable stores the timestamp as soon as the execution starts
//	std::cout << "INITIALIZING...\n";
//
//	int rdr = fileReader();
//
//	// If the fileReader() function returns anything thats not 0, its an error and the execution must stop
//	if (rdr != 0) {
//		printf("\n\nERROR READING PROPS FILES\n\n");
//		_getch();
//		return 1;
//	}
//
//	int init = initialize();
//	std::chrono::high_resolution_clock::time_point init_end = std::chrono::high_resolution_clock::now(); // This variable stores the timestamp as soon as the initialization ends
//	auto init_time = std::chrono::duration_cast<std::chrono::seconds>(init_end - start).count();
//	std::cout << "It took " << init_time << " s to initialize\n"
//	<< "----------------------------------------------------------------\n\n";
//
//	// If the initialize() function returns anything thats not 0, its an error and the execution must stop
//	if (init != 0) {
//		printf("\n\nINITIALIZATION ERROR\n\n");
//		_getch();
//		return 1;
//	} 
//
//	std::cout << "MAIN LOOP:\n" << "Progress:" << std::endl;
//	while (simulation_time < final_time)
//	{
//		
//		displayProgress(start);
//		int main_loop = mainLoop();
//		
//		// If the main_loop() function returns anything but 0, its an error and the execution must stop
//		if (main_loop != 0) {
//			printf("\n\nMAIN LOOP ERROR\n\n");
//			_getch();
//			return 1;
//		}
//		
//		save_count += delta_t;
//
//		//writes files in every save_step defined in the system.txt file inside /props folder
//		if (save_count > save_steps / 1000) {
//			multiprocessor_writer();
//			save_count = fmod(simulation_time,(save_steps / 1000));
//		}
//
//		
//	}
//
//	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now(); //stores a timestamp as soon as the execution ends
//	cudaDeviceReset(); // deleting all used memory in this execution
//
//	std::cout << "\n\nIt took " << std::chrono::duration_cast<std::chrono::minutes>(end - start).count() << " minutes to execute this simulation.\n";
//
//	_getch(); //Makes
//	return 0;
//}