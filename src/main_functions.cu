//This file defines the main functions of the simulation. These functions are called in the "main" function inside the "main.cu" file.

#define _USE_MATH_DEFINES

#include "particle_positions.cuh"
#include "utilities.cuh"
#include "VTK.cuh"
#include "hashing.cuh"
#include "particle_parameters.cuh"
#include <math.h>
#include <future> 
#include <chrono>
#include <math.h>
#include "vector_types.h"

//declaration of all global variables that are going to be used in this file by all functions

char main_path[1024]; //stores the main path of the result instance
char vtk_group_path[1024]; //stores the path of the vtk group file
char vtu_fullpath[1024]; //stores the path of the current iteration file
char vtu_path[1024]; //stores the path of the vtu directory (where the vtu files are saved)

std::string pointDataNames[] = { "density", "pressure" }; //stores the names of the point data to display in Paraview
std::string vectorDataNames[] = {"velocity", "pressure force", "viscosity force", "st force" }; //stores the names of the vector data to display in Paraview

//initial conditions
float PARTICLE_RADIUS; //stores the particle radius value
float MASS_calc; //stores the calculated mass value
float USER_MASS; //stores the mass defined by the user in
float PARTICLE_DIAMETER; //stores the particle diameter value
float F_INITIAL_POSITION[3]; //fluid particles initial position
float F_FINAL_POSITION[3]; //fluid particles final position
float B_INITIAL_POSITION[3]; //boundary particles final position
float B_FINAL_POSITION[3]; //boundary particles final position
float V_INITIAL[3]; //initial velocity defined by the user



//controlling iteration number and simulation time
int iteration = 1; //iteration counter
float simulation_time; //in seconds
float final_time; //in seconds

//CUDA variables
int block_size;
int grid_size;

//PCISPH variables
float3* d_POSITION; //stores the pointer to the position data in the GPU
float3* d_PRED_POSITION; //stores the pointer to the predicted position data in the GPU
float3* d_VELOCITY; //stores the pointer to the velocity data in the GPU
float3* d_PRED_VELOCITY; //stores the pointer to the predicted data in the GPU
float3* d_ST_FORCE; //stores the pointer to the surface tension force data in the GPU
float3* d_VISCOSITY_FORCE; //stores the pointer to the viscosity force data in the GPU
float3* d_PRESSURE_FORCE; //stores the pointer to the pressure force data in the GPU
float3* d_NORMAL; //stores the pointer to the normal data in the GPU
float* DENSITY; //stores the pointer to the density data in the CPU
float* d_DENSITY; //stores the pointer to the density data in the GPU
float* d_PRESSURE; //stores the pointer to the pressure data in the GPU
float* d_MASS; //stores the pointer to the mass data in the GPU
int* d_TYPE; //stores the pointer to the type data in the GPU
int* d_hashtable; //stores the pointer to the hashtable data in the GPU

float vol_comp_perc; //user defined volume compression rate <- defined in section 3.3 of [2]
float dens_fluc_perc; //user defined density fluctuation rate <- defined in section 3.3 of [2]
float* d_max_force; // GPU pointer to max_force variable
float* d_max_velocity; // GPU pointer to max_velocity variable
float* d_max_rho_err; // GPU pointer to max_rho_err variable (max density error)
float* d_sum_rho_err; // GPU pointer to sum_rho_err variable (sum of all density errors across all variables to compute mean density error)
float delta_t; // time step
float max_vol_comp; // variable to stored computed value of max volume compression ( = params.rho_0 * vol_comp_perc / 100 )
float max_rho_fluc; // variable to stored computed value of max density fluctuation ( = params.rho_0 * dens_fluc_perc / 100 )
float BOUNDARY_DIAMETER; // diameter of boundary particles
float BOUNDARY_RADIUS; // radius of boundary particles
float pressure_delta; // defined in section 2.3 of [1] -> here this value is calculated without the "beta" variable, which is calculated afterwards
float max_rho_err_t_1 = 0.f; // max density error in the previous time_step
float max_rho_err = 0.f; // max density error in the current time_step (CPU memory)
bool write_pvd = true; // this tells either the program should or not write a file
char* user_results_folder = new char[256]; // user defined results folder
float save_steps; // user defined time steps to save a file

SimParams params; //host parameters

// this function reads all files in the /props folder and stores the values in the designated variables.
// If any new variable should be added or deleted in any of the props files, this function must be edited.
int fileReader() {

	//allocating memory

	char* row = new char[256]; //buffer for rows
	int row_buff_index = 0; //index for row buffer
	char* num_buffer = new char[256]; //buffer for numbers
	int num_buffer_index = 0; //index for number buffer
	float num; //stores a float variable
	float3 vec; //stores a float3 variable

	//Storing the names of varibles as they are in the files in /props folder

	char* phys_props_names[] = { "rho_0","visc_const","surface_tension_const","collision_dumping_coeff" };
	char* init_cond_names[] = {"particle_radius","mass","fluid_initial_coord","fluid_final_coord","boundary_initial_coord","boundary_final_coord","fluid_initial_velocity","maximum_volume_compression","maximum_density_fluctuation"};
	char* system_names[] = { "initial_delta_t","initial_time","final_time","neighbors_per_particle", "save_steps","results_folder"};
	
	int phys_props_size = sizeof(phys_props_names) / 8; 
	int init_cond_size = sizeof(init_cond_names) / 8;
	int system_size = sizeof(system_names) / 8;

	//storing the paths for each file

	char* phys_props_path = "./props/physical_props.txt";
	char* initial_conditions_path = "./props/initial_conditions.txt";
	char* system_path = "./props/system.txt";

	//Checking either the files exist or not -> give error and stops execution in case of error

	if (fileExists(phys_props_path) != 0) {
		std::cout << "\nERROR! Could not find physical properties file at " << phys_props_path << "\n";
		return 1;
	}

	if (fileExists(phys_props_path) != 0) {
		std::cout << "\nERROR! Could not find initial conditions file at " << phys_props_path << "\n";
		return 1;
	}

	if (fileExists(phys_props_path) != 0) {
		std::cout << "\nERROR! Could not find system names file at " << phys_props_path << "\n";
		return 1;
	}

	//reading physical properties
	std::ifstream phys_props (phys_props_path);

	for (char write2line; phys_props.get(write2line);) {
		if (phys_props.eof()) {
			break;
		}

		if (write2line == 10) {

			int i = 0;

			for (i; i < phys_props_size; i++) {
				if (strstr(row, phys_props_names[i]) != nullptr) {
					break;
				}
			}
			if (i < phys_props_size) {
				bool save_char = false;
				for (int j = 0; j < strlen(row); j++) {
					if (row[j] == 61) {
						save_char = true;
						for (int k = j; k < strlen(row); k++) {
							if (!isdigit(row[k + 1])) {
								j++;
							}
							else { break; }
						}
					}
					else if (row[j] == 59) {
						num = (float)atof(num_buffer);
						num_buffer_index = 0;
						num_buffer = new char[256];
						break;
					}
					else if ((isdigit(row[j]) || row[j] == 46 || row[j] == 45) && save_char) {
						num_buffer[num_buffer_index] = row[j];
						num_buffer_index++;
					}

				}

				if (i == 0) {
					params.rho_0 = num;
				}
				else if (i == 1) {
					params.visc_const = num;
				}
				else if (i == 2) {
					params.st_const = num;
				}
				else if (i == 3) {
					params.epsilon = num;
				}
			}
			row = new char[256];
			row_buff_index = 0;
		}
		else if (write2line != 10) {
			row[row_buff_index] = write2line;
			row_buff_index++;
		}



	}

	row = new char[256];
	row_buff_index = 0;
	phys_props.close();

	//reading initial conditions
	std::ifstream init_conds(initial_conditions_path);
	
	for (char write2line; init_conds.get(write2line);) {
		if (init_conds.eof()) {
			break;
		}

		if (write2line == 10) {

			int i = 0;

			for (i; i < init_cond_size; i++) {
				if (strstr(row, init_cond_names[i]) != nullptr) {
					break;
				}
			}
			if (i < init_cond_size) {
				if (strstr(row, "[") != nullptr) {
					bool save_char = false;
					int axis_count = 0;
					for (int j = 0; j < strlen(row); j++) {
						if (axis_count > 2) {
							axis_count = 0;
							break;
						}
						if (row[j] == 91) {
							save_char = true;
							for (int k = j; k < strlen(row); k++) {
								if (!isdigit(row[k + 1])) {
									j++;
								}
								else { break; }
							}
						}
						else if (row[j] == 44 || row[j] == 93) {
							num = (float)atof(num_buffer);
							if (axis_count == 0) {
								vec.x = num;
							} else if (axis_count == 1) {
								vec.y = num;
							}
							else if (axis_count == 2) {
								vec.z = num;
							}
							axis_count++;

							if (row[j] == 32) { 
								j++; 
								
							}

							num_buffer_index = 0;
							num_buffer = new char[256];
						}
						else if ((isdigit(row[j]) || row[j] == 46 || row[j] == 45) && save_char) {
							num_buffer[num_buffer_index] = row[j];
							num_buffer_index++;
						}
					}
				}
				else {
					bool save_char = false;
					for (int j = 0; j < strlen(row); j++) {
						if (row[j] == 61) {
							save_char = true;
							for (int k = j; k < strlen(row); k++) {
								if (!isdigit(row[k + 1])) {
									j++;
								}
								else { break; }
							}
						}
						else if (row[j] == 59) {
							num = (float)atof(num_buffer);
							num_buffer_index = 0;
							num_buffer = new char[256];
							break;
						}
						else if ((isdigit(row[j]) || row[j] == 46 || row[j] == 45) && save_char) {
							num_buffer[num_buffer_index] = row[j];
							num_buffer_index++;
						}

					}
				}


				if (i == 0) {
					PARTICLE_RADIUS = num;
				}
				else if (i == 1) {
					USER_MASS = num;
				}
				else if (i == 2) {
					F_INITIAL_POSITION[0] = vec.x;
					F_INITIAL_POSITION[1] = vec.y;
					F_INITIAL_POSITION[2] = vec.z;
				}
				else if (i == 3) {
					F_FINAL_POSITION[0] = vec.x;
					F_FINAL_POSITION[1] = vec.y;
					F_FINAL_POSITION[2] = vec.z;
				}
				else if (i == 4) {
					B_INITIAL_POSITION[0] = vec.x;
					B_INITIAL_POSITION[1] = vec.y;
					B_INITIAL_POSITION[2] = vec.z;
				}
				else if (i == 5) {
					B_FINAL_POSITION[0] = vec.x;
					B_FINAL_POSITION[1] = vec.y;
					B_FINAL_POSITION[2] = vec.z;
				}
				else if (i == 6) {
					V_INITIAL[0] = vec.x;
					V_INITIAL[1] = vec.y;
					V_INITIAL[2] = vec.z;
				}
				else if (i == 7) {
					vol_comp_perc = num;
				}
				else if (i == 8) {
					dens_fluc_perc = num;
				}
			}
			row = new char[256];
			row_buff_index = 0;
		}
		else if (write2line != 10) {
			row[row_buff_index] = write2line;
			row_buff_index++;
		}



	}

	row = new char[256];
	row_buff_index = 0;
	init_conds.close();

	std::ifstream system_vars(system_path);

	for (char write2line; system_vars.get(write2line);) {
		if (system_vars.eof()) {
			break;
		}

		if (write2line == 10) {

			int i = 0;

			for (i; i < system_size; i++) {
				if (strstr(row, system_names[i]) != nullptr) {
					break;
				}
			}
			if (i < system_size) {
				bool save_char = false;
				if (strstr(row, "\"") != nullptr) {
					for (int j = 0; j < strlen(row); j++) {
						if (row[j] == 34 && !save_char) {
							save_char = true;
							for (int k = j; k < strlen(row); k++) {
								if (row[k+1] == 32) {
									j++;
								}
								else { break; }
							}
						}
						else if (row[j] == 34 && save_char) {
							break;
						}
						else if (save_char){
							num_buffer[num_buffer_index] = row[j];
							num_buffer_index++;
						}

					}
				}
				else {
					for (int j = 0; j < strlen(row); j++) {
						if (row[j] == 61) {
							save_char = true;
							for (int k = j; k < strlen(row); k++) {
								if (!isdigit(row[k + 1])) {
									j++;
								}
								else { break; }
							}
						}
						else if (row[j] == 59) {
							num = (float)atof(num_buffer);
							num_buffer_index = 0;
							num_buffer = new char[256];
							break;
						}
						else if ((isdigit(row[j]) || row[j] == 46 || row[j] == 45) && save_char) {
							num_buffer[num_buffer_index] = row[j];
							num_buffer_index++;
						}

					}
				}
				

				if (i == 0) {
					delta_t = num;
				}
				else if (i == 1) {
					simulation_time = num;
				}
				else if (i == 2) {
					final_time = num;
				}
				else if (i == 3) {
					params.particles_per_row = (int)num;
				}
				else if (i == 4) {
					save_steps = num;
				}
				else if (i == 5) {
					user_results_folder = num_buffer;
				}

			}
			row = new char[256];
			row_buff_index = 0;
		}
		else if (write2line != 10) {
			row[row_buff_index] = write2line;
			row_buff_index++;
		}

	}



	return 0;
}

// this function initialized the execution. It creates the particles, calculates some variables and allocate memory in the GPU for the main loop.
int initialize() {
	
	//Display GPU information and checking if the program is running in a CUDA capable machine or not.

	cudaDeviceProp* prop = new cudaDeviceProp;
	gpuErrchk(cudaGetDeviceProperties(prop,0));
	std::cout << "-----------------------------------------------\n";
	std::cout << "DEVICE PROPERTIES:\n" << "Device name: " << prop->name << "\n" <<
		"Max number of threads per block: " << prop->maxThreadsPerBlock << "\n" <<
		"Total global memory: " << dround(prop->totalGlobalMem/1e9,2) << " gigabytes\n" <<
		"Registers per block: " << prop->regsPerBlock << "\n" << 
		"Shared Memory per block: " << prop->sharedMemPerBlock << " bytes\n" <<
		"-----------------------------------------------\n";

	block_size = prop->maxThreadsPerBlock; //stores the size of the thread blocks. Here its set to be the same size of the max threads per block of your GPU (1024 in the modern devices).

	max_vol_comp = params.rho_0 * vol_comp_perc / 100; 
	max_rho_fluc = params.rho_0 * dens_fluc_perc / 100;

	//If the user did not define a mass, calculate it.

	if (USER_MASS == 0) {
		MASS_calc = params.rho_0 * (float)M_PI * pow(PARTICLE_RADIUS, 3.f) / 3.f * 4.f;
	}
	else {
		MASS_calc = USER_MASS;
	}

	PARTICLE_DIAMETER = 2 * PARTICLE_RADIUS;

	// get main path of simulation
	getMainPath(main_path);

	// write path for vtu files
	strcpy(vtu_path, main_path);
	strcat(vtu_path, "/vtu");

	// write path for vtk group file
	strcpy(vtk_group_path, main_path);
	strcat(vtk_group_path, "/PCISPH.pvd");

	// create directory for vtu files
	CreateDir(vtu_path);

	float VOLUME = 1;
	const int SIMULATION_DIMENSION = 3; //3 for a 3D simulation

	// Get number per dimension (NPD) of FLUID particles for hexadecimal packing (assuming use of makeprism function)
	
	int NPD[3]; //Number per dimension

	for (int i = 0; i < 3; i++) {
		if (i == 1) {
			NPD[i] = static_cast<int>(floor((F_FINAL_POSITION[i] - F_INITIAL_POSITION[i]) / (sqrt(3.f) / 2.f * PARTICLE_DIAMETER)));
			VOLUME = VOLUME * (F_FINAL_POSITION[i] - F_INITIAL_POSITION[i]);
		}
		else {
			NPD[i] = static_cast<int>(floor((F_FINAL_POSITION[i] - F_INITIAL_POSITION[i]) / PARTICLE_DIAMETER));
			VOLUME = VOLUME * (F_FINAL_POSITION[i] - F_INITIAL_POSITION[i]);
		}
	}

	//Writing NPD to device
	int* D_NPD; //Device pointer to NPD variable
	gpuErrchk(cudaMalloc((void**)&D_NPD, SIMULATION_DIMENSION * sizeof(float))); //Allocate GPU memory
	gpuErrchk(cudaMemcpy(D_NPD, NPD, SIMULATION_DIMENSION * sizeof(float), cudaMemcpyHostToDevice)); //Write NPD to D_NPD

	params.N = NPD[0] * NPD[1] * NPD[2]; //number of fluid particles
	int SIM_SIZE = params.N * SIMULATION_DIMENSION; //size of the fluid part of the simulation
	const int x = 40; // Number of particles inside the smoothing length
	params.h = powf(3.f * VOLUME * x / (4.f * (float)M_PI * params.N), 1.f / 3.f); //smoothing length
	params.invh = 1 / params.h; // inverse of smoothing length (this is calculated to make things faster in the main loop)

	float3 f_initial; //initial position taking in account the offset of the particle radius
	f_initial.x = F_INITIAL_POSITION[0] + PARTICLE_RADIUS; 
	f_initial.y = F_INITIAL_POSITION[1] + PARTICLE_RADIUS;
	f_initial.z = F_INITIAL_POSITION[2] + PARTICLE_RADIUS;

	size_t bytes_fluid_particles = SIM_SIZE * sizeof(float);

	float3* FLUID_POSITIONS; //host pointer (CPU memory)
	FLUID_POSITIONS = (float3*)malloc(bytes_fluid_particles); //allocating CPU memory

	float3* D_FLUID_POSITIONS; //device pointer (GPU memory)
	gpuErrchk(cudaMalloc((void**)&D_FLUID_POSITIONS, bytes_fluid_particles)); //allocating GPU memory

	// grid -> number of blocks
	// block -> number of threads

	grid_size = params.N / block_size + 1; //defining number of blocks

	//generate locations for each particle
	//check "particle_positions.cuh" file in /lib folder for more details
	makePrism << <grid_size, block_size >> > (D_FLUID_POSITIONS, PARTICLE_DIAMETER, f_initial, D_NPD, params.N);

	BOUNDARY_DIAMETER = params.h/2; //defining the diameter of a boundary particle as stated in section 3.2 in [2]
	BOUNDARY_RADIUS = params.h/4;

	// Get number per dimension (NPD) of BOUNDARY particles without compact packing (assuming use of makebox function)
	for (int i = 0; i < 3; i++) {
		NPD[i] = static_cast<int>(ceil((B_FINAL_POSITION[i] - B_INITIAL_POSITION[i]) / BOUNDARY_DIAMETER)) + 2;

	}

	params.B = NPD[0] * NPD[1] * NPD[2] - (NPD[0] - 2) * (NPD[1] - 2) * (NPD[2] - 2); //Number of boundary particles
	SIM_SIZE = NPD[0] * NPD[1] * NPD[2] * SIMULATION_DIMENSION;

	float3 b_initial; //initial position taking in account the offset of the boundary particle radius
	b_initial.x = B_INITIAL_POSITION[0] - BOUNDARY_RADIUS;
	b_initial.y = B_INITIAL_POSITION[1] - BOUNDARY_RADIUS;
	b_initial.z = B_INITIAL_POSITION[2] - BOUNDARY_RADIUS;
	float3 b_final; //final position taking in account the offset of the boundary particle radius
	b_final.x = b_initial.x + BOUNDARY_DIAMETER * (NPD[0] - 1);
	b_final.y = b_initial.y + BOUNDARY_DIAMETER * (NPD[1] - 1);
	b_final.z = b_initial.z + BOUNDARY_DIAMETER * (NPD[2] - 1);

	size_t bytes_boundary_particles = SIM_SIZE * sizeof(float); // number of bytes the boundary particles are occupying
	float3* BOUNDARY_POSITIONS; //host pointer (CPU memory)
	BOUNDARY_POSITIONS = (float3*)malloc(bytes_boundary_particles); //allocate memory in the host

	float3* D_BOUNDARY_POSITIONS; //device pointer (GPU memory)
	gpuErrchk(cudaMalloc((void**)&D_BOUNDARY_POSITIONS, bytes_boundary_particles)); //allocate memory in the device

	// this function makes an empty box with walls with 1 particle of thickness
	// check "particle_positions.cuh" file in /lib folder for more details
	makeBox(D_BOUNDARY_POSITIONS, BOUNDARY_DIAMETER, b_initial, b_final, block_size, D_NPD,NPD, SIMULATION_DIMENSION);

	params.T = params.N + params.B; //Total number of particles

	//writing particle position memory from GPU to CPU (note the "cudaMemcpyDeviceToHost" statement in the functions below)
	gpuErrchk(cudaMemcpy(FLUID_POSITIONS, D_FLUID_POSITIONS, bytes_fluid_particles, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(BOUNDARY_POSITIONS, D_BOUNDARY_POSITIONS, bytes_boundary_particles, cudaMemcpyDeviceToHost));

	// Free GPU memory for fluid particles (this memory will be reallocated with another name soon)
	cudaFree(D_FLUID_POSITIONS);

	// Getting hashtable size
	uint tmp_elev = 19;
	params.hashtable_size = unsignedIntPow(2, tmp_elev);
	while (params.T > params.hashtable_size) {
		tmp_elev++;
		params.hashtable_size = unsignedIntPow(2, tmp_elev);
	}

	// Transfering params to GPU
	
	gpuErrchk(cudaMemcpyToSymbol(&d_params, &params, sizeof(SimParams)))

	int* hashtable = new int[params.hashtable_size * params.particles_per_row];

	//this loop creates an empty hashtable (full of -1s)
	for (uint i = 0; i < params.hashtable_size; ++i) {
		for (uint j = 0; j < params.particles_per_row; j++) {
			hashtable[i * params.particles_per_row + j] = -1;
		}
	}

	//allocating 2D memory for hashtable 
	gpuErrchk(cudaMallocPitch(&params.d_hashtable, &params.pitch, params.particles_per_row * sizeof(int), params.hashtable_size));
	//writing clean hashtable to GPU
	gpuErrchk(cudaMemcpy2D(params.d_hashtable, params.pitch, hashtable, params.particles_per_row * sizeof(int), params.particles_per_row * sizeof(int), params.hashtable_size, cudaMemcpyHostToDevice));

	grid_size = params.B / block_size + 1;
	//this function makes a functional hashtable
	hashParticlePositions << <grid_size, block_size >> > (D_BOUNDARY_POSITIONS, params.B);
	gpuErrchk(cudaPeekAtLastError()); // this is for checking if there was any error during the kernel execution
	gpuErrchk(cudaDeviceSynchronize());
	float* d_boundary_mass; //pointer to device memory of boundary "fake" mass ( or psi )
	gpuErrchk(cudaMalloc((void**)&d_boundary_mass, params.B * sizeof(float)));

	// calculates "fake" mass (or psi) for each boundary particle as state in [3]
	// check "particle_parameters.cuh" file in /lib folder for more details
	boundaryPsi << <grid_size, block_size >> > (d_boundary_mass,D_BOUNDARY_POSITIONS);

	float* boundary_mass = (float*)malloc(params.B * sizeof(float)); //CPU pointer to boundary mass
	//copy boundary mass from GPU to CPU
	gpuErrchk(cudaMemcpy(boundary_mass, d_boundary_mass, (size_t)params.B * sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_boundary_mass));

	float3* d_boundary_normal; //device pointer for boundary normal
	gpuErrchk(cudaMalloc((void**)&d_boundary_normal, params.B * 3 * sizeof(float)));

	// calculate normal for boundary particles
	// check "particle_parameters.cuh" file in /lib folder for more details
	boundaryNormal << <grid_size, block_size >> > (d_boundary_normal, D_BOUNDARY_POSITIONS, b_initial, b_final);

	float3* boundary_normal = (float3*)malloc(params.B * sizeof(float3)); //pointer for CPU memory of boundary normal
	// copying boundary normal memory from GPU to CPU
	gpuErrchk(cudaMemcpy(boundary_normal, d_boundary_normal, (size_t)params.B * 3 * sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_boundary_normal)); //cleaning GPU memory of boundary normal (this will be reallocated later with another name)

	// writing boundary vtu file

	float** boundary_point_data[] = { &boundary_mass };
	int size_pointData = sizeof(boundary_point_data) / sizeof(double);
	float3** boundary_vectorData[] = { &boundary_normal };
	int size_vectorData = sizeof(boundary_vectorData) / sizeof(double);

	std::string boundary_pointDataNames[] = { "psi" };
	std::string boundary_vectorDataNames[] = { "normal" };

	VTU_Writer(main_path, iteration, BOUNDARY_POSITIONS, params.B, boundary_point_data, boundary_vectorData, boundary_pointDataNames, boundary_vectorDataNames, size_pointData, size_vectorData, vtu_fullpath, 1);

	cudaFree(params.d_hashtable); //cleaning GPU from hashtable memory

	cudaFree(D_BOUNDARY_POSITIONS); //cleaning GPU from boundary particle memory

	// calculating pressure delta (without the beta variable) as stated in section 2.3 of [1]

	int count = 0;
	float min_r = std::numeric_limits<float>::infinity();
	int selected_index;
	int tmp_size = static_cast<int>(ceil((2 * (params.h + PARTICLE_DIAMETER)) / PARTICLE_DIAMETER));
	float3* tmp_points = (float3*)malloc(tmp_size * tmp_size * tmp_size * 3 * sizeof(float));

	// generating fake particle positions without any packing method (the same is done in [5])
	for (float i = -params.h - PARTICLE_DIAMETER; i <= params.h + PARTICLE_DIAMETER; i += PARTICLE_DIAMETER) {
		for (float j = -params.h - PARTICLE_DIAMETER; j <= params.h + PARTICLE_DIAMETER; j += PARTICLE_DIAMETER) {
			for (float k = -params.h - PARTICLE_DIAMETER; k <= params.h + PARTICLE_DIAMETER; k += PARTICLE_DIAMETER) {
				tmp_points[count].x = i;
				tmp_points[count].y = j;
				tmp_points[count].z = k;
				count++;
				float r = sqrt(i*i+j*j+k*k);
				if (r < min_r) {
					min_r = r;
					selected_index = count;
				}
			}
		}
	}

	float3 selected_point = tmp_points[selected_index];
	float3 r_vector;
	float r;
	float3 Grad_W;
	Grad_W.x = 0.f;
	Grad_W.y = 0.f;
	Grad_W.z = 0.f;
	float dot_Grad_W = 0.f;

	// summation of the calculated kernel gradients
	for (int i = 0; i < count; i++) {
		r_vector.x = tmp_points[i].x - selected_point.x;
		r_vector.y = tmp_points[i].y - selected_point.y;
		r_vector.z = tmp_points[i].z - selected_point.z;
		r = sqrt(r_vector.x* r_vector.x + r_vector.y* r_vector.y + r_vector.z* r_vector.z);

		if (r <= params.h) {
			float3 inst_Grad_W = Poly6_Gradient(selected_index, i, tmp_points, r, params.h, params.invh);

			Grad_W.x += inst_Grad_W.x;
			Grad_W.y += inst_Grad_W.y;
			Grad_W.z += inst_Grad_W.z;

			dot_Grad_W += dot_product(inst_Grad_W, inst_Grad_W);
		}

	}

	pressure_delta = -dot_product(Grad_W, Grad_W) - dot_Grad_W;

	//Initializing main particle variables

	//Defining and allocating main position variable
	
	float3* POSITION = (float3*)malloc(params.T*sizeof(float3));
	for (uint i = 0; i < params.N; i++) {
		POSITION[i].x = FLUID_POSITIONS[i].x;
		POSITION[i].y = FLUID_POSITIONS[i].y;
		POSITION[i].z = FLUID_POSITIONS[i].z;
	}

	for (uint i = params.N; i < params.T; i++) {
		POSITION[i].x = BOUNDARY_POSITIONS[i - params.N].x;
		POSITION[i].y = BOUNDARY_POSITIONS[i - params.N].y;
		POSITION[i].z = BOUNDARY_POSITIONS[i - params.N].z;
	}

	free(BOUNDARY_POSITIONS);
	free(FLUID_POSITIONS);

	
	gpuErrchk(cudaMalloc((void**)&d_POSITION, 3*params.T*sizeof(float)));
	gpuErrchk(cudaMemcpy(d_POSITION, POSITION, 3*params.T*sizeof(float), cudaMemcpyHostToDevice));

	//Allocating memory for predicted positions and copying previous position vectors
	gpuErrchk(cudaMalloc((void**)&d_PRED_POSITION, 3 * params.T * sizeof(float)));
	gpuErrchk(cudaMemcpy(d_PRED_POSITION, POSITION, 3 * params.T * sizeof(float), cudaMemcpyHostToDevice));

	//Allocating memory for predicted velocity
	gpuErrchk(cudaMalloc((void**)&d_PRED_VELOCITY, 3 * params.N * sizeof(float)));

	//Defining and allocating main velocity variable
	
	float3* VELOCITY = (float3*)malloc(params.N*sizeof(float3));
	for (uint i = 0; i < params.N; i++) {
		VELOCITY[i].x = V_INITIAL[0];
		VELOCITY[i].y = V_INITIAL[1];
		VELOCITY[i].z = V_INITIAL[2];
	}

	gpuErrchk(cudaMalloc((void**)&params.d_VELOCITY,  params.N*sizeof(float3)));
	gpuErrchk(cudaMemcpy(params.d_VELOCITY, VELOCITY,  params.N*sizeof(float3), cudaMemcpyHostToDevice));

	//Defining and allocating main st force variable
	
	float3* ST_FORCE = (float3*)malloc( params.N*sizeof(float3));
	for (uint i = 0; i < params.N; i++) {
		ST_FORCE[i].x = 0.f;
		ST_FORCE[i].y = 0.f;
		ST_FORCE[i].z = 0.f;
	}

	gpuErrchk(cudaMalloc((void**)&params.d_ST_FORCE,  params.N*sizeof(float3)));

	//Defining and allocating main viscosity force variable
	float3* VISCOSITY_FORCE = (float3*)malloc( params.N*sizeof(float3));
	for (uint i = 0; i < params.N; i++) {
		VISCOSITY_FORCE[i].x = 0.f;
		VISCOSITY_FORCE[i].y = 0.f;
		VISCOSITY_FORCE[i].z = 0.f;
	}

	gpuErrchk(cudaMalloc((void**)&params.d_VISCOSITY_FORCE,  params.N*sizeof(float3)));

	//Defining and allocating main pressure force variable
	float3* PRESSURE_FORCE = (float3*)malloc( params.N*sizeof(float3));
	for (uint i = 0; i < params.N; i++) {
		PRESSURE_FORCE[i].x = 0.f;
		PRESSURE_FORCE[i].y = 0.f;
		PRESSURE_FORCE[i].z = 0.f;
	}

	gpuErrchk(cudaMalloc((void**)&params.d_PRESSURE_FORCE,  params.N*sizeof(float3)));

	//Defining and allocating main normal variable
	float3* NORMAL = (float3*)malloc(3*params.T*sizeof(float3));
	for (uint i = 0; i < params.N; i++) {
		NORMAL[i].x = 0.f;
		NORMAL[i].y = 0.f;
		NORMAL[i].z = 0.f;
	}	

	for (uint i = params.N; i < params.T; i++) {
		NORMAL[i].x = boundary_normal[i - params.N].x;
		NORMAL[i].y = boundary_normal[i - params.N].y;
		NORMAL[i].z = boundary_normal[i - params.N].z;
	}

	free(boundary_normal);

	gpuErrchk(cudaMalloc((void**)&params.d_NORMAL, params.T*sizeof(float3)));
	gpuErrchk(cudaMemcpy(params.d_NORMAL, NORMAL, params.T*sizeof(float3), cudaMemcpyHostToDevice));

	//Defining and allocating main density array
	float* DENSITY = (float*)malloc(params.N * sizeof(float));
	for (uint i = 0; i < params.N; i++) {
		DENSITY[i] = 0.f;
	}

	gpuErrchk(cudaMalloc((void**)&params.d_DENSITY, params.N * sizeof(float)));
	gpuErrchk(cudaMemcpy(params.d_DENSITY, DENSITY, params.N * sizeof(float), cudaMemcpyHostToDevice));

	//Defining and allocating main pressure array
	float* PRESSURE = (float*)malloc(params.N * sizeof(float));
	for (uint i = 0; i < params.N; i++) {
		PRESSURE[i] = 0;
	}

	gpuErrchk(cudaMalloc((void**)&params.d_PRESSURE, params.N * sizeof(float)));

	//Defining and allocating main mass array
	
	float* MASS = (float*)malloc(params.T * sizeof(float));
	for (uint i = 0; i < params.N; i++) {
		MASS[i] = MASS_calc;
	}

	for (uint i = params.N; i < params.T; i++) {
		MASS[i] = boundary_mass[i - params.N];
	}

	free(boundary_mass);
	
	gpuErrchk(cudaMalloc((void**)&params.d_MASS, params.T * sizeof(float)));
	gpuErrchk(cudaMemcpy(params.d_MASS, MASS, params.T * sizeof(float), cudaMemcpyHostToDevice));

	//Defining and allocating main type array (0 if fluid, 1 if boundary)
	int* TYPE = (int*)malloc(params.T * sizeof(int));
	for (uint i = 0; i < params.N; i++) {
		TYPE[i] = 0;
	}

	for (uint i = params.N; i < params.T; i++) {
		TYPE[i] = 1;
	}

	gpuErrchk(cudaMalloc((void**)&params.d_TYPE, params.T * sizeof(int)));
	gpuErrchk(cudaMemcpy(params.d_TYPE, TYPE, params.T * sizeof(int), cudaMemcpyHostToDevice));

	//Defining and allocating memory to store max density error
	gpuErrchk(cudaMalloc((void**)&d_max_rho_err, sizeof(float)));

	//Defining and allocating memory to store max force value
	gpuErrchk(cudaMalloc((void**)&d_max_force, sizeof(float)));

	//Defining and allocating memory to store max velocity value
	gpuErrchk(cudaMalloc((void**)&d_max_velocity, sizeof(float)));

	//Defining and allocating memory to store summation of density errors to calculate average error
	gpuErrchk(cudaMalloc((void**)&d_sum_rho_err, sizeof(float)));

	//defining gravity vector
	params.gravity.x = 0.f;
	params.gravity.y = -9.81f;
	params.gravity.z = 0.f;

	//Defining variables to write VTU files
	float** pointData[] = { &DENSITY, &PRESSURE }; // here the CPU pointers to the FLOAT variables that you want to write in the VTU must be defined
	float3** vectorData[] = { &VELOCITY, &PRESSURE_FORCE, &VISCOSITY_FORCE, &ST_FORCE  }; // here the CPU pointers to the VEC3D variables that you want to write in the VTU must be defined

	size_pointData = sizeof(pointData) / 8;
	size_vectorData = sizeof(vectorData) / 8;

	VTU_Writer(vtu_path, iteration, POSITION, params.N, pointData, vectorData, pointDataNames, vectorDataNames, size_pointData, size_vectorData, vtu_fullpath);

	VTK_Group(vtk_group_path, vtu_fullpath, simulation_time);

	// Initialize main hashtable

	//allocating memory for GPU hashtable
	gpuErrchk(cudaMallocPitch(&params.d_hashtable, &params.pitch, params.particles_per_row * sizeof(int), params.hashtable_size));

	writeTimeKeeper(main_path,max_rho_err); //time keeper file with values for time, iteration and max density error

	cudaMemcpyToSymbol(&d_params,&params,sizeof(SimParams),cudaMemcpyHostToDevice);

	std::cout << params.N << " Fluid particles\n"
		<< params.B << " Boundary particles\n"
		<< "Total of " << params.T << " particles.\n"
		<< "Smoothing radius = " << params.h << " m.\n"
		<< "hashtable size = " << params.hashtable_size << "\n";

	gpuErrchk(cudaPeekAtLastError()); // this is for checking if there was any error during the kernel execution
	gpuErrchk(cudaDeviceSynchronize());

	return 0;
}

// here is where the magic happens
// comments with -> refer to the same lines of the pseudo code in Algorithm 2 in [2]
// -> while animating do
int mainLoop() {
	
	// -> for each particle i,b do
	//	-> find neighbors Ni,b(t)

	// here the hashtable is initialized and reset
	grid_size = params.hashtable_size / block_size + 1;
	hashtableReset << <grid_size, block_size >> >  ();
	
	// then a new hashtable is created
	grid_size = params.T / block_size + 1;
	hashParticlePositions << <grid_size, block_size >> > ();

	// -> for each particle i do

	// here there are tow more step than the pseudo algorithm:
	
	// calculate density
	grid_size = params.N / block_size + 1;
	DensityCalc << <grid_size, block_size >> > ();

	// and the normal for each fluid particle
	
	fluidNormal << <grid_size, block_size >> > ();
	
	// -> compute forces Fi for viscosity and surface tension (gravity is only accounted later)
	nonPressureForces << <grid_size, block_size >> > ();
	
	// -> set pressure pi(t) = 0 

	resetPressure << <grid_size, block_size >> > ();
	// here the step to set the pressure force value as 0 is ignored as it is done on later steps

	// calculate the pressure coefficient as in Equation 8 of [1]
	float pressure_coeff = -1 / (2 * powf(MASS_calc * delta_t / params.rho_0, 2) * pressure_delta);
	
	gpuErrchk(cudaPeekAtLastError()); // this is for checking if there was any error during the kernel execution
	gpuErrchk(cudaDeviceSynchronize()); 

	int _k_ = 0; // defined with underscores to prevent overwritting 
	// -> while k < 3 do
	while (_k_ < 3) {
		
		// -> for each particle i do
		//  -> predicit velocity 
		//  -> predicit position 
		grid_size = params.N / block_size + 1;
		positionAndVelocity << <grid_size, block_size >> > (d_params.d_PRED_POSITION,d_params.d_PRED_VELOCITY,d_params.d_POSITION, d_params.d_VELOCITY, delta_t);

		// -> predict world collision
		collisionHandler << <grid_size, block_size >> > ();
		
		// reset and create new hashtable
		grid_size = params.hashtable_size / block_size + 1;
		hashtableReset << <grid_size, block_size >> > ();
		grid_size = params.T / block_size + 1;
		hashParticlePositions << <grid_size, block_size >> > ();

		// update distances to neighbors is unnecessary here

		// -> predict density
		grid_size = params.N / block_size + 1;
		DensityCalc << <grid_size, block_size >> > ();

		// -> predict density variation and -> update pressure
		PressureCalc << <grid_size, block_size >> > (pressure_coeff);

		// -> compute pressure force

		PressureForceCalc << <grid_size, block_size >> > ();

		_k_++;
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// -> compute new velocity and compute new position
	positionAndVelocity << <grid_size, block_size >> > (d_params.d_POSITION, d_params.d_VELOCITY, d_params.d_POSITION, d_params.d_VELOCITY, delta_t);

	// -> compute new world collision
	collisionHandler << <grid_size, block_size >> > ();

	// -> adapt time step

	// criterias for changes in delta_t value according to session 3.3 of [2]

	// getting max velocity, max force, max density error and average density error
	max_rho_err_t_1 = max_rho_err;
	float max_velocity = 0.f;
	float max_force = 0.f;
	float sum_rho_err = 0.f;
	resetValues<<<1,1>>>(d_max_velocity, d_max_force, d_sum_rho_err, d_max_rho_err);
	grid_size = params.N / block_size + 1;
	getMaxVandF << <grid_size, block_size >> > (d_max_force, d_max_velocity, d_sum_rho_err,d_max_rho_err);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaMemcpy(&max_velocity, d_max_velocity, sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&max_force, d_max_force, sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&sum_rho_err, d_sum_rho_err, sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&max_rho_err, d_max_rho_err, sizeof(float), cudaMemcpyDeviceToHost));
	
	float avg_rho_err = sum_rho_err / params.N;

	// criterias for delta_t increase

	bool criteria1 = 0.19f * sqrt(params.h / max_force) > delta_t;
	bool criteria2 = max_rho_err < 4.5f * max_vol_comp;
	bool criteria3 = avg_rho_err < 0.9f * max_vol_comp;
	bool criteria4 = 0.39f * (params.h/max_velocity) > delta_t;

	if (criteria1 && criteria2 && criteria3 && criteria4) {
		delta_t += delta_t * 0.2f / 100;
	}

	// criterias for delta_t decrease

	criteria1 = 0.2f * sqrt(params.h / max_force) < delta_t;
	criteria2 = max_rho_err > 5.5f * max_vol_comp;
	criteria3 = avg_rho_err > max_vol_comp;
	criteria4 = 0.4f * (params.h / max_velocity) <= delta_t;

	if (criteria1 || criteria2 || criteria3 || criteria4) {
		delta_t -= delta_t * 0.2f / 100;
	}

	// criterias for shock handling

	criteria1 = max_rho_err - max_rho_err_t_1 > 8 * max_vol_comp;
	criteria2 = max_rho_err > max_rho_fluc;
	criteria3 = 0.45f * (params.h/max_velocity) < delta_t;

	if (criteria1 || criteria2 || criteria3) {

		//get last iteration greater or equal to 2
		int last_iter = getLastIter(main_path);
		char* iter_path = new char[100];
		char* num_buffer = new char[32];
		while (iteration - last_iter < 2) {
			itoa(last_iter, num_buffer, 10);
			strcpy(iter_path, vtu_path);
			strcat(iter_path, "/iter");
			strcat(iter_path, num_buffer);
			strcat(iter_path, ".vtu");
			remove(iter_path);
			last_iter = getLastIter(main_path);
			num_buffer = new char[32];
			iter_path = new char[100];
		}

		std::cout << "\n\nSHOCK DETECTED! RETURNING " << iteration - last_iter << " ITERATIONS!\n" << std::endl;
		write_pvd = false;
		//SHOCK DETECTED

		delta_t -= delta_t * 0.5f;

		iteration = last_iter;
		if (iteration <= 0) {
			std::cout << "\nIMPOSSIBLE TO RETURN 2 ITERATIONS! TERMINATING SIMULATION\n" << std::endl;
			return 1;
		}

		float3* position = (float3*)malloc(params.N * sizeof(float3));
		float3* velocity = (float3*)malloc(params.N * sizeof(float3));

		itoa(iteration, num_buffer, 10);
		strcpy(iter_path, vtu_path);
		strcat(iter_path, "/iter");
		strcat(iter_path, num_buffer);
		strcat(iter_path, ".vtu");

		//read VTU file to go to the required step backwards
		readVTU(iter_path, position, velocity); 

		//get the correct time of the previous iteration 
		getNewSimTime(main_path);
		//edit PVD (group) file with the correct information
		rewritePVD(main_path);


		gpuErrchk(cudaMemcpy(d_POSITION, position, 3 * params.N * sizeof(float), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_VELOCITY, velocity, 3 * params.N * sizeof(float), cudaMemcpyHostToDevice));

		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		return 0;
	}

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	if (simulation_time + delta_t > final_time) {
		simulation_time = final_time;
	}
	else {
		simulation_time += delta_t;
	}
	
	iteration++;

	writeTimeKeeper(main_path,max_rho_err);

	return 0;
}

// This function writes VTU files using multiple CPU cores
void multiprocessor_writer() {

	char buf[1024];
	itoa(iteration, buf, 10);
	strcpy(vtu_fullpath, vtu_path);
	strcat(vtu_fullpath, "/iter");
	strcat(vtu_fullpath, buf);
	strcat(vtu_fullpath, ".vtu");

	std::future<void> write_vtu;

	float3* write_position = (float3*)malloc(params.N * sizeof(float3));
	float3* write_velocity = (float3*)malloc(params.N * sizeof(float3));
	float3* write_p_force = (float3*)malloc(params.N * sizeof(float3));
	float3* write_st_force = (float3*)malloc(params.N * sizeof(float3));
	float3* write_v_force = (float3*)malloc(params.N * sizeof(float3));
	float* write_pressure = (float*)malloc(params.N * sizeof(float));
	float* write_density = (float*)malloc(params.N * sizeof(float));

	void* d_params_gpu_pptr;
	cudaGetSymbolAddress(&d_params_gpu_pptr, &d_params);
	SimParams* d_params_gpu_ptr = (SimParams*)d_params_gpu_pptr;
	SimParams d_params_gpu = d_params_gpu_ptr[0];

	gpuErrchk(cudaMemcpy(write_position, d_POSITION, params.N * sizeof(float3), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(write_velocity, d_VELOCITY, params.N * sizeof(float3), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(write_p_force, d_params_gpu.d_PRESSURE_FORCE, params.N * sizeof(float3), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(write_st_force, d_params_gpu.d_ST_FORCE, params.N * sizeof(float3), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(write_v_force, d_params_gpu.d_VISCOSITY_FORCE, params.N * sizeof(float3), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(write_density, d_params_gpu.d_DENSITY, params.N * sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(write_pressure, d_params_gpu.d_PRESSURE, params.N * sizeof(float), cudaMemcpyDeviceToHost));

	float** pointData[] = { &write_density, &write_pressure };
	float3** vectorData[] = { &write_velocity, &write_p_force, &write_v_force, &write_st_force };

	int size_pointData = sizeof(pointData) / 8;
	int size_vectorData = sizeof(vectorData) / 8;

	write_vtu = std::async(std::launch::async, VTU_Writer, vtu_path, iteration, write_position, params.N, pointData, vectorData, pointDataNames, vectorDataNames, size_pointData, size_vectorData, vtu_fullpath,2);
		
	if (write_pvd == true) {
		strcpy(buf, vtu_fullpath);
		VTK_Group(vtk_group_path, buf, simulation_time);
	}
	write_pvd = true;

	return;
}