#include "particle_positions.cuh"
#include "utilities.cuh"
#include "VTK.cuh"
#include "hashing.cuh"
#include "particle_parameters.cuh"
#include "global_variables.cuh"
#include <math.h>
#include <future> 
#include <chrono>

//declaration of all global variables that are going to be used in this file
char main_path[1024];
char vtk_group_path[1024];
char vtu_fullpath[1024];
char vtu_path[1024];
float** pointData[2];
vec3d** vectorData[5];
std::string pointDataNames[2];
std::string vectorDataNames[5];
int size_pointData;
int size_vectorData;

vec3d* POSITION;
vec3d* d_POSITION;
vec3d* VELOCITY;
vec3d* d_VELOCITY;
vec3d* ST_FORCE;
vec3d* d_ST_FORCE;
vec3d* VISCOSITY_FORCE;
vec3d* d_VISCOSITY_FORCE;
vec3d* PRESSURE_FORCE;
vec3d* d_PRESSURE_FORCE;
vec3d* NORMAL;
vec3d* d_NORMAL;
float* DENSITY;
float* d_DENSITY;
float* PRESSURE;
float* d_PRESSURE;
float* MASS;
float* d_MASS;
int* TYPE;
int* d_TYPE;
int* hashtable;
int* d_hashtable;
vec3d gravity;

//initial conditions
const float PARTICLE_RADIUS = 0.011f;
const float MASS_calc = (float)M_PI * -pow(PARTICLE_RADIUS, 3.f) / 3.f * 4.f;
const float PARTICLE_DIAMETER = 2 * PARTICLE_RADIUS;
const float F_INITIAL_POSITION[3] = { 0.f,0.f,0.f }; //Fluid particles initial position
const float F_FINAL_POSITION[3] = { 0.5f,1.f,0.5f }; //Fluid particles final position
const float B_INITIAL_POSITION[3] = { 0.f,0.f,0.f }; //Boundary particles final position
const float B_FINAL_POSITION[3] = { 1.f,1.f,1.f }; //Boundary particles final position

//physical constants
const float rho_0 = 1000.f;
const float visc_const = 0.0010518f;
const float st_const = 0.0728f;

//controlling iteration number and simulation time
int iteration = 1;
float simulation_time = 0.f; //in seconds
float final_time = 10.f; //in seconds

//number of particles
int N; //fluid particles
int B; //bondary particles
int T; //total particles

//variables for hashtable
size_t pitch;
const int particles_per_row = 200;
int hashtable_size;
//const int n_p_neighbors = 8000; //in case of memory failure, raise this number

//simulation parameters
float invh;
float h;

//CUDA variables
int block_size = 1024;
int grid_size;

int initialize() {

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
	const int SIMULATION_DIMENSION = 3;

	// Get number per dimension (NPD) of FLUID particles for hexadecimal packing (assuming use of makeprism function)
	int NPD[3];
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

	//Passing NPD to device
	int* D_NPD;
	gpuErrchk(cudaMalloc((void**)&D_NPD, SIMULATION_DIMENSION * sizeof(float)));
	gpuErrchk(cudaMemcpy(D_NPD, NPD, SIMULATION_DIMENSION * sizeof(float), cudaMemcpyHostToDevice));

	N = NPD[0] * NPD[1] * NPD[2]; //number of fluid particles
	int SIM_SIZE = N * SIMULATION_DIMENSION;
	const int x = 40; // Number of particles inside the smoothing length
	h = powf(3.f * VOLUME * x / (4.f * (float)M_PI * N), 1.f / 3.f);
	//h = 0.02;
	invh = 1 / h;

	//const float boundary_radius = h/4;
	//const float boundary_diameter = h/2;

	//cudaError_t cudaStatus;
	//cudaStatus = cudaSetDevice(0);

	vec3d f_initial;
	f_initial.x = F_INITIAL_POSITION[0] + PARTICLE_RADIUS;
	f_initial.y = F_INITIAL_POSITION[1] + PARTICLE_RADIUS;
	f_initial.z = F_INITIAL_POSITION[2] + PARTICLE_RADIUS;

	size_t bytes_fluid_particles = SIM_SIZE * sizeof(float);

	vec3d* FLUID_POSITIONS; //host pointer
	FLUID_POSITIONS = (vec3d*)malloc(bytes_fluid_particles);

	vec3d* D_FLUID_POSITIONS; //device pointer
	gpuErrchk(cudaMalloc((void**)&D_FLUID_POSITIONS, bytes_fluid_particles));

	// grid -> number of blocks
	// block -> number of threads

	grid_size = N / block_size + 1;

	//generate locations for each particle
	makePrism << <grid_size, block_size >> > (D_FLUID_POSITIONS, PARTICLE_DIAMETER, f_initial, D_NPD, N);

	float BOUNDARY_DIAMETER = h/2;
	float BOUNDARY_RADIUS = h/4;

	// Get number per dimension (NPD) of BOUNDARY particles without compact packing (assuming use of makebox function)
	for (int i = 0; i < 3; i++) {
		NPD[i] = static_cast<int>(ceil((B_FINAL_POSITION[i] - B_INITIAL_POSITION[i]) / BOUNDARY_DIAMETER)) + 2;

	}

	//copy new NPD to device memory
	gpuErrchk(cudaMemcpy(D_NPD, NPD, SIMULATION_DIMENSION * sizeof(float), cudaMemcpyHostToDevice));

	B = NPD[0] * NPD[1] * NPD[2] - (NPD[0] - 2) * (NPD[1] - 2) * (NPD[2] - 2); //Number of boundary particles
	SIM_SIZE = NPD[0] * NPD[1] * NPD[2] * SIMULATION_DIMENSION;

	vec3d b_initial;
	b_initial.x = B_INITIAL_POSITION[0] - BOUNDARY_RADIUS;
	b_initial.y = B_INITIAL_POSITION[1] - BOUNDARY_RADIUS;
	b_initial.z = B_INITIAL_POSITION[2] - BOUNDARY_RADIUS;
	vec3d b_final;
	b_final.x = b_initial.x + BOUNDARY_DIAMETER * (NPD[0] - 1);
	b_final.y = b_initial.y + BOUNDARY_DIAMETER * (NPD[1] - 1);
	b_final.z = b_initial.z + BOUNDARY_DIAMETER * (NPD[2] - 1);

	//printf("[%g %g %g] [%g %g %g]\n", b_final.x, b_final.y, b_final.z, B_FINAL_POSITION[0] + BOUNDARY_RADIUS, B_FINAL_POSITION[1] + BOUNDARY_RADIUS, B_FINAL_POSITION[2] + BOUNDARY_RADIUS);

	size_t bytes_boundary_particles = SIM_SIZE * sizeof(float);
	vec3d* BOUNDARY_POSITIONS; //host pointer
	BOUNDARY_POSITIONS = (vec3d*)malloc(bytes_boundary_particles); //allocate memory in the host

	vec3d* D_BOUNDARY_POSITIONS; //device pointer
	gpuErrchk(cudaMalloc((void**)&D_BOUNDARY_POSITIONS, bytes_boundary_particles)); // allocate memory in the device

	makeBox(D_BOUNDARY_POSITIONS, BOUNDARY_DIAMETER, b_initial, b_final, block_size, D_NPD);

	T = N + B; //Total number of particles

	gpuErrchk(cudaMemcpy(FLUID_POSITIONS, D_FLUID_POSITIONS, bytes_fluid_particles, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(BOUNDARY_POSITIONS, D_BOUNDARY_POSITIONS, bytes_boundary_particles, cudaMemcpyDeviceToHost));

	// Free GPU memory for fluid particles
	cudaFree(D_FLUID_POSITIONS);

	// HASHING ONLY FOR BOUNDARY PARTICLES
	hashtable_size = nextPrime(2*B) + 1;

	Hash b_hash(hashtable_size);
	const int particles_per_row = 200;
	pitch = 0;
	hashtable = new int[hashtable_size * particles_per_row];
	for (int i = 0; i < hashtable_size; ++i) {
		for (int j = 0; j < particles_per_row; j++) {
			hashtable[i * particles_per_row + j] = -1;
		}
	}

	gpuErrchk(cudaMallocPitch(&d_hashtable, &pitch, particles_per_row * sizeof(int), hashtable_size));
	gpuErrchk(cudaMemcpy2D(d_hashtable, pitch, hashtable, particles_per_row * sizeof(int), particles_per_row * sizeof(int), hashtable_size, cudaMemcpyHostToDevice));

	grid_size = B / block_size + 1;
	hashParticlePositions << <grid_size, block_size >> > (d_hashtable, D_BOUNDARY_POSITIONS, invh, b_hash, B, pitch, particles_per_row);

	// Calculate mass (or psi) for each boundary particle

	float* d_boundary_mass;
	gpuErrchk(cudaMalloc((void**)&d_boundary_mass, B * sizeof(float)));

	boundaryPsi << <grid_size, block_size >> > (d_boundary_mass, d_hashtable, rho_0, D_BOUNDARY_POSITIONS, h, invh, particles_per_row, pitch, b_hash, B);

	float* boundary_mass = (float*)malloc(B * sizeof(float));
	gpuErrchk(cudaMemcpy(boundary_mass, d_boundary_mass, (size_t)B * sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_boundary_mass));

	//Calculate normal for boundary particles

	vec3d* d_boundary_normal;
	gpuErrchk(cudaMalloc((void**)&d_boundary_normal, B * 3 * sizeof(float)));

	boundaryNormal << <grid_size, block_size >> > (d_boundary_normal, D_BOUNDARY_POSITIONS, b_initial, b_final, B);

	vec3d* boundary_normal = (vec3d*)malloc(B * 3 * sizeof(float));
	gpuErrchk(cudaMemcpy(boundary_normal, d_boundary_normal, (size_t)B * 3 * sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_boundary_normal));

	//Write boundary vtu file

	float** boundary_point_data[] = { &boundary_mass };
	size_pointData = sizeof(boundary_point_data) / sizeof(double);
	vec3d** boundary_vectorData[] = { &boundary_normal };
	size_vectorData = sizeof(boundary_vectorData) / sizeof(double);

	std::string boundary_pointDataNames[] = { "psi" };
	std::string boundary_vectorDataNames[] = { "normal" };

	VTU_Writer(main_path, iteration, BOUNDARY_POSITIONS, B, boundary_point_data, boundary_vectorData, boundary_pointDataNames, boundary_vectorDataNames, size_pointData, size_vectorData, vtu_fullpath, 1);

	//gpuErrchk(cudaMemcpy2D(hashtable, particles_per_row * sizeof(int), d_hashtable, pitch, width, height, cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();

	//END OF HASHING FOR BOUNDARIES

	cudaFree(d_hashtable);

	cudaFree(D_BOUNDARY_POSITIONS);

	//Initializing main particle variables

	//Defining and allocating main position variable
	
	POSITION = (vec3d*)malloc(3*T*sizeof(float));
	for (int i = 0; i < N; i++) {
		POSITION[i].x = FLUID_POSITIONS[i].x;
		POSITION[i].y = FLUID_POSITIONS[i].y;
		POSITION[i].z = FLUID_POSITIONS[i].z;
	}

	for (int i = N; i < T; i++) {
		POSITION[i].x = BOUNDARY_POSITIONS[i - N].x;
		POSITION[i].y = BOUNDARY_POSITIONS[i - N].y;
		POSITION[i].z = BOUNDARY_POSITIONS[i - N].z;
	}

	free(BOUNDARY_POSITIONS);
	free(FLUID_POSITIONS);

	
	gpuErrchk(cudaMalloc((void**)&d_POSITION, 3*T*sizeof(float)));
	gpuErrchk(cudaMemcpy(d_POSITION, POSITION, 3*T*sizeof(float), cudaMemcpyHostToDevice));

	//Defining and allocating main velocity variable
	
	VELOCITY = (vec3d*)malloc(3*T*sizeof(float));
	for (int i = 0; i < T; i++) {
		VELOCITY[i].x = 0.f;
		VELOCITY[i].y = 0.f;
		VELOCITY[i].z = 0.f;
	}

	gpuErrchk(cudaMalloc((void**)&d_VELOCITY, 3*T*sizeof(float)));
	gpuErrchk(cudaMemcpy(d_VELOCITY, VELOCITY, 3*T*sizeof(float), cudaMemcpyHostToDevice));

	//Defining and allocating main st force variable
	
	ST_FORCE = (vec3d*)malloc(3*T*sizeof(float));
	for (int i = 0; i < T; i++) {
		ST_FORCE[i].x = 0.f;
		ST_FORCE[i].y = 0.f;
		ST_FORCE[i].z = 0.f;
	}

	gpuErrchk(cudaMalloc((void**)&d_ST_FORCE, 3*T*sizeof(float)));

	//Defining and allocating main viscosity force variable
	VISCOSITY_FORCE = (vec3d*)malloc(3*T*sizeof(float));
	for (int i = 0; i < T; i++) {
		VISCOSITY_FORCE[i].x = 0.f;
		VISCOSITY_FORCE[i].y = 0.f;
		VISCOSITY_FORCE[i].z = 0.f;
	}

	gpuErrchk(cudaMalloc((void**)&d_VISCOSITY_FORCE, 3*T*sizeof(float)));

	//Defining and allocating main pressure force variable
	PRESSURE_FORCE = (vec3d*)malloc(3*T*sizeof(float));
	for (int i = 0; i < T; i++) {
		PRESSURE_FORCE[i].x = 0.f;
		PRESSURE_FORCE[i].y = 0.f;
		PRESSURE_FORCE[i].z = 0.f;
	}

	gpuErrchk(cudaMalloc((void**)&d_PRESSURE_FORCE, 3*T*sizeof(float)));

	//Defining and allocating main normal variable
	NORMAL = (vec3d*)malloc(3*T*sizeof(float));
	for (int i = 0; i < N; i++) {
		NORMAL[i].x = 0.f;
		NORMAL[i].y = 0.f;
		NORMAL[i].z = 0.f;
	}	

	for (int i = N; i < T; i++) {
		NORMAL[i].x = boundary_normal[i - N].x;
		NORMAL[i].y = boundary_normal[i - N].y;
		NORMAL[i].z = boundary_normal[i - N].z;
	}

	free(boundary_normal);

	gpuErrchk(cudaMalloc((void**)&d_NORMAL, 3*T*sizeof(float)));
	gpuErrchk(cudaMemcpy(d_NORMAL, NORMAL, 3*T*sizeof(float), cudaMemcpyHostToDevice));

	//Defining and allocating main density array
	DENSITY = (float*)malloc(T * sizeof(float));
	for (int i = 0; i < T; i++) {
		DENSITY[i] = rho_0;
	}

	gpuErrchk(cudaMalloc((void**)&d_DENSITY, T * sizeof(float)));
	gpuErrchk(cudaMemcpy(d_DENSITY, DENSITY, T * sizeof(float), cudaMemcpyHostToDevice));

	//Defining and allocating main pressure array
	PRESSURE = (float*)malloc(T * sizeof(float));
	for (int i = 0; i < T; i++) {
		PRESSURE[i] = 0;
	}

	gpuErrchk(cudaMalloc((void**)&d_PRESSURE, T * sizeof(float)));

	//Defining and allocating main mass array
	
	MASS = (float*)malloc(T * sizeof(float));
	for (int i = 0; i < N; i++) {
		MASS[i] = MASS_calc;
	}

	for (int i = N; i < T; i++) {
		MASS[i] = boundary_mass[i - N];
	}

	free(boundary_mass);
	
	gpuErrchk(cudaMalloc((void**)&d_MASS, T * sizeof(float)));
	gpuErrchk(cudaMemcpy(d_MASS, MASS, T * sizeof(float), cudaMemcpyHostToDevice));

	//Defining and allocating main type array (0 if fluid, 1 if boundary)
	TYPE = (int*)malloc(T * sizeof(int));
	for (int i = 0; i < N; i++) {
		TYPE[i] = 0;
	}

	for (int i = N; i < T; i++) {
		TYPE[i] = 1;
	}

	gpuErrchk(cudaMalloc((void**)&d_TYPE, T * sizeof(int)));
	gpuErrchk(cudaMemcpy(d_TYPE, TYPE, T * sizeof(int), cudaMemcpyHostToDevice));

	//defining gravity vector
	gravity.x = 0.f;
	gravity.y = 9.81f;
	gravity.z = 0.f;

	//Defining variables to write VTU files
	pointData[0] = &DENSITY;
	pointData[1] = &PRESSURE;
	size_pointData = sizeof(pointData) / sizeof(double);

	vectorData[0] = &VELOCITY;
	vectorData[1] = &PRESSURE_FORCE;
	vectorData[2] = &VISCOSITY_FORCE;
	vectorData[3] = &ST_FORCE;
	vectorData[4] = &NORMAL;
	size_vectorData = sizeof(vectorData) / sizeof(double);

	pointDataNames[0] = "density";
	pointDataNames[1] = "pressure";
	vectorDataNames[0] = "velocity";
	vectorDataNames[1] = "pressure force";
	vectorDataNames[2] = "viscosity force";
	vectorDataNames[3] = "st force";
	vectorDataNames[4] = "normal";

	auto started = std::chrono::high_resolution_clock::now();

	VTU_Writer(vtu_path, iteration, POSITION, N, pointData, vectorData, pointDataNames, vectorDataNames, size_pointData, size_vectorData, vtu_fullpath);

	auto done = std::chrono::high_resolution_clock::now();

	std::cout << "First VTU_Writer() -> " << std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count() << " ms\n";

	VTK_Group(vtk_group_path, vtu_fullpath, simulation_time);

	// Initialize main hashtable

	hashtable_size = nextPrime(2 * T) + 1;

	hashtable = new int[hashtable_size * particles_per_row];
	for (int i = 0; i < hashtable_size; ++i) {
		for (int j = 0; j < particles_per_row; j++) {
			hashtable[i * particles_per_row + j] = -1;
		}
	}

	gpuErrchk(cudaMallocPitch(&d_hashtable, &pitch, particles_per_row * sizeof(int), hashtable_size));
	gpuErrchk(cudaMemcpy2D(d_hashtable, pitch, hashtable, particles_per_row * sizeof(int), particles_per_row * sizeof(int), hashtable_size, cudaMemcpyHostToDevice));

	std::cout << "Initializing with " << N << " fluid particles and " << B << " boundary particles.\n"
		<< "Total of " << T << " particles.\n"
		<< "Smoothing radius = " << h << " m.\n"
		<< "hashtable size = " << hashtable_size << "\n";

	return 0;
}

int mainLoop() {

	Hash hash(hashtable_size);

	grid_size = T / block_size + 1;
	hashParticlePositions << <grid_size, block_size >> > (d_hashtable, d_POSITION, invh, hash, T, pitch, particles_per_row);
	gpuErrchk(cudaDeviceSynchronize());
	printf("hashing done\n");

	grid_size = N / block_size + 1;
	fluidNormal << <grid_size, block_size >> > (d_NORMAL, d_POSITION, d_MASS, d_DENSITY, h,invh, hash,d_hashtable, particles_per_row,pitch, N);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());
	nonPressureForces << <grid_size, block_size >> > (d_POSITION, d_VISCOSITY_FORCE, d_ST_FORCE, d_MASS, d_DENSITY, d_VELOCITY, d_NORMAL, gravity, h, invh, rho_0, visc_const, st_const, particles_per_row, pitch,d_hashtable, hash, N);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaMemcpy(NORMAL, d_NORMAL, N * 3 * sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(VISCOSITY_FORCE, d_VISCOSITY_FORCE, N * 3 * sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(ST_FORCE, d_ST_FORCE, N * 3 * sizeof(float), cudaMemcpyDeviceToHost));

	iteration++;

	return 0;
}

void multiprocessor_writer() {

	char buf[1024];
	itoa(iteration, buf, 10);
	strcpy(vtu_fullpath, vtu_path);
	strcat(vtu_fullpath, "/iter");
	strcat(vtu_fullpath, buf);
	strcat(vtu_fullpath, ".vtu");

	std::future<void> write_vtu;

	auto started = std::chrono::high_resolution_clock::now();

	try {
		write_vtu.wait();
	}
	catch (std::exception& e) {
		//DO NOTHING
	}

	write_vtu = std::async(std::launch::async, VTU_Writer, vtu_path, iteration, POSITION, N, pointData, vectorData, pointDataNames, vectorDataNames, size_pointData, size_vectorData, vtu_fullpath,2);
	auto done = std::chrono::high_resolution_clock::now();

	std::cout << "Second VTU_Writer() -> " << std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count() << " ms\n";
	strcpy(buf, vtu_fullpath);

	VTK_Group(vtk_group_path, buf, simulation_time);
	//write_vtu.get();
	return;
}