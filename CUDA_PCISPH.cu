//#define _USE_MATH_DEFINES
//
//#include "particle_positions.cuh"
//#include "utilities.cuh"
//#include "VTK.cuh"
////#include "hashing.cu"
//
////float inf = std::numeric_limits<float>::infinity();
//
//// Initial conditions
//const float PARTICLE_RADIUS = 0.01f;
//const float MASS = (float)M_PI *- pow(PARTICLE_RADIUS, 3.f) / 3.f * 4.f;
//const float PARTICLE_DIAMETER = 2 * PARTICLE_RADIUS;
//const float F_INITIAL_POSITION[3] = { -0.5,-0.5,-0.5 }; //Fluid particles initial position
//const float F_FINAL_POSITION[3] = { 0.5,0.5,0.5 }; //Fluid particles final position
//const float B_INITIAL_POSITION[3] = { -0.5,-0.5,-0.5 }; //Boundary particles final position
//const float B_FINAL_POSITION[3] = { 0.5,0.5,0.5 }; //Boundary particles final position
//float VOLUME = 1;
//const int SIMULATION_DIMENSION = 3;
//const int x = 40; // Number of particles inside the smoothing length
//vec3d gravity;
//
//int iteration = 1;
//float simulation_time = 0;
//
//// Value for PI -> M_PI
//
//int main(void)
//{
//	int block_size = 1024;
//	// get main path of simulation
//	char main_path[1024];
//	getMainPath(main_path);
//
//	// write path for vtu files
//	char vtu_path[1024];
//	strcpy(vtu_path, main_path);
//	strcat(vtu_path, "/vtu");
//
//	// write path for vtk group file
//	char vtk_group_path[1024];
//	strcpy(vtk_group_path, main_path);
//	strcat(vtk_group_path, "/PCISPH.pvd");
//
//	// create directory for vtu files
//	CreateDir(vtu_path);
//
//	// Get number per dimension (NPD) of FLUID particles for hexadecimal packing (assuming use of makeprism function)
//	int NPD[3];
//	for (int i = 0; i < 3; i++) {
//		if (i == 1) {
//			NPD[i] = floor((F_FINAL_POSITION[i] - F_INITIAL_POSITION[i]) / (sqrt(3.f) / 2.f * PARTICLE_DIAMETER));
//			VOLUME = VOLUME * (F_FINAL_POSITION[i] - F_INITIAL_POSITION[i]);
//		}
//		else {
//			NPD[i] = floor((F_FINAL_POSITION[i] - F_INITIAL_POSITION[i]) / PARTICLE_DIAMETER);
//			VOLUME = VOLUME * (F_FINAL_POSITION[i] - F_INITIAL_POSITION[i]);
//		}
//	}
//
//	//Passing NPD to device
//	int* D_NPD;
//	cudaMalloc((void**)&D_NPD,SIMULATION_DIMENSION*sizeof(float));
//	cudaMemcpy(D_NPD, NPD, SIMULATION_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);
//
//	int N = NPD[0] * NPD[1] * NPD[2]; //number of fluid particles
//	int SIM_SIZE = N * SIMULATION_DIMENSION;
//	const float h = pow(3.f * VOLUME * x / (4.f * M_PI * N), 1.f / 3.f);
//
//	//defining gravity vector
//	gravity.x = 0;
//	gravity.y = -9.81;
//	gravity.z = 0;
//
//	const float boundary_radius = h/4;
//	const float boundary_diameter = h/2;
//
//	cudaError_t cudaStatus;
//	cudaStatus = cudaSetDevice(0);
//
//	vec3d f_initial;
//	f_initial.x = F_INITIAL_POSITION[0] + PARTICLE_RADIUS;
//	f_initial.y = F_INITIAL_POSITION[1] + PARTICLE_RADIUS;
//	f_initial.z = F_INITIAL_POSITION[2] + PARTICLE_RADIUS;
//
//	size_t bytes_fluid_particles = SIM_SIZE * sizeof(float);
//
//	vec3d* FLUID_POSITIONS; //host pointer
//	FLUID_POSITIONS = (vec3d*)malloc(bytes_fluid_particles);
//
//	vec3d* D_FLUID_POSITIONS; //device pointer
//	cudaMalloc((void**)&D_FLUID_POSITIONS, bytes_fluid_particles);
//
//	// grid -> number of blocks
//	// block -> number of threads
//
//	int grid_size = N/ block_size + 1;
//
//	//generate locations for each particle
//	makePrism << <grid_size, block_size >> > (D_FLUID_POSITIONS, PARTICLE_DIAMETER, f_initial, D_NPD, N);
//
//	// Get number per dimension (NPD) of BOUNDARY particles without compact packing (assuming use of makebox function)
//	for (int i = 0; i < 3; i++) {
//		NPD[i] = ceil((B_FINAL_POSITION[i] - B_INITIAL_POSITION[i]) / PARTICLE_DIAMETER) + 2;
//	}
//
//	//copy new NPD to device memory
//	cudaMemcpy(D_NPD, NPD, SIMULATION_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);
//
//	int B = NPD[0] * NPD[1] * NPD[2] - (NPD[0] - 2) * (NPD[1] - 2) * (NPD[2] - 2); //Number of boundary particles
//	SIM_SIZE = NPD[0] * NPD[1] * NPD[2] * SIMULATION_DIMENSION;
//
//	vec3d b_initial;
//	b_initial.x = B_INITIAL_POSITION[0] - PARTICLE_RADIUS;
//	b_initial.y = B_INITIAL_POSITION[1] - PARTICLE_RADIUS;
//	b_initial.z = B_INITIAL_POSITION[2] - PARTICLE_RADIUS;
//	vec3d b_final;
//	b_final.x = b_initial.x + PARTICLE_DIAMETER * (NPD[0] - 1);
//	b_final.y = b_initial.y + PARTICLE_DIAMETER * (NPD[1] - 1);
//	b_final.z = b_initial.z + PARTICLE_DIAMETER * (NPD[2] - 1);
//
//	size_t bytes_boundary_particles = SIM_SIZE * sizeof(float);
//	vec3d* BOUNDARY_POSITIONS; //host pointer
//	BOUNDARY_POSITIONS = (vec3d*)malloc(bytes_boundary_particles); //allocate memory in the host
//
//	vec3d* D_BOUNDARY_POSITIONS; //device pointer
//	cudaMalloc((void**)&D_BOUNDARY_POSITIONS, bytes_boundary_particles); // allocate memory in the device
//
//	makeBox(D_BOUNDARY_POSITIONS, PARTICLE_DIAMETER, b_initial, b_final, block_size,D_NPD);
//
//	int T = N + B; //Total number of particles
//
//	std::cout << "Initializing with " << N << " fluid particles and " << B << " boundary particles.\n"
//		<< "Total of " << T << " particles.\n"
//		<< "Smoothing radius = " << h << " m.\n";
//
//	float* density = new float[N];
//	for (int i = 0; i < N; i++) {
//		density[i] = 1000;
//	}
//
//	vec3d* velocity = new vec3d[N];
//	for (int i = 0; i < N; i++) {
//		velocity[i].x = i;
//		velocity[i].y = i;
//		velocity[i].z = i;
//	}
//
//	float** pointData[] = { &density };
//	int size_pointData = sizeof(pointData) / sizeof(double);
//	vec3d** vectorData[] = { &velocity };
//	int size_vectorData = sizeof(vectorData) / sizeof(double);
//
//	std::string pointDataNames[] = { "density" };
//	std::string vectorDataNames[] = { "velocity" };
//
//	char vtu_fullpath[1024];
//	// cudaDeviceSynchronize();
//
//	cudaMemcpy(FLUID_POSITIONS, D_FLUID_POSITIONS, bytes_fluid_particles, cudaMemcpyDeviceToHost);
//	cudaMemcpy(BOUNDARY_POSITIONS, D_BOUNDARY_POSITIONS, bytes_boundary_particles, cudaMemcpyDeviceToHost);
//
//	// Free memory
//	cudaFree(D_FLUID_POSITIONS);
//	cudaFree(D_BOUNDARY_POSITIONS);
//
//	strcpy(vtu_fullpath, VTU_Writer(vtu_path, iteration, FLUID_POSITIONS, N, pointData, vectorData, pointDataNames, vectorDataNames, size_pointData, size_vectorData, vtu_fullpath));
//
//	float* psi = new float[B];
//	for (int i = 0; i < B; i++) {
//		psi[i] = i;
//	}
//
//	vec3d* velocity2 = new vec3d[B];
//	for (int i = 0; i < B; i++) {
//		velocity2[i].x = i;
//		velocity2[i].y = i;
//		velocity2[i].z = i;
//	}
//
//	float** pointData2[] = { &psi };
//	int size_pointData2 = sizeof(pointData2) / sizeof(double);
//	vec3d** vectorData2[] = { &velocity2 };
//	int size_vectorData2 = sizeof(vectorData2) / sizeof(double);
//
//	std::string pointDataNames2[] = { "psi" };
//	std::string vectorDataNames2[] = { "velocity2" };
//
//	VTU_Writer(main_path, iteration, BOUNDARY_POSITIONS, B, pointData2, vectorData2, pointDataNames2, vectorDataNames2, size_pointData2, size_vectorData2, vtu_fullpath, 1);
//
//	VTK_Group(vtk_group_path, vtu_fullpath, simulation_time);
//
//	return 0;
//}