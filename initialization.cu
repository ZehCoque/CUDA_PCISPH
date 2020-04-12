#include "initialize.cuh"

int initialize() {
	int block_size = 1024;
	// get main path of simulation
	char main_path[1024];
	getMainPath(main_path);

	// write path for vtu files
	char vtu_path[1024];
	strcpy(vtu_path, main_path);
	strcat(vtu_path, "/vtu");

	// write path for vtk group file
	char vtk_group_path[1024];
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

	int N = NPD[0] * NPD[1] * NPD[2]; //number of fluid particles
	int SIM_SIZE = N * SIMULATION_DIMENSION;
	const int x = 40; // Number of particles inside the smoothing length
	const float h = powf(3.f * VOLUME * x / (4.f * (float)M_PI * N), 1.f / 3.f);
	const float invh = 1 / h;

	//defining gravity vector
	gravity.x = 0.f;
	gravity.y = -9.81f;
	gravity.z = 0.f;

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

	int grid_size = N / block_size + 1;

	//generate locations for each particle
	makePrism << <grid_size, block_size >> > (D_FLUID_POSITIONS, PARTICLE_DIAMETER, f_initial, D_NPD, N);

	// Get number per dimension (NPD) of BOUNDARY particles without compact packing (assuming use of makebox function)
	for (int i = 0; i < 3; i++) {
		NPD[i] = static_cast<int>(ceil((B_FINAL_POSITION[i] - B_INITIAL_POSITION[i]) / PARTICLE_DIAMETER)) + 2;
	}

	//copy new NPD to device memory
	gpuErrchk(cudaMemcpy(D_NPD, NPD, SIMULATION_DIMENSION * sizeof(float), cudaMemcpyHostToDevice));

	int B = NPD[0] * NPD[1] * NPD[2] - (NPD[0] - 2) * (NPD[1] - 2) * (NPD[2] - 2); //Number of boundary particles
	SIM_SIZE = NPD[0] * NPD[1] * NPD[2] * SIMULATION_DIMENSION;

	vec3d b_initial;
	b_initial.x = B_INITIAL_POSITION[0] - PARTICLE_RADIUS;
	b_initial.y = B_INITIAL_POSITION[1] - PARTICLE_RADIUS;
	b_initial.z = B_INITIAL_POSITION[2] - PARTICLE_RADIUS;
	vec3d b_final;
	b_final.x = b_initial.x + PARTICLE_DIAMETER * (NPD[0] - 1);
	b_final.y = b_initial.y + PARTICLE_DIAMETER * (NPD[1] - 1);
	b_final.z = b_initial.z + PARTICLE_DIAMETER * (NPD[2] - 1);

	size_t bytes_boundary_particles = SIM_SIZE * sizeof(float);
	vec3d* BOUNDARY_POSITIONS; //host pointer
	BOUNDARY_POSITIONS = (vec3d*)malloc(bytes_boundary_particles); //allocate memory in the host

	vec3d* D_BOUNDARY_POSITIONS; //device pointer
	gpuErrchk(cudaMalloc((void**)&D_BOUNDARY_POSITIONS, bytes_boundary_particles)); // allocate memory in the device

	makeBox(D_BOUNDARY_POSITIONS, PARTICLE_DIAMETER, b_initial, b_final, block_size, D_NPD);

	int T = N + B; //Total number of particles

	gpuErrchk(cudaMemcpy(FLUID_POSITIONS, D_FLUID_POSITIONS, bytes_fluid_particles, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(BOUNDARY_POSITIONS, D_BOUNDARY_POSITIONS, bytes_boundary_particles, cudaMemcpyDeviceToHost));

	// Free GPU memory for fluid particles
	cudaFree(D_FLUID_POSITIONS);

	// HASHING ONLY FOR BOUNDARY PARTICLES
	const int hashtable_size = nextPrime(2 * B) + 1;

	Hash hash(hashtable_size);
	const int particles_per_row = 200;
	size_t pitch = 0;
	int* hashtable = new int[hashtable_size * particles_per_row];
	for (int i = 0; i < hashtable_size; ++i) {
		for (int j = 0; j < particles_per_row; j++) {
			hashtable[i * particles_per_row + j] = -1;
		}
	}

	int* d_hashtable;

	size_t width = particles_per_row * sizeof(int);
	size_t height = hashtable_size;

	gpuErrchk(cudaMallocPitch(&d_hashtable, &pitch, particles_per_row * sizeof(int), hashtable_size));
	gpuErrchk(cudaMemcpy2D(d_hashtable, pitch, hashtable, particles_per_row * sizeof(int), width, height, cudaMemcpyHostToDevice));

	grid_size = B / block_size + 1;
	hashParticlePositions << <grid_size, block_size >> > (d_hashtable, D_BOUNDARY_POSITIONS, invh, hash, B, pitch, particles_per_row);

	float* d_boundary_mass;
	gpuErrchk(cudaMalloc((void**)&d_boundary_mass, B * sizeof(float)));

	boundaryPsi << <grid_size, block_size >> > (d_boundary_mass, d_hashtable, rho_0, D_BOUNDARY_POSITIONS, h, invh, particles_per_row, pitch, hash, B);

	float* boundary_mass = (float*)malloc(B * sizeof(float));
	gpuErrchk(cudaMemcpy(boundary_mass, d_boundary_mass, (size_t)B * sizeof(float), cudaMemcpyDeviceToHost));

	float** boundary_point_data[] = { &boundary_mass };
	int size_pointData = sizeof(boundary_point_data) / sizeof(double);
	vec3d** boundary_vectorData[1] = { };
	int size_vectorData = 0;

	std::string boundary_pointDataNames[] = { "psi" };
	std::string boundary_vectorDataNames[1] = {  };

	char vtu_fullpath[1024];
	int iteration = 1;
	float simulation_time = 0;
	VTU_Writer(main_path, iteration, BOUNDARY_POSITIONS, B, boundary_point_data, boundary_vectorData, boundary_pointDataNames, boundary_vectorDataNames, size_pointData, size_vectorData, vtu_fullpath, 1);

	//gpuErrchk(cudaMemcpy2D(hashtable, particles_per_row * sizeof(int), d_hashtable, pitch, width, height, cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();

	//END OF HASHING FOR BOUNDARIES

	cudaFree(d_hashtable);

	cudaFree(D_BOUNDARY_POSITIONS);

	//Initializing main particle variables

	//Defining and allocating main position variable
	vec3d* POSITION;
	POSITION = (vec3d*)malloc(bytes_fluid_particles + bytes_boundary_particles);
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
	//free(FLUID_POSITIONS);

	vec3d* d_POSITION;
	gpuErrchk(cudaMalloc((void**)&d_POSITION, bytes_fluid_particles + bytes_boundary_particles));
	gpuErrchk(cudaMemcpy(d_POSITION, POSITION, bytes_fluid_particles + bytes_boundary_particles, cudaMemcpyHostToDevice));

	//Defining and allocating main velocity variable
	vec3d* VELOCITY;
	VELOCITY = (vec3d*)malloc(bytes_fluid_particles + bytes_boundary_particles);
	for (int i = 0; i < T; i++) {
		VELOCITY[i].x = 0.f;
		VELOCITY[i].y = 0.f;
		VELOCITY[i].z = 0.f;
	}

	vec3d* d_VELOCITY;
	gpuErrchk(cudaMalloc((void**)&d_VELOCITY, bytes_fluid_particles + bytes_boundary_particles));
	gpuErrchk(cudaMemcpy(d_VELOCITY, VELOCITY, bytes_fluid_particles + bytes_boundary_particles, cudaMemcpyHostToDevice));

	//Defining and allocating main st force variable
	vec3d* ST_FORCE;
	ST_FORCE = (vec3d*)malloc(bytes_fluid_particles + bytes_boundary_particles);
	for (int i = 0; i < T; i++) {
		ST_FORCE[i].x = 0.f;
		ST_FORCE[i].y = 0.f;
		ST_FORCE[i].z = 0.f;
	}

	vec3d* d_ST_FORCE;
	gpuErrchk(cudaMalloc((void**)&d_ST_FORCE, bytes_fluid_particles + bytes_boundary_particles));
	gpuErrchk(cudaMemcpy(d_ST_FORCE, ST_FORCE, bytes_fluid_particles + bytes_boundary_particles, cudaMemcpyHostToDevice));

	//Defining and allocating main viscosity force variable
	vec3d* VISCOSITY_FORCE;
	VISCOSITY_FORCE = (vec3d*)malloc(bytes_fluid_particles + bytes_boundary_particles);
	for (int i = 0; i < T; i++) {
		VISCOSITY_FORCE[i].x = 0.f;
		VISCOSITY_FORCE[i].y = 0.f;
		VISCOSITY_FORCE[i].z = 0.f;
	}

	vec3d* d_VISCOSITY_FORCE;
	gpuErrchk(cudaMalloc((void**)&d_VISCOSITY_FORCE, bytes_fluid_particles + bytes_boundary_particles));
	gpuErrchk(cudaMemcpy(d_VISCOSITY_FORCE, VISCOSITY_FORCE, bytes_fluid_particles + bytes_boundary_particles, cudaMemcpyHostToDevice));

	//Defining and allocating main pressure force variable
	vec3d* PRESSURE_FORCE;
	PRESSURE_FORCE = (vec3d*)malloc(bytes_fluid_particles + bytes_boundary_particles);
	for (int i = 0; i < T; i++) {
		PRESSURE_FORCE[i].x = 0.f;
		PRESSURE_FORCE[i].y = 0.f;
		PRESSURE_FORCE[i].z = 0.f;
	}

	vec3d* d_PRESSURE_FORCE;
	gpuErrchk(cudaMalloc((void**)&d_PRESSURE_FORCE, bytes_fluid_particles + bytes_boundary_particles));
	gpuErrchk(cudaMemcpy(d_PRESSURE_FORCE, PRESSURE_FORCE, bytes_fluid_particles + bytes_boundary_particles, cudaMemcpyHostToDevice));

	//Defining and allocating main density array
	float* DENSITY;
	DENSITY = (float*)malloc(T * sizeof(float));
	for (int i = 0; i < T; i++) {
		DENSITY[i] = rho_0;
	}

	float* d_DENSITY;
	gpuErrchk(cudaMalloc((void**)&d_DENSITY, T * sizeof(float)));
	gpuErrchk(cudaMemcpy(d_DENSITY, DENSITY, T * sizeof(float), cudaMemcpyHostToDevice));

	//Defining and allocating main pressure array
	float* PRESSURE;
	PRESSURE = (float*)malloc(T * sizeof(float));
	for (int i = 0; i < T; i++) {
		PRESSURE[i] = 0;
	}

	float* d_PRESSURE;
	gpuErrchk(cudaMalloc((void**)&d_PRESSURE, T * sizeof(float)));
	gpuErrchk(cudaMemcpy(d_PRESSURE, PRESSURE, T * sizeof(float), cudaMemcpyHostToDevice));

	//Defining and allocating main mass array
	float* MASS;
	MASS = (float*)malloc(T * sizeof(float));
	for (int i = 0; i < N; i++) {
		MASS[i] = MASS_calc;
	}

	for (int i = N; i < T; i++) {
		MASS[i] = boundary_mass[i - N];
	}

	float* d_MASS;
	gpuErrchk(cudaMalloc((void**)&d_MASS, T * sizeof(float)));
	gpuErrchk(cudaMemcpy(d_MASS, MASS, T * sizeof(float), cudaMemcpyHostToDevice));

	//Defining and allocating main type array (0 if fluid, 1 if boundary)
	int* TYPE;
	TYPE = (int*)malloc(T * sizeof(int));
	for (int i = 0; i < N; i++) {
		TYPE[i] = 0;
	}

	for (int i = N; i < T; i++) {
		TYPE[i - N] = 1;
	}

	int* d_TYPE;
	gpuErrchk(cudaMalloc((void**)&d_TYPE, T * sizeof(int)));
	gpuErrchk(cudaMemcpy(d_TYPE, TYPE, T * sizeof(int), cudaMemcpyHostToDevice));

	//Defining variables to write VTU files
	float** pointData[] = { &DENSITY, &PRESSURE };
	size_pointData = sizeof(pointData) / sizeof(double);
	vec3d** vectorData[] = { &VELOCITY,&PRESSURE_FORCE,&VISCOSITY_FORCE,&ST_FORCE };
	size_vectorData = sizeof(vectorData) / sizeof(double);

	std::string pointDataNames[] = { "density","pressure" };
	std::string vectorDataNames[] = { "velocity","pressure force","viscosity force","st force" };

	strcpy(vtu_fullpath, VTU_Writer(vtu_path, iteration, FLUID_POSITIONS, N, pointData, vectorData, pointDataNames, vectorDataNames, size_pointData, size_vectorData, vtu_fullpath));

	VTK_Group(vtk_group_path, vtu_fullpath, simulation_time);

	std::cout << "Initializing with " << N << " fluid particles and " << B << " boundary particles.\n"
		<< "Total of " << T << " particles.\n"
		<< "Smoothing radius = " << h << " m.\n";

	return 0;
}