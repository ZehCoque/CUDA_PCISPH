#pragma once
#include "device_functions.cuh"
#include "hashing.cuh"
#include "kernels.cuh"
#include "common.cuh"
#include "helper.cuh"
#include "forces.cuh"
#include "helper_math.h"

extern __shared__ float3 shared_array[]; //pointer to dynamically allocated shared memory

// NOTES:
// 1. All functions with __global__ in front of its declaration and/or definition are called CUDA kernels and run ONLY in the GPU.
// 2. In this file, all functions marked with ** are mostly the same, only changing its core. The functions are basically searching for particle neighbors in the hashing table and performing the required calculation with the results. The function core is defined with the //CORE comment

// This kernel calculates the boundary "fake" mass (psi) as defined by Equation 5 of [3]
__global__ void boundaryPsi(float* psi, float3* position,uint *cellStart, uint *cellEnd) {
	
	uint index = getGlobalIdx_1D_1D();

	if (index >= d_params.B) {
		return;
	}

	float current_psi = 0.f;
	float3 *sharedPos = shared_array;
	int sharedMemIndex = threadIdx.x - blockIdx.x * blockDim.x;

	sharedPos[sharedMemIndex] = position[index];

	__syncthreads();

	float3 current_position = sharedPos[sharedMemIndex];

	int3 gridPos = calcGridPos(current_position);

	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {

				int3 neighbourGridPos = make_int3(gridPos.x + i, gridPos.y + j, gridPos.z + k);
				uint gridHash = calcGridHash(neighbourGridPos);
				uint startIndex = cellStart[gridHash];

				if (startIndex != 0xffffffff) {
					uint endIndex = cellEnd[gridHash];

					for (uint cell_id = startIndex; cell_id < endIndex; cell_id++) {
						
						
						sharedMemIndex = cell_id - blockIdx.x * blockDim.x;
						float3 neighbor_position = sharedPos[sharedMemIndex];

						float r = distance(current_position, neighbor_position);
						if (r <= d_params.h) {
							current_psi += Poly6_Kernel(&r);
						}
					}
				}
			}
		}
	}
	
	psi[index] = d_params.rho_0 / current_psi;
	
	return;

}

// This kernel calculates the boundary Normal in a "hardcode" way. It is not a very good approach and it works only with the interior of boxes
__global__ void boundaryNormal(float3* position, float3* normal, float3 b_initial, float3 b_final) {

	uint index = getGlobalIdx_1D_1D();

	if (index >= d_params.B) {
		return;
	}

	float3 point = position[index];

	normal[index].x = 0.f;
	normal[index].y = 0.f;
	normal[index].z = 0.f;

	if (point.x == b_initial.x) {
		normal[index].x = 1.f;

		if (point.y == b_initial.y) {
			normal[index].y = 1.f;

			if (point.z == b_initial.z) {
				normal[index].z = 1.f;
			}
			else if (point.z == b_final.z) {
				normal[index].z = -1.f;
			}
		}
		else if (point.y == b_final.y) {
			normal[index].y = -1.f;

			if (point.z == b_initial.z) {
				normal[index].z = 1.f;
			}
			else if (point.z == b_final.z) {
				normal[index].z = -1.f;
			}
		}
		else if (point.z == b_initial.z) {
			normal[index].z = 1.f;

			if (point.y == b_initial.y) {
				normal[index].y = 1.f;
			}
			else if (point.y == b_final.y) {
				normal[index].y = -1.f;
			}

		} 
		else if (point.z == b_final.y) {
			normal[index].z = -1.f;

			if (point.y == b_initial.y) {
				normal[index].y = 1.f;
			}
			else if (point.y == b_final.y) {
				normal[index].y = -1.f;
			}
		}
		return;
	}

	if (point.y == b_initial.y) {
		normal[index].y = 1.f;

		if (point.x == b_initial.x) {
			normal[index].x = 1.f;

			if (point.z == b_initial.z) {
				normal[index].z = 1.f;
			}
			else if (point.z == b_final.z) {
				normal[index].z = -1.f;
			}

		}
		else if (point.z == b_initial.z) {
			normal[index].z = 1.f;

			if (point.x == b_initial.x) {
				normal[index].x = 1.f;
			}
			else if (point.x == b_final.x) {
				normal[index].x = -1.f;
			}
		}
		else if (point.x == b_final.x) {
			normal[index].x = -1.f;

			if (point.z == b_initial.z) {
				normal[index].z = 1.f;
			}
			else if (point.z == b_final.z) {
				normal[index].z = -1.f;
			}
		}
		else if (point.z == b_final.z) {
			normal[index].z = -1.f;

			if (point.x == b_initial.x) {
				normal[index].x = 1.f;
			}
			else if (point.x == b_final.x) {
				normal[index].x = -1.f;
			}
		}
		return;
	}

	if (point.z == b_initial.z) {
		normal[index].z = 1.f;

		if (point.x == b_initial.x) {
			normal[index].x = 1.f;

			if (point.y == b_initial.y) {
				normal[index].y = 1.f;
			}
			else if (point.y == b_final.y) {
				normal[index].y = -1.f;
			}
		}
		else if (point.y == b_initial.y) {
			normal[index].y = 1.f;

			if (point.x == b_initial.x) {
				normal[index].x = 1.f;
			}
			else if (point.x == b_final.x) {
				normal[index].x = -1.f;
			}
		}
		else if (point.x == b_final.x) {
			normal[index].x = -1.f;

			if (point.y == b_initial.y) {
				normal[index].y = 1.f;
			}
			else if (point.y == b_final.y) {
				normal[index].y = -1.f;
			}
		}
		else if (point.y == b_final.y) {
			normal[index].y = -1.f;

			if (point.x == b_initial.x) {
				normal[index].x = 1.f;
			}
			else if (point.x == b_final.x) {
				normal[index].x = -1.f;
			}
		}
		return;
	}

	if (point.x == b_final.x) {
		normal[index].x = -1.f;
		if (point.y == b_initial.y) {
			normal[index].y = 1.f;

			if (point.z == b_initial.z) {
				normal[index].z = 1.f;
			}
			else if (point.z == b_final.z) {
				normal[index].z = -1.f;
			}
		}
		else if (point.z == b_initial.z) {
			normal[index].z = 1.f;

			if (point.y == b_initial.y) {
				normal[index].y = 1.f;
			}
			else if (point.y == b_final.y) {
				normal[index].y = -1.f;
			}
		}
		else if (point.y == b_final.y) {
			normal[index].y = -1.f;

			if (point.z == b_initial.z) {
				normal[index].z = 1.f;
			}
			else if (point.z == b_final.z) {
				normal[index].z = -1.f;
			}
		}
		else if (point.z == b_final.z) {
			normal[index].z = -1.f;

			if (point.y == b_initial.y) {
				normal[index].y = 1.f;
			}
			else if (point.y == b_final.y) {
				normal[index].y = -1.f;
			}
		}
		return;
	}

	if (point.y == b_final.y) {
		normal[index].y = -1.f;

		if (point.x == b_initial.x) {
			normal[index].x = 1.f;

			if (point.z == b_initial.z) {
				normal[index].z = 1.f;
			}
			else if (point.z == b_final.z) {
				normal[index].z = -1.f;
			}
		}
		else if (point.z == b_initial.z) {
			normal[index].z = 1.f;

			if (point.x == b_initial.x) {
				normal[index].x = 1.f;
			}
			else if (point.x == b_final.x) {
				normal[index].x = -1.f;
			}
		}
		else if (point.x == b_final.x) {
			normal[index].x = -1.f;

			if (point.z == b_initial.z) {
				normal[index].z = 1.f;
			}
			else if (point.z == b_final.z) {
				normal[index].z = -1.f;
			}
		}
		else if (point.z == b_final.z) {
			normal[index].z = -1.f;

			if (point.x == b_initial.x) {
				normal[index].x = 1.f;
			}
			else if (point.x == b_final.x) {
				normal[index].x = -1.f;
			}
		}
		return;
	}

	if (point.z == b_final.z) {
		normal[index].z = -1.f;
		if (point.x == b_initial.x) {
			normal[index].x = 1.f;

			if (point.y == b_initial.y) {
				normal[index].y = 1.f;
			}
			else if (point.y == b_final.y) {
				normal[index].y = -1.f;
			}
		}
		else if (point.y == b_initial.y) {
			normal[index].y = 1.f;

			if (point.x == b_initial.x) {
				normal[index].x = 1.f;
			}
			else if (point.x == b_final.x ){
				normal[index].x = -1.f;
			}
		}
		else if (point.x == b_final.x) {
			normal[index].x = -1.f;

			if (point.y == b_initial.y) {
				normal[index].y = 1.f;
			}
			else if (point.y == b_final.y) {
				normal[index].y = -1.f;
			}
		}
		else if (point.y == b_final.y) {
			normal[index].y = -1.f;

			if (point.x == b_initial.x) {
				normal[index].x = 1.f;
			}
			else if (point.x == b_final.x) {
				normal[index].x = -1.f;
			}
		}
		return;
	}

	return;

}

// This kernel calculates the fluid normal according to Equation between equations 2 and 3 of [4] (it does not have a number)
__global__ void fluidNormal(float3 *position, float3 *normal,float* mass, float* density, int* type, uint* cellStart, uint* cellEnd) {

	uint index = getGlobalIdx_1D_1D();

	if (index >= d_params.N) {
		return;
	}

	float3 current_normal = make_float3(0.f,0.f,0.f);

	float3* sharedPos = shared_array;
	float* sharedDensity = (float*)&sharedPos[d_params.block_size];
	float* sharedMass = &sharedDensity[d_params.block_size];
	int* sharedType = (int*)&sharedMass[d_params.block_size];

	int sharedMemIndex = threadIdx.x - blockIdx.x * blockDim.x;

	sharedPos[sharedMemIndex] = position[index];
	sharedDensity[sharedMemIndex] = density[index];
	sharedMass[sharedMemIndex] = mass[index];
	sharedType[sharedMemIndex] = type[index];

	__syncthreads();

	float3 current_position = sharedPos[sharedMemIndex];
	float current_density = sharedDensity[sharedMemIndex];
	float current_mass = sharedMass[sharedMemIndex];
	
	int3 gridPos = calcGridPos(current_position);

	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {

				int3 neighbourGridPos = make_int3(gridPos.x + i, gridPos.y + j, gridPos.z + k);
				uint gridHash = calcGridHash(neighbourGridPos);
				uint startIndex = cellStart[gridHash];

				if (startIndex != 0xffffffff) {
					uint endIndex = cellEnd[gridHash];

					for (uint cell_id = startIndex; cell_id < endIndex; cell_id++) {

						sharedMemIndex = cell_id - blockIdx.x * blockDim.x;

						float3 neighbor_position = sharedPos[sharedMemIndex];

						float r = distance(current_position, neighbor_position);
						if (r <= d_params.h && r > 0) {

							float3 poly6_gradient = Poly6_Gradient(&current_position, &neighbor_position, &r);

							float neigbor_density = sharedDensity[sharedMemIndex];
							float neighbor_mass = sharedMass[sharedMemIndex];
							int neighbor_type = sharedType[sharedMemIndex];

							float tmp;
							if (neighbor_type == 0) {
								tmp = d_params.h * neighbor_mass / neigbor_density;
							}
							else if (neighbor_type == 1) {
								tmp = d_params.h * neighbor_mass / d_params.rho_0;
							}

							current_normal = make_float3(tmp * poly6_gradient.x, tmp * poly6_gradient.y, tmp * poly6_gradient.z);
						}
							
					}
				}
			}
		}
	}

	return;

}

// This kernel calculates the viscosity_force (according to [5]), surface tension and adhesion (according to [4]) forces.
// Note: The adhesion and surface tension forces are calculated in the same functions to conserve memory and lines of code
__global__ void nonPressureForces(float3* position,float3* velocity, float3* viscosity_force, float3* st_force,float3* normal, float* mass, float* density, int* type, uint * cellStart, uint * cellEnd) {

	uint index = getGlobalIdx_1D_1D();

	if (index >= d_params.N) {
		return;
	}

	float3 current_viscosity_force = make_float3(0.f, 0.f, 0.f);
	float3 current_st_force = make_float3(0.f, 0.f, 0.f);

	float3* sharedPos = shared_array;
	float* sharedDensity = (float*)&sharedPos[d_params.block_size];
	float* sharedMass = &sharedDensity[d_params.block_size];
	int* sharedType = (int*)&sharedMass[d_params.block_size];
	float3* sharedVel = (float3*)&sharedType[d_params.block_size];
	float3* sharedNormal = &sharedVel[d_params.block_size];

	int sharedMemIndex = threadIdx.x - blockIdx.x * blockDim.x;

	sharedPos[sharedMemIndex] = position[index];
	sharedVel[sharedMemIndex] = velocity[index];
	sharedDensity[sharedMemIndex] = density[index];
	sharedMass[sharedMemIndex] = mass[index];
	sharedNormal[sharedMemIndex] = normal[index];
	sharedType[sharedMemIndex] = type[index];

	__syncthreads();

	float3 current_position = sharedPos[sharedMemIndex];
	float3 current_velocity = sharedVel[sharedMemIndex];
	float3 current_normal = sharedNormal[sharedMemIndex];
	float current_density = sharedDensity[sharedMemIndex];
	float current_mass = sharedMass[sharedMemIndex];

	int3 gridPos = calcGridPos(current_position);

	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {

				int3 neighbourGridPos = make_int3(gridPos.x + i, gridPos.y + j, gridPos.z + k);
				uint gridHash = calcGridHash(neighbourGridPos);
				uint startIndex = cellStart[gridHash];

				if (startIndex != 0xffffffff) {
					uint endIndex = cellEnd[gridHash];

					for (uint cell_id = startIndex; cell_id < endIndex; cell_id++) {

						sharedMemIndex = cell_id - blockIdx.x * blockDim.x;

						float3 neighbor_position = sharedPos[sharedMemIndex];

						float r = distance(current_position, neighbor_position);
						if (r <= d_params.h && r > 0) {

							float neighbor_mass = sharedMass[sharedMemIndex];
							float neighbor_density = sharedDensity[sharedMemIndex];
							float3 neighbor_velocity = sharedVel[sharedMemIndex];
							float3 neighbor_normal = sharedNormal[sharedMemIndex];
							int neighbor_type = sharedType[sharedMemIndex];

							float3 visc = ViscosityForce(&current_mass, &neighbor_mass, &current_density, &neighbor_density, &current_velocity, &neighbor_velocity, &neighbor_type, Viscosity_Laplacian(&r));

							//summation of calcualted value to main array
							current_viscosity_force += visc;

							//Surface tension calculation
							float3 st = STForce(&current_position,&neighbor_position, &current_mass, &neighbor_mass, &current_density, &neighbor_density,&current_normal,&neighbor_normal, &neighbor_type,&r, ST_Kernel(&r,&neighbor_type));

							//summation of calculated value to main array
							current_st_force += st;
						}
					}
				}
			}
		}
	}

	viscosity_force[index] = current_viscosity_force;
	st_force[index] = current_st_force;

}

// A kernel to calculate velocity and positions according to the applyed forces
__global__ void positionAndVelocity(float3* position1,float3* velocity1, float3* position2, float3* velocity2,float3* pressure_force,float3* viscosity_force, float3* st_force, float* mass, float *delta_t) {

	// 1 -> Will be changed by this kernel
	// 2 -> Wont be changed by this kernel

	uint index = getGlobalIdx_1D_1D();

	if (index >= d_params.N) {
		return;
	}

	float tmp = *delta_t / mass[index];

	float3 tmp_velocity;

	//calculating velocity


	tmp_velocity = velocity2[index] + (pressure_force[index] + viscosity_force[index] + st_force[index] + d_params.gravity * mass[index]) * tmp;
	velocity1[index] = tmp_velocity;

	//calculating position

	position1[index] = position2[index] + *delta_t * tmp_velocity;

	return;
}

// A collision handler according to [2]
__global__ void collisionHandler(float3* position,float3* velocity, float3* normal, int* type, uint* cellStart, uint* cellEnd) {

	uint index = getGlobalIdx_1D_1D();

	if (index >= d_params.N) {
		return;
	}

	float3 current_viscosity_force = make_float3(0.f, 0.f, 0.f);
	float3 current_st_force = make_float3(0.f, 0.f, 0.f);

	float3* sharedPos = shared_array;
	int* sharedType = (int*)&sharedType[d_params.block_size];
	float3* sharedVel = (float3*)&sharedType[d_params.block_size];
	float3* sharedNormal = &sharedVel[d_params.block_size];

	int sharedMemIndex = threadIdx.x - blockIdx.x * blockDim.x;

	sharedPos[sharedMemIndex] = position[index];
	sharedVel[sharedMemIndex] = velocity[index];
	sharedNormal[sharedMemIndex] = normal[index];
	sharedType[sharedMemIndex] = type[index];

	__syncthreads();

	float3 current_position = sharedPos[sharedMemIndex];
	float3 current_velocity = sharedVel[sharedMemIndex];
	float3 current_normal = sharedNormal[sharedMemIndex];

	int3 gridPos = calcGridPos(current_position);

	float3 n_c_i;
	float w_c_ib_sum;
	float w_c_ib_second_sum;

	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {

				int3 neighbourGridPos = make_int3(gridPos.x + i, gridPos.y + j, gridPos.z + k);
				uint gridHash = calcGridHash(neighbourGridPos);
				uint startIndex = cellStart[gridHash];

				if (startIndex != 0xffffffff) {
					uint endIndex = cellEnd[gridHash];

					for (uint cell_id = startIndex; cell_id < endIndex; cell_id++) {

						sharedMemIndex = cell_id - blockIdx.x * blockDim.x;

						float3 neighbor_position = sharedPos[sharedMemIndex];

						float r = distance(current_position, neighbor_position);
						if (r <= d_params.h && r > 0) {

							float3 neighbor_normal = sharedNormal[sharedMemIndex];

							float r = distance(current_position, neighbor_position);
							float w_c_ib = fmaxf((d_params.boundary_diameter - r) / d_params.boundary_diameter, 0.f);
							float3 n_b = neighbor_normal;

							n_c_i += n_b * w_c_ib;

							w_c_ib_sum += w_c_ib;
							w_c_ib_second_sum += w_c_ib * (d_params.boundary_diameter - r);
						}
					}
				}
			}
		}
	}

	if (w_c_ib_sum == 0) {
		return;
	}


	//calculating new position
	float inv_norm_normal = 1 / norm3df(n_c_i.x, n_c_i.y, n_c_i.z);
	float inv_w = 1 / w_c_ib_sum;
	float tmp = inv_norm_normal * w_c_ib_second_sum * inv_w;

	position[index] += n_c_i * tmp;

	//calculating new velocity
	float dot = dot_product(current_velocity, current_normal);
	float3 v_n;
	v_n = dot * n_c_i;

	velocity[index] = d_params.epsilon * (current_velocity - v_n);

	return;
}

// A kernel to compute density according to all references
__global__ void DensityCalc(float3* position, float* density, float* mass, uint* cellStart, uint* cellEnd) {

	uint index = getGlobalIdx_1D_1D();

	if (index >= d_params.N) {
		return;
	}

	float current_density = 0.f;
	float3* sharedPos = shared_array;
	float* sharedMass = (float*)&sharedPos[d_params.block_size];

	int sharedMemIndex = threadIdx.x - blockIdx.x * blockDim.x;

	sharedPos[sharedMemIndex] = position[index];
	sharedMass[sharedMemIndex] = mass[index];

	__syncthreads();

	float3 current_position = sharedPos[sharedMemIndex];

	int3 gridPos = calcGridPos(current_position);

	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {

				int3 neighbourGridPos = make_int3(gridPos.x + i, gridPos.y + j, gridPos.z + k);
				uint gridHash = calcGridHash(neighbourGridPos);
				uint startIndex = cellStart[gridHash];

				if (startIndex != 0xffffffff) {
					uint endIndex = cellEnd[gridHash];

					for (uint cell_id = startIndex; cell_id < endIndex; cell_id++) {


						sharedMemIndex = cell_id - blockIdx.x * blockDim.x;
						float3 neighbor_position = sharedPos[sharedMemIndex];

						float r = distance(current_position, neighbor_position);
						if (r <= d_params.h) {

							float neighbor_mass = sharedMass[sharedMemIndex];

							current_density += neighbor_mass * Poly6_Kernel(&r);
							
						}
					}
				}
			}
		}
	}
	
	density[index] = current_density;

	return;
}

// calculates pressure according to [1] and [2]
__global__ void PressureCalc(float* pressure, float* density, float *delta_t) {
	
	__shared__ float pressure_coeff; 

	if (threadIdx.x == 0) {
		// calculates the pressure coefficient as in Equation 8 of [1] only once per thread block and stores it in shared memory
		pressure_coeff = -1 / (2 * powf(d_params.mass * *delta_t / d_params.rho_0, 2) * d_params.pressure_delta);
	}

	uint index = getGlobalIdx_1D_1D();

	if (index >= d_params.N) {
		return;
	}

	float current_density_diff = density[index] - d_params.rho_0;

	if (current_density_diff <= 0) {
		return;
	}

	__syncthreads();

	pressure[index] += current_density_diff * pressure_coeff;
	

	return;
}

// Calculates pressure force according to [1] and [2]
__global__ void PressureForceCalc(float3* position, float3* pressure_force, float* density, float* pressure, float* mass, int* type, uint* cellStart, uint* cellEnd) {

	uint index = getGlobalIdx_1D_1D();

	if (index >= d_params.N) {
		return;
	}

	float3 current_pressure_force = make_float3(0.f,0.f,0.f);

	float3* sharedPos = shared_array;
	int* sharedType = (int*)&sharedType[d_params.block_size];
	float* sharedDensity = (float*)&sharedType[d_params.block_size];
	float* sharedPressure = &sharedDensity[d_params.block_size];
	float* sharedMass = &sharedPressure[d_params.block_size];

	int sharedMemIndex = threadIdx.x - blockIdx.x * blockDim.x;

	sharedPos[sharedMemIndex] = position[index];
	sharedType[sharedMemIndex] = type[index];
	sharedDensity[sharedMemIndex] = density[index];
	sharedPressure[sharedMemIndex] = pressure[index];
	sharedMass[sharedMemIndex] = mass[index];

	__syncthreads();

	float3 current_position = sharedPos[sharedMemIndex];
	float current_density = sharedDensity[sharedMemIndex];
	float current_pressure = sharedPressure[sharedMemIndex];
	float current_mass = sharedMass[sharedMemIndex];

	int3 gridPos = calcGridPos(current_position);

	float3 n_c_i;
	float w_c_ib_sum;
	float w_c_ib_second_sum;

	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {

				int3 neighbourGridPos = make_int3(gridPos.x + i, gridPos.y + j, gridPos.z + k);
				uint gridHash = calcGridHash(neighbourGridPos);
				uint startIndex = cellStart[gridHash];

				if (startIndex != 0xffffffff) {
					uint endIndex = cellEnd[gridHash];

					for (uint cell_id = startIndex; cell_id < endIndex; cell_id++) {

						sharedMemIndex = cell_id - blockIdx.x * blockDim.x;

						float3 neighbor_position = sharedPos[sharedMemIndex];

						float r = distance(current_position, neighbor_position);
						if (r <= d_params.h && r > 0) {

							float neighbor_pressure = sharedPressure[sharedMemIndex];
							float neighbor_mass = sharedMass[sharedMemIndex];
							float neighbor_density = sharedDensity[sharedMemIndex];
							int neighbor_type = sharedType[sharedMemIndex];

							current_pressure_force = PressureForce(&current_pressure,&neighbor_pressure,&current_mass,&neighbor_mass,&current_density,&neighbor_density,&neighbor_type, Spiky_Gradient(&current_position, &neighbor_position, &r));
						}
					}
				}
			}
		}
	}

	pressure_force[index] = current_pressure_force;

	return;
}

// This kernel gets maximum values of velocity, force and density error and calculates the sum of all density errors
__global__ void getMaxVandF(float3* velocity,float3* pressure_force,float3* viscosity_force,float3* st_force,float* density, float* mass,float* max_force,float* max_velocity, float* sum_rho_error, float* max_rho_err) {

	uint index = getGlobalIdx_1D_1D();

	if (index >= d_params.N) {
		return;
	}

	float tmp_mass = mass[index];
	
	float max_p = maxValueInVec3D(pressure_force[index]);
	float max_v = maxValueInVec3D(viscosity_force[index]);
	float max_st = maxValueInVec3D(st_force[index]);

	float3 g = d_params.gravity * tmp_mass;

	float max_g = maxValueInVec3D(g);

	atomicMaxFloat(max_force, fmaxf(max_p,fmaxf(max_v,fmaxf(max_st,max_g))));
	atomicMaxFloat(max_velocity, maxValueInVec3D(velocity[index]));

	float rho_err = density[index] - d_params.rho_0;

	if (rho_err > 0) {
		atomicAddFloat(sum_rho_error, rho_err);
		atomicMaxFloat(max_rho_err, rho_err);
	}
	
	return;
}

__global__ void deltaTCriteria(float* max_force,float* max_velocity, float* max_rho_err, float* sum_rho_err, float* delta_t, float* max_rho_err_t_1, float3* d_POSITION_0, float3* d_POSITION_1, float3* d_POSITION_2, float3* d_VELOCITY_0, float3* d_VELOCITY_1, float3* d_VELOCITY_2) {

	bool criteria1;
	bool criteria2;
	bool criteria3;
	bool criteria4;

	if (getGlobalIdx_1D_1D() == 0) {
		// criterias for delta_t increase
		criteria1 = 0.19f * sqrt(d_params.h / *max_force) > *delta_t;
		criteria2 = *max_rho_err < 4.5f * d_params.max_vol_comp;
		criteria3 = *sum_rho_err/d_params.N < 0.9f * d_params.max_vol_comp;
		criteria4 = 0.39f * (d_params.h / *max_velocity) > *delta_t;

		if (criteria1 && criteria2 && criteria3 && criteria4) {
			*delta_t += *delta_t * 0.2f / 100.f;

			cudaMemcpyAsync(d_POSITION_2, d_POSITION_1, d_params.N * sizeof(float3), cudaMemcpyDeviceToDevice);
			cudaMemcpyAsync(d_POSITION_1, d_POSITION_0, d_params.N * sizeof(float3), cudaMemcpyDeviceToDevice);
			cudaMemcpyAsync(d_VELOCITY_2, d_VELOCITY_1, d_params.N * sizeof(float3), cudaMemcpyDeviceToDevice);
			cudaMemcpyAsync(d_VELOCITY_1, d_VELOCITY_0, d_params.N * sizeof(float3), cudaMemcpyDeviceToDevice);

		}
	}
	else if (getGlobalIdx_1D_1D() == 1) {
		// criterias for delta_t decrease

		criteria1 = 0.2f * sqrt(d_params.h / *max_force) < *delta_t;
		criteria2 = *max_rho_err > 5.5f * d_params.max_vol_comp;
		criteria3 = *sum_rho_err / d_params.N > d_params.max_vol_comp;
		criteria4 = 0.4f * (d_params.h / *max_velocity) <= *delta_t;

		if (criteria1 || criteria2 || criteria3 || criteria4) {
			*delta_t -= *delta_t * 0.2f / 100.f;

			cudaMemcpyAsync(d_POSITION_2, d_POSITION_1, d_params.N * sizeof(float3), cudaMemcpyDeviceToDevice);
			cudaMemcpyAsync(d_POSITION_1, d_POSITION_0, d_params.N * sizeof(float3), cudaMemcpyDeviceToDevice);
			cudaMemcpyAsync(d_VELOCITY_2, d_VELOCITY_1, d_params.N * sizeof(float3), cudaMemcpyDeviceToDevice);
			cudaMemcpyAsync(d_VELOCITY_1, d_VELOCITY_0, d_params.N * sizeof(float3), cudaMemcpyDeviceToDevice);
		}
	}
	else if (getGlobalIdx_1D_1D() == 2) {
		// criterias for shock detection

		criteria1 = *max_rho_err - *max_rho_err_t_1 > 8 * d_params.max_vol_comp;
		criteria2 = *max_rho_err > d_params.max_rho_fluc;
		criteria3 = 0.45f * (d_params.h / *max_velocity) < *delta_t;

		if (criteria1 || criteria2 || criteria3) {
			*delta_t *= 0.5f;

			cudaMemcpyAsync(d_POSITION_0, d_POSITION_2, d_params.N * sizeof(float3), cudaMemcpyDeviceToDevice);
			cudaMemcpyAsync(d_VELOCITY_0, d_VELOCITY_2, d_params.N * sizeof(float3), cudaMemcpyDeviceToDevice);
		}

	}

	return;
}