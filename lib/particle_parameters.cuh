#pragma once
#include "device_functions.cuh"
#include "hashing.cuh"
#include "kernels.cuh"
#include "common.cuh"
#include "helper.cuh"
#include "forces.cuh"

// NOTES:
// 1. All functions with __global__ in front of its declaration and/or definition are called CUDA kernels and run ONLY in the GPU.
// 2. In this file, all functions marked with ** are mostly the same, only changing its core. The functions are basically searching for particle neighbors in the hashing table and performing the required calculation with the results. The function core is defined with the //CORE comment

// This kernel calculates the boundary "fake" d_params.d_MASS (psi) as defined by Equation 5 of [3]
__global__ void boundaryPsi(float* psi, float3* points) {
	
	int index = getGlobalIdx_1D_1D();

	if (index >= d_params.B) {
		return;
	}

	psi[index] = 0.f;
	int hash_list[27];
	bool skip = false;
	int count = 0;
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {
				float3 BB;
				BB.x = points[index].x + i * d_params.h;
				BB.y = points[index].y + j * d_params.h;
				BB.z = points[index].z + k * d_params.h;

				int hash_index = d_hash.hashFunction(BB, d_params.invh);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)d_params.d_hashtable + hash_index * d_params.pitch);
					for (int t = 0; t < d_params.particles_per_row; t++) {
						//CORE
						if (row[t] != -1) {
							
							float r = distance(points[index], points[row[t]]);
							if (r <= d_params.h) {
								psi[index] += Poly6_Kernel(r, d_params.h, d_params.invh);
							}
						}
					}
				}
			}
		}
	}
	
	psi[index] = d_params.rho_0 / psi[index];
	
	return;

}

// This kernel calculates the boundary Normal in a "hardcode" way. It is not a very good approach and it works only with the interior of boxes
__global__ void boundaryNormal(float3* points, float3* normal, float3 b_initial, float3 b_final) {

	int index = getGlobalIdx_1D_1D();

	if (index >= d_params.B) {
		return;
	}

	float3 point = points[index];

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
__global__ void fluidNormal() {

	int index = getGlobalIdx_1D_1D();

	if (index >= d_params.N) {
		return;
	}

	assignToVec3d(&d_params.d_NORMAL[index]);

	int hash_list[27];
	bool skip = false;
	int count = 0;
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {
				float3 BB;
				BB.x = d_params.d_POSITION[index].x + i * d_params.h;
				BB.y = d_params.d_POSITION[index].y + j * d_params.h;
				BB.z = d_params.d_POSITION[index].z + k * d_params.h;

				int hash_index = d_hash.hashFunction(BB, d_params.invh);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)d_params.d_hashtable + hash_index * d_params.pitch);
					for (int t = 0; t < d_params.particles_per_row; t++) {
	
						if (row[t] != -1) {

							float r = distance(d_params.d_POSITION[index], d_params.d_POSITION[row[t]]);
							if (r <= d_params.h && r > 0) {

								float3 poly6_gradient = Poly6_Gradient(index, row[t], d_params.d_POSITION, r, d_params.h, d_params.invh);
								float tmp;
								if (d_params.d_TYPE[row[t]] == 0) {
									tmp = d_params.h * d_params.d_MASS[row[t]] / d_params.d_DENSITY[row[t]];
								}
								else if (d_params.d_TYPE[row[t]] == 1) {
									tmp = d_params.h * d_params.d_MASS[row[t]] / d_params.rho_0;
								}

								d_params.d_NORMAL[index].x += tmp * poly6_gradient.x;
								d_params.d_NORMAL[index].y += tmp * poly6_gradient.y;
								d_params.d_NORMAL[index].z += tmp * poly6_gradient.z;
							}
						}
					}
				}
			}
		}
	}

	return;
}

// This kernel calculates the viscosity (according to [5]), surface tension and adhesion (according to [4]) forces.
// Note: The adhesion and surface tension forces are calculated in the same functions to conserve memory and lines of code
__global__ void nonPressureForces() {

	int index = getGlobalIdx_1D_1D();

	if (index >= d_params.N) {
		return;
	}

	assignToVec3d(&d_params.d_VISCOSITY_FORCE[index]);
	assignToVec3d(&d_params.d_ST_FORCE[index]);

	float3 BB;
	int hash_list[27];
	bool skip = false;
	int count = 0;
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {
				
				BB.x = d_params.d_POSITION[index].x + i * d_params.h;
				BB.y = d_params.d_POSITION[index].y + j * d_params.h;
				BB.z = d_params.d_POSITION[index].z + k * d_params.h;

				int hash_index = d_hash.hashFunction(BB, d_params.invh);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)d_params.d_hashtable + hash_index * d_params.pitch);
					for (int t = 0; t < d_params.particles_per_row; t++) {
						//CORE
						if (row[t] != -1) {
							float r = distance(d_params.d_POSITION[index], d_params.d_POSITION[row[t]]);
							if (r <= d_params.h && r > 0) {

								//Viscosity calculation

								float3 visc = ViscosityForce(index, row[t], d_params.d_MASS, d_params.d_DENSITY, d_params.d_VELOCITY, d_params.d_TYPE[row[t]], d_params.visc_const, d_params.rho_0, Viscosity_Laplacian(r, d_params.h, d_params.invh));

								//summation of calcualted value to main array
								d_params.d_VISCOSITY_FORCE[index].x += visc.x;
								d_params.d_VISCOSITY_FORCE[index].y += visc.y;
								d_params.d_VISCOSITY_FORCE[index].z += visc.z;

								//Surface tension calculation
								float3 st = STForce(index, row[t], r, d_params.d_POSITION, d_params.d_MASS, d_params.d_DENSITY, d_params.d_NORMAL, d_params.d_TYPE[row[t]], d_params.st_const, d_params.rho_0, ST_Kernel(r, d_params.h, d_params.invh, d_params.d_TYPE[row[t]]));

								//summation of calculated value to main array
								d_params.d_ST_FORCE[index].x += st.x;
								d_params.d_ST_FORCE[index].y += st.y;
								d_params.d_ST_FORCE[index].z += st.z;

								
							}
						}
					}
				}
			}
		}
	}
	
	return;
}

// A kernel to calculate d_params.d_VELOCITY and positions according to the applyed forces
__global__ void positionAndVelocity(float3* position1,float3* velocity1, float3* position2, float3* velocity2,float delta_t) {

	// array 1 -> Will be changed by this kernel
	// array 2 -> Wont be changed by this kernel

	int index = getGlobalIdx_1D_1D();

	if (index >= d_params.N) {
		return;
	}

	float tmp = delta_t / d_params.d_MASS[index];

	//calculating d_params.d_VELOCITY
	velocity1[index].x = velocity2[index].x + (d_params.d_PRESSURE_FORCE[index].x + d_params.d_VISCOSITY_FORCE[index].x + d_params.d_ST_FORCE[index].x + d_params.gravity.x * d_params.d_MASS[index]) * (tmp);
	velocity1[index].y = velocity2[index].y + (d_params.d_PRESSURE_FORCE[index].y + d_params.d_VISCOSITY_FORCE[index].y + d_params.d_ST_FORCE[index].y + d_params.gravity.y * d_params.d_MASS[index]) * (tmp);
	velocity1[index].z = velocity2[index].z + (d_params.d_PRESSURE_FORCE[index].z + d_params.d_VISCOSITY_FORCE[index].z + d_params.d_ST_FORCE[index].z + d_params.gravity.z * d_params.d_MASS[index]) * (tmp);

	//calculating position
	position1[index].x = position2[index].x + delta_t * velocity1[index].x;
	position1[index].y = position2[index].y + delta_t * velocity1[index].y;
	position1[index].z = position2[index].z + delta_t * velocity1[index].z;

	return;
}

// A collision handler according to [2]
__global__ void collisionHandler() {

	int index = getGlobalIdx_1D_1D();

	if (index >= d_params.N) {
		return;
	}

	float3 n_c_i;
	assignToVec3d(&n_c_i);
	float w_c_ib_sum = 0.f;
	float w_c_ib_second_sum = 0.f;

	float3 BB;
	int count = 0;
	bool skip = false;
	int hash_list[27];
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {

				BB.x = d_params.d_POSITION[index].x + i * d_params.h;
				BB.y = d_params.d_POSITION[index].y + j * d_params.h;
				BB.z = d_params.d_POSITION[index].z + k * d_params.h;

				int hash_index = d_hash.hashFunction(BB, d_params.invh);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)d_params.d_hashtable + hash_index * d_params.pitch);
					for (int t = 0; t < d_params.particles_per_row; t++) {
						//CORE
						if (row[t] != -1 && d_params.d_TYPE[row[t]] == 1) {
							float r = distance(d_params.d_POSITION[index], d_params.d_POSITION[row[t]]);
							float w_c_ib = fmaxf((d_params.boundary_diameter - r) / d_params.boundary_diameter,0.f);
							float3 n_b = d_params.d_NORMAL[row[t]];
							
							n_c_i.x += n_b.x * w_c_ib;
							n_c_i.y += n_b.y * w_c_ib;
							n_c_i.z += n_b.z * w_c_ib;

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

	d_params.d_POSITION[index].x += n_c_i.x * inv_norm_normal * w_c_ib_second_sum * inv_w;
	d_params.d_POSITION[index].y += n_c_i.y * inv_norm_normal * w_c_ib_second_sum * inv_w;
	d_params.d_POSITION[index].z += n_c_i.z * inv_norm_normal * w_c_ib_second_sum * inv_w;

	//calculating new velocity
	float dot = dot_product(d_params.d_VELOCITY[index], d_params.d_NORMAL[index]);
	float3 v_n;
	v_n.x = dot * n_c_i.x;
	v_n.y = dot * n_c_i.y;
	v_n.z = dot * n_c_i.z;

	d_params.d_VELOCITY[index].x = d_params.epsilon * (d_params.d_VELOCITY[index].x - v_n.x);
	d_params.d_VELOCITY[index].y = d_params.epsilon * (d_params.d_VELOCITY[index].x - v_n.y);
	d_params.d_VELOCITY[index].z = d_params.epsilon * (d_params.d_VELOCITY[index].x - v_n.z);

	return;
}

// A kernel to compute d_params.d_DENSITY according to all references
__global__ void DensityCalc() {

	int index = getGlobalIdx_1D_1D();

	if (index >= d_params.N) {
		return;
	}

	d_params.d_DENSITY[index] = 0.f;
	
	float3 BB;
	int hash_list[27];
	bool skip = false;
	int count = 0;
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {

				BB.x = d_params.d_POSITION[index].x + i * d_params.h;
				BB.y = d_params.d_POSITION[index].y + j * d_params.h;
				BB.z = d_params.d_POSITION[index].z + k * d_params.h;

				int hash_index = d_hash.hashFunction(BB, d_params.invh);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)d_params.d_hashtable + hash_index * d_params.pitch);
					for (int t = 0; t < d_params.particles_per_row; t++) {

						//CORE

						if (row[t] != -1) {
							float r = distance(d_params.d_POSITION[index], d_params.d_POSITION[row[t]]);
							if (r <= d_params.h) {
								d_params.d_DENSITY[index] += d_params.d_MASS[row[t]] * Poly6_Kernel(r, d_params.h, d_params.invh);
								
							}
						}
					}
				}
			}
		}
	}

	return;
}

// calculates d_params.d_PRESSURE according to [1] and [2]
__global__ void PressureCalc(float pressure_coeff) {
	
	int index = getGlobalIdx_1D_1D();

	if (index >= d_params.N || (d_params.d_DENSITY[index] - d_params.rho_0) <= 0) {
		return;
	}

	d_params.d_PRESSURE[index] += (d_params.d_DENSITY[index] - d_params.rho_0) * pressure_coeff;
	

	return;
}

// Calculates d_params.d_PRESSURE force according to [1] and [2]
__global__ void PressureForceCalc() {

	int index = getGlobalIdx_1D_1D();

	if (index >= d_params.N) {
		return;
	}

	//reseting float3 value to 0
	assignToVec3d(&d_params.d_PRESSURE_FORCE[index]);

	float3 BB;
	int hash_list[27];
	bool skip = false;
	int count = 0;
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {

				BB.x = d_params.d_POSITION[index].x + i * d_params.h;
				BB.y = d_params.d_POSITION[index].y + j * d_params.h;
				BB.z = d_params.d_POSITION[index].z + k * d_params.h;

				int hash_index = d_hash.hashFunction(BB, d_params.invh);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)d_params.d_hashtable + hash_index * d_params.pitch);
					for (int t = 0; t < d_params.particles_per_row; t++) {
						if (row[t] != -1) {

							float r = distance(d_params.d_POSITION[index], d_params.d_POSITION[row[t]]);
							if (r <= d_params.h && r > 0) {

								float3 spiky_grad = Spiky_Gradient(index, row[t], d_params.d_POSITION, r, d_params.h, d_params.invh);
								float3 p = PressureForce(index, row[t], d_params.d_PRESSURE, d_params.d_MASS, d_params.d_DENSITY, d_params.d_TYPE[row[t]], spiky_grad);
								sum2Vec3d(&d_params.d_PRESSURE_FORCE[index], &p);

							}
						}
					}
				}
			}
		}
	}

	return;
}

// This kernel gets maximum values of d_params.d_VELOCITY, force and d_params.d_DENSITY error and calculates the sum of all d_params.d_DENSITY errors
__global__ void getMaxVandF(float* max_force,float* max_velocity, float* sum_rho_error,float* max_rho_err) {

	int index = getGlobalIdx_1D_1D();

	if (index >= d_params.N) {
		return;
	}
	
	float max_p = maxValueInVec3D(d_params.d_PRESSURE_FORCE[index]);
	float max_v = maxValueInVec3D(d_params.d_VISCOSITY_FORCE[index]);
	float max_st = maxValueInVec3D(d_params.d_ST_FORCE[index]);

	float3 g;
	g.x = d_params.gravity.x * d_params.d_MASS[index];
	g.y = d_params.gravity.y * d_params.d_MASS[index];
	g.z = d_params.gravity.z * d_params.d_MASS[index];

	float max_g = maxValueInVec3D(g);

	atomicMaxFloat(max_force, fmaxf(max_p,fmaxf(max_v,fmaxf(max_st,max_g))));
	atomicMaxFloat(max_velocity, maxValueInVec3D(d_params.d_VELOCITY[index]));

	float rho_err = d_params.d_DENSITY[index] - d_params.rho_0;

	if (rho_err > 0) {
		atomicAddFloat(sum_rho_error, rho_err);
		atomicMaxFloat(max_rho_err, rho_err);
	}
	
	return;
}

//resets the hashtable to a clean state (full of -1)
__global__ void hashtableReset() {

	int index = getGlobalIdx_1D_1D();

	if (index >= d_params.hashtable_size) {
		return;
	}

	int* row = (int*)((char*)d_params.d_hashtable + index * d_params.pitch);
	for (int t = 0; t < d_params.particles_per_row; t++) {
		row[t] = -1;
	}

	return;

}

//resets the value of max_volocity, max_force, sum of d_params.d_DENSITY error and max d_params.d_DENSITY error
__global__ void resetValues(float* max_velocity, float* max_force, float* sum_rho_err,float* max_rho_err) {
	max_velocity[0] = 0.f;
	max_force[0] = 0.f;
	sum_rho_err[0] = 0.f;
	max_rho_err[0] = 0.f;
	return;
}

//resets pressure values
__global__ void resetPressure() {

	int index = getGlobalIdx_1D_1D();

	if (index >= d_params.N) {
		return;
	}

	d_params.d_PRESSURE[index] = 0.f;
	return;
}