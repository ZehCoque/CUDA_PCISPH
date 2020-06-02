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

// This kernel calculates the boundary "fake" mass (psi) as defined by Equation 5 of [3]
__global__ void boundaryPsi(float* psi, float3* position, int* hashtable) {
	
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
				BB.x = position[index].x + i * d_params.h;
				BB.y = position[index].y + j * d_params.h;
				BB.z = position[index].z + k * d_params.h;

				int hash_index = hashFunction(BB);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)hashtable + hash_index * d_params.pitch);
					for (int t = 0; t < d_params.particles_per_row; t++) {
						//CORE
						if (row[t] != -1) {
							
							float r = distance(position[index], position[row[t]]);
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
__global__ void boundaryNormal(float3* position, float3* normal, float3 b_initial, float3 b_final) {

	int index = getGlobalIdx_1D_1D();

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
__global__ void fluidNormal(float3 *position, float3 *normal,float* mass, float* density, int* type, int *hashtable) {

	int index = getGlobalIdx_1D_1D();

	if (index >= d_params.N) {
		return;
	}

	assignToVec3d(&normal[index]);

	int hash_list[27];
	bool skip = false;
	int count = 0;
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {
				float3 BB;
				BB.x = position[index].x + i * d_params.h;
				BB.y = position[index].y + j * d_params.h;
				BB.z = position[index].z + k * d_params.h;

				int hash_index = hashFunction(BB);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)hashtable + hash_index * d_params.pitch);
					for (int t = 0; t < d_params.particles_per_row; t++) {
	
						if (row[t] != -1) {

							float r = distance(position[index], position[row[t]]);
							if (r <= d_params.h && r > 0) {

								float3 poly6_gradient = Poly6_Gradient(index, row[t], position, r, d_params.h, d_params.invh);
								float tmp;
								if (type[row[t]] == 0) {
									tmp = d_params.h * mass[row[t]] / density[row[t]];
								}
								else if (type[row[t]] == 1) {
									tmp = d_params.h * mass[row[t]] / d_params.rho_0;
								}

								normal[index].x += tmp * poly6_gradient.x;
								normal[index].y += tmp * poly6_gradient.y;
								normal[index].z += tmp * poly6_gradient.z;
							}
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
__global__ void nonPressureForces(float3* position,float3* velocity, float3* viscosity_force, float3* st_force,float3* normal, float* mass, float* density, int* type, int* hashtable) {

	int index = getGlobalIdx_1D_1D();

	if (index >= d_params.N) {
		return;
	}

	assignToVec3d(&viscosity_force[index]);
	assignToVec3d(&st_force[index]);

	float3 BB;
	int hash_list[27];
	bool skip = false;
	int count = 0;
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {
				
				BB.x = position[index].x + i * d_params.h;
				BB.y = position[index].y + j * d_params.h;
				BB.z = position[index].z + k * d_params.h;

				int hash_index = hashFunction(BB);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)hashtable + hash_index * d_params.pitch);
					for (int t = 0; t < d_params.particles_per_row; t++) {
						//CORE
						if (row[t] != -1) {
							float r = distance(position[index], position[row[t]]);
							if (r <= d_params.h && r > 0) {

								//Viscosity calculation

								float3 visc = ViscosityForce(index, row[t], mass, density, velocity, type[row[t]], d_params.visc_const, d_params.rho_0, Viscosity_Laplacian(r, d_params.h, d_params.invh));

								//summation of calcualted value to main array
								viscosity_force[index].x += visc.x;
								viscosity_force[index].y += visc.y;
								viscosity_force[index].z += visc.z;

								//Surface tension calculation
								float3 st = STForce(index, row[t], r, position, mass, density, normal, type[row[t]], d_params.st_const, d_params.rho_0, ST_Kernel(r, d_params.h, d_params.invh, type[row[t]]));

								//summation of calculated value to main array
								st_force[index].x += st.x;
								st_force[index].y += st.y;
								st_force[index].z += st.z;

								
							}
						}
					}
				}
			}
		}
	}
	
	return;
}

// A kernel to calculate velocity and positions according to the applyed forces
__global__ void positionAndVelocity(float3* position1,float3* velocity1, float3* position2, float3* velocity2,float3* pressure_force,float3* viscosity_force, float3* st_force, float* mass, float delta_t) {

	// 1 -> Will be changed by this kernel
	// 2 -> Wont be changed by this kernel

	int index = getGlobalIdx_1D_1D();

	if (index >= d_params.N) {
		return;
	}

	float tmp = delta_t / mass[index];

	//calculating velocity
	velocity1[index].x = velocity2[index].x + (pressure_force[index].x + viscosity_force[index].x + st_force[index].x + d_params.gravity.x * mass[index]) * (tmp);
	velocity1[index].y = velocity2[index].y + (pressure_force[index].y + viscosity_force[index].y + st_force[index].y + d_params.gravity.y * mass[index]) * (tmp);
	velocity1[index].z = velocity2[index].z + (pressure_force[index].z + viscosity_force[index].z + st_force[index].z + d_params.gravity.z * mass[index]) * (tmp);

	//calculating position
	position1[index].x = position2[index].x + delta_t * velocity1[index].x;
	position1[index].y = position2[index].y + delta_t * velocity1[index].y;
	position1[index].z = position2[index].z + delta_t * velocity1[index].z;

	return;
}

// A collision handler according to [2]
__global__ void collisionHandler(float3* position,float3* velocity, float3* normal, int* type, int* hashtable) {

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

				BB.x = position[index].x + i * d_params.h;
				BB.y = position[index].y + j * d_params.h;
				BB.z = position[index].z + k * d_params.h;

				int hash_index = hashFunction(BB);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)hashtable + hash_index * d_params.pitch);
					for (int t = 0; t < d_params.particles_per_row; t++) {
						//CORE
						if (row[t] != -1 && type[row[t]] == 1) {
							float r = distance(position[index], position[row[t]]);
							float w_c_ib = fmaxf((d_params.boundary_diameter - r) / d_params.boundary_diameter,0.f);
							float3 n_b = normal[row[t]];
							
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
	float tmp = inv_norm_normal * w_c_ib_second_sum * inv_w;

	position[index].x += n_c_i.x * tmp;
	position[index].y += n_c_i.y * tmp;
	position[index].z += n_c_i.z * tmp;

	//calculating new velocity
	float dot = dot_product(velocity[index], normal[index]);
	float3 v_n;
	v_n.x = dot * n_c_i.x;
	v_n.y = dot * n_c_i.y;
	v_n.z = dot * n_c_i.z;

	velocity[index].x = d_params.epsilon * (velocity[index].x - v_n.x);
	velocity[index].y = d_params.epsilon * (velocity[index].y - v_n.y);
	velocity[index].z = d_params.epsilon * (velocity[index].z - v_n.z);

	return;
}

// A kernel to compute density according to all references
__global__ void DensityCalc(float3* position, float* density, float* mass, int* hashtable) {

	int index = getGlobalIdx_1D_1D();

	if (index >= d_params.N) {
		return;
	}

	density[index] = 0.f;
	
	float3 BB;
	int hash_list[27];
	bool skip = false;
	int count = 0;
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {

				BB.x = position[index].x + i * d_params.h;
				BB.y = position[index].y + j * d_params.h;
				BB.z = position[index].z + k * d_params.h;

				int hash_index = hashFunction(BB);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)hashtable + hash_index * d_params.pitch);
					for (int t = 0; t < d_params.particles_per_row; t++) {

						//CORE

						if (row[t] != -1) {
							float r = distance(position[index], position[row[t]]);
							if (r <= d_params.h) {
								density[index] += mass[row[t]] * Poly6_Kernel(r, d_params.h, d_params.invh);
								
							}
						}
					}
				}
			}
		}
	}

	return;
}

// calculates pressure according to [1] and [2]
__global__ void PressureCalc(float* pressure, float* density, float* pressure_coeff) {
	
	int index = getGlobalIdx_1D_1D();

	if (index >= d_params.N || (density[index] - d_params.rho_0) <= 0) {
		return;
	}

	pressure[index] += (density[index] - d_params.rho_0) * *pressure_coeff;
	

	return;
}

// Calculates pressure force according to [1] and [2]
__global__ void PressureForceCalc(float3* position, float3* pressure_force, float* density, float* pressure, float* mass, int* type, int* hashtable) {

	int index = getGlobalIdx_1D_1D();

	if (index >= d_params.N) {
		return;
	}

	//reseting float3 value to 0
	assignToVec3d(&pressure_force[index]);

	float3 BB;
	int hash_list[27];
	bool skip = false;
	int count = 0;
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {

				BB.x = position[index].x + i * d_params.h;
				BB.y = position[index].y + j * d_params.h;
				BB.z = position[index].z + k * d_params.h;

				int hash_index = hashFunction(BB);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)hashtable + hash_index * d_params.pitch);
					for (int t = 0; t < d_params.particles_per_row; t++) {
						if (row[t] != -1) {

							float r = distance(position[index], position[row[t]]);
							if (r <= d_params.h && r > 0) {

								float3 spiky_grad = Spiky_Gradient(index, row[t], position, r, d_params.h, d_params.invh);
								float3 p = PressureForce(index, row[t], pressure, mass, density, type[row[t]], spiky_grad);
								sum2Vec3d(&pressure_force[index], &p);

							}
						}
					}
				}
			}
		}
	}

	return;
}

// This kernel gets maximum values of velocity, force and density error and calculates the sum of all density errors
__global__ void getMaxVandF(float3* velocity,float3* pressure_force,float3* viscosity_force,float3* st_force,float* density, float* mass,float* max_force,float* max_velocity, float* sum_rho_error,float* max_rho_err) {

	int index = getGlobalIdx_1D_1D();

	if (index >= d_params.N) {
		return;
	}
	
	float max_p = maxValueInVec3D(pressure_force[index]);
	float max_v = maxValueInVec3D(viscosity_force[index]);
	float max_st = maxValueInVec3D(st_force[index]);

	float3 g;
	g.x = d_params.gravity.x * mass[index];
	g.y = d_params.gravity.y * mass[index];
	g.z = d_params.gravity.z * mass[index];

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

//resets the hashtable to a clean state (full of -1)
__global__ void hashtableReset(int* hashtable) {

	int index = getGlobalIdx_1D_1D();

	if (index >= d_params.hashtable_size) {
		return;
	}

	int* row = (int*)((char*)hashtable + index * d_params.pitch);
	for (int t = 0; t < d_params.particles_per_row; t++) {
		row[t] = -1;
	}

	return;

}

//resets the value of max_volocity, max_force, sum of density error and max density error
__global__ void resetValues(float* max_velocity, float* max_force, float* sum_rho_err,float* max_rho_err) {
	*max_velocity = 0.f;
	*max_force = 0.f;
	*sum_rho_err = 0.f;
	*max_rho_err = 0.f;
	return;
}

//resets pressure values
__global__ void resetPressure(float* pressure) {

	int index = getGlobalIdx_1D_1D();

	if (index >= d_params.N) {
		return;
	}

	pressure[index] = 0.f;
	return;
}

// calculate the pressure coefficient as in Equation 8 of [1]
__global__ void pressureCoeff(float *pressure_coeff, float delta_t) {
	*pressure_coeff = -1 / (2 * powf(d_params.mass * delta_t / d_params.rho_0, 2) * d_params.pressure_delta);
}