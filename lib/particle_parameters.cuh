#pragma once
#include "device_functions.cuh"
#include "hashing.cuh"
#include "kernels.cuh"
#include "common.cuh"
#include "helper.cuh"
#include "forces.cuh"

// simulation parameters
struct SimParams
{
	float rho_0;
	float h;
	float invh;
	float Ncols;
	size_t pitch;
	Hash hash;
	uint N;
	uint B;
	uint T;
	uint hashtable_size;

	float boundary_diameter;

	float visc_const;
	float st_const;
	float epsilon;
};

__constant__ SimParams params;
// NOTES:
// 1. All functions with __global__ in front of its declaration and/or definition are called CUDA kernels and run ONLY in the GPU.
// 2. In this file, all functions marked with ** are mostly the same, only changing its core. The functions are basically searching for particle neighbors in the hashing table and performing the required calculation with the results. The function core is defined with the //CORE comment

// This kernel calculates the boundary "fake" mass (psi) as defined by Equation 5 of [3]
__global__ void boundaryPsi(float* psi, int* d_hashtable, float3* points) {
	
	int index = getGlobalIdx_1D_1D();

	if (index >= params.B) {
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
				BB.x = points[index].x + i * params.h;
				BB.y = points[index].y + j * params.h;
				BB.z = points[index].z + k * params.h;

				int hash_index = params.hash.hashFunction(BB, params.invh);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)d_hashtable + hash_index * params.pitch);
					for (int t = 0; t < params.Ncols; t++) {
						//CORE
						if (row[t] != -1) {
							
							float r = distance(points[index], points[row[t]]);
							if (r <= params.h) {
								psi[index] += Poly6_Kernel(r, params.h, params.invh);
							}
						}
					}
				}
			}
		}
	}
	
	psi[index] = params.rho_0 / psi[index];
	
	return;

}

// This kernel calculates the boundary Normal in a "hardcode" way. It is not a very good approach and it works only with the interior of boxes
__global__ void boundaryNormal(float3* normal,float3* points,float3 b_initial, float3 b_final) {

	int index = getGlobalIdx_1D_1D();

	if (index >= params.B) {
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
__global__ void fluidNormal(float3* normal, float3* points, float* mass, float* density, int* type, int* d_hashtable) {

	int index = getGlobalIdx_1D_1D();

	if (index >= params.N) {
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
				BB.x = points[index].x + i * params.h;
				BB.y = points[index].y + j * params.h;
				BB.z = points[index].z + k * params.h;

				int hash_index = params.hash.hashFunction(BB, params.invh);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)d_hashtable + hash_index * params.pitch);
					for (int t = 0; t < params.Ncols; t++) {
	
						if (row[t] != -1) {

							float r = distance(points[index], points[row[t]]);
							if (r <= params.h && r > 0) {

								float3 poly6_gradient = Poly6_Gradient(index, row[t], points, r, params.h, params.invh);
								float tmp;
								if (type[row[t]] == 0) {
									tmp = params.h * mass[row[t]] / density[row[t]];
								}
								else if (type[row[t]] == 1) {
									tmp = params.h * mass[row[t]] / params.rho_0;
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

// This kernel calculates the viscosity (according to [5]), surface tension and adhesion (according to [4]) forces.
// Note: The adhesion and surface tension forces are calculated in the same functions to conserve memory and lines of code
__global__ void nonPressureForces(float3* points,float3* viscosity_force, float3* st_force,float* mass,float* density, float3* velocity,float3* normal, float3 gravity, int* type,int* d_hashtable) {

	int index = getGlobalIdx_1D_1D();

	if (index >= params.N) {
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
				
				BB.x = points[index].x + i * params.h;
				BB.y = points[index].y + j * params.h;
				BB.z = points[index].z + k * params.h;

				int hash_index = params.hash.hashFunction(BB, params.invh);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)d_hashtable + hash_index * params.pitch);
					for (int t = 0; t < params.Ncols; t++) {
						//CORE
						if (row[t] != -1) {
							float r = distance(points[index], points[row[t]]);
							if (r <= params.h && r > 0) {

								//Viscosity calculation

								float3 visc = ViscosityForce(index, row[t], mass, density, velocity, type[row[t]], params.visc_const, params.rho_0, Viscosity_Laplacian(r, params.h, params.invh));

								//summation of calcualted value to main array
								viscosity_force[index].x += visc.x;
								viscosity_force[index].y += visc.y;
								viscosity_force[index].z += visc.z;

								//Surface tension calculation
								float3 st = STForce(index, row[t], r, points, mass, density, normal, type[row[t]], params.st_const, params.rho_0, ST_Kernel(r, params.h, params.invh, type[row[t]]));

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

// A kernel to calculate velocities and positions according to the applyed forces
__global__ void positionAndVelocity(float3* points1,float3* velocities1, float3* points2, float3* velocities2, float3* pressure_force, float3* viscosity_force, float3* st_force,float3 gravity,float* mass,float delta_t) {

	// array 1 -> Will be changed by this kernel
	// array 2 -> Wont be changed by this kernel

	int index = getGlobalIdx_1D_1D();

	if (index >= params.N) {
		return;
	}

	float tmp = delta_t / mass[index];

	//calculating velocity
	velocities1[index].x = velocities2[index].x + (pressure_force[index].x + viscosity_force[index].x + st_force[index].x + gravity.x * mass[index]) * (tmp);
	velocities1[index].y = velocities2[index].y + (pressure_force[index].y + viscosity_force[index].y + st_force[index].y + gravity.y * mass[index]) * (tmp);
	velocities1[index].z = velocities2[index].z + (pressure_force[index].z + viscosity_force[index].z + st_force[index].z + gravity.z * mass[index]) * (tmp);

	//calculating position
	points1[index].x = points2[index].x + delta_t * velocities1[index].x;
	points1[index].y = points2[index].y + delta_t * velocities1[index].y;
	points1[index].z = points2[index].z + delta_t * velocities1[index].z;

	return;
}

// A collision handler according to [2]
__global__ void collisionHandler(float3* points, float3* velocities,float3* normal,int* type,int* d_hashtable) {

	int index = getGlobalIdx_1D_1D();

	if (index >= params.N) {
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

				BB.x = points[index].x + i * params.h;
				BB.y = points[index].y + j * params.h;
				BB.z = points[index].z + k * params.h;

				int hash_index = params.hash.hashFunction(BB, params.invh);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)d_hashtable + hash_index * params.pitch);
					for (int t = 0; t < params.Ncols; t++) {
						//CORE
						if (row[t] != -1 && type[row[t]] == 1) {
							float r = distance(points[index], points[row[t]]);
							float w_c_ib = fmaxf((params.boundary_diameter - r) / params.boundary_diameter,0.f);
							float3 n_b = normal[row[t]];
							
							n_c_i.x += n_b.x * w_c_ib;
							n_c_i.y += n_b.y * w_c_ib;
							n_c_i.z += n_b.z * w_c_ib;

							w_c_ib_sum += w_c_ib;
							w_c_ib_second_sum += w_c_ib * (params.boundary_diameter - r);
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

	points[index].x += n_c_i.x * inv_norm_normal * w_c_ib_second_sum * inv_w;
	points[index].y += n_c_i.y * inv_norm_normal * w_c_ib_second_sum * inv_w;
	points[index].z += n_c_i.z * inv_norm_normal * w_c_ib_second_sum * inv_w;

	//calculating new velocity
	float dot = dot_product(velocities[index], normal[index]);
	float3 v_n;
	v_n.x = dot * n_c_i.x;
	v_n.y = dot * n_c_i.y;
	v_n.z = dot * n_c_i.z;

	velocities[index].x = params.epsilon * (velocities[index].x - v_n.x);
	velocities[index].y = params.epsilon * (velocities[index].x - v_n.y);
	velocities[index].z = params.epsilon * (velocities[index].x - v_n.z);

	return;
}

// A kernel to compute density according to all references
__global__ void DensityCalc(float3* points, float* mass, float* density,int* d_hashtable) {

	int index = getGlobalIdx_1D_1D();

	if (index >= params.N) {
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

				BB.x = points[index].x + i * params.h;
				BB.y = points[index].y + j * params.h;
				BB.z = points[index].z + k * params.h;

				int hash_index = params.hash.hashFunction(BB, params.invh);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)d_hashtable + hash_index * params.pitch);
					for (int t = 0; t < params.Ncols; t++) {

						//CORE

						if (row[t] != -1) {
							float r = distance(points[index], points[row[t]]);
							if (r <= params.h) {
								density[index] += mass[row[t]] * Poly6_Kernel(r, params.h, params.invh);
								
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
__global__ void PressureCalc(float* pressure, float* density,float rho_0,float pressure_coeff,int size) {
	
	int index = getGlobalIdx_1D_1D();

	if (index >= size || (density[index] - rho_0) <= 0) {
		return;
	}

	pressure[index] += (density[index] - rho_0) * pressure_coeff;
	

	return;
}

// Calculates pressure force according to [1] and [2]
__global__ void PressureForceCalc(float3* points, float3* pressure_force, float* pressure, float* mass, float* density, int* type, int* d_hashtable) {

	int index = getGlobalIdx_1D_1D();

	if (index >= params.N) {
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

				BB.x = points[index].x + i * params.h;
				BB.y = points[index].y + j * params.h;
				BB.z = points[index].z + k * params.h;

				int hash_index = params.hash.hashFunction(BB, params.invh);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)d_hashtable + hash_index * params.pitch);
					for (int t = 0; t < params.Ncols; t++) {
						if (row[t] != -1) {

							float r = distance(points[index], points[row[t]]);
							if (r <= params.h && r > 0) {

								float3 spiky_grad = Spiky_Gradient(index, row[t], points, r, params.h, params.invh);
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
__global__ void getMaxVandF(float* max_velocity, float* max_force, float3* velocities, float3* pressure_force, float3* viscosity_force, float3* st_force, float3 gravity,float* mass,float* density,float* sum_rho_error,float* max_rho_err,float rho_0,int size) {

	int index = getGlobalIdx_1D_1D();

	if (index >= size) {
		return;
	}
	
	float max_p = maxValueInVec3D(pressure_force[index]);
	float max_v = maxValueInVec3D(viscosity_force[index]);
	float max_st = maxValueInVec3D(st_force[index]);

	float3 g;
	g.x = gravity.x * mass[index];
	g.y = gravity.y * mass[index];
	g.z = gravity.z * mass[index];

	float max_g = maxValueInVec3D(g);

	atomicMaxFloat(max_force, fmaxf(max_p,fmaxf(max_v,fmaxf(max_st,max_g))));
	atomicMaxFloat(max_velocity, maxValueInVec3D(velocities[index]));

	float rho_err = density[index] - rho_0;

	if (rho_err > 0) {
		atomicAddFloat(sum_rho_error, rho_err);
		atomicMaxFloat(max_rho_err, rho_err);
	}
	
	return;
}

//resets the hashtable to a clean state (full of -1)
__global__ void hashtableReset(int* d_hashtable) {

	int index = getGlobalIdx_1D_1D();

	if (index >= params.hashtable_size) {
		return;
	}

	int* row = (int*)((char*)d_hashtable + index * params.pitch);
	for (int t = 0; t < params.Ncols; t++) {
		row[t] = -1;
	}

	return;

}

//resets the value of max_volocity, max_force, sum of density error and max density error
__global__ void resetValues(float* max_velocity, float* max_force, float* sum_rho_err,float* max_rho_err) {
	max_velocity[0] = 0.f;
	max_force[0] = 0.f;
	sum_rho_err[0] = 0.f;
	max_rho_err[0] = 0.f;
	return;
}

//resets pressure values
__global__ void resetPressure(float* pressure, int size) {

	int index = getGlobalIdx_1D_1D();

	if (index >= size) {
		return;
	}

	pressure[index] = 0.f;
	return;
}