#pragma once
#include "device_functions.cuh"
#include "hashing.cuh"
#include "kernels.cuh"
#include "common.cuh"
#include "helper.cuh"
#include "forces.cuh"

__global__ void boundaryPsi(float* psi, int* d_hashtable, const float rho_0, vec3d* points, float h,float invh, int Ncols, size_t pitch, Hash hash, int size) {
	
	int index = getGlobalIdx_1D_1D();

	if (index >= size) {
		return;
	}

	psi[index] = 0.f;
	int hash_list[27];
	bool skip = false;
	int count = 0;
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {
				vec3d BB;
				BB.x = points[index].x + i * h;
				BB.y = points[index].y + j * h;
				BB.z = points[index].z + k * h;

				int hash_index = hash.hashFunction(BB, invh);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)d_hashtable + hash_index * pitch);
					for (int t = 0; t < Ncols; t++) {
						/*int* element = (int*)((char*)d_hashtable + hash_index * pitch + t * sizeof(int));*/
						//printf("%d\n",element[0]);
						if (row[t] != -1) {
							//printf("%g %g %g %d %d %d %d %g %g %g\n", points[index].x, points[index].y, points[index].z,i,j,k,hash_index, points[row[t]].x, points[row[t]].y, points[row[t]].z);
							float r = distance(points[index], points[row[t]]);
							if (r <= h) {
								psi[index] += Poly6_Kernel(r, h, invh);
							}
						}
					}
				}
			}
		}
	}
	
	psi[index] = rho_0 / psi[index];
	
	return;

}

__global__ void boundaryNormal(vec3d* normal,vec3d* points,vec3d b_initial, vec3d b_final,int size) {

	int index = getGlobalIdx_1D_1D();

	if (index >= size) {
		return;
	}

	vec3d point = points[index];

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

__global__ void fluidNormal(vec3d* normal, vec3d* points, float* mass, float* density, int* type, float rho_0,float h, float invh, Hash hash, int* d_hashtable, int Ncols,size_t pitch, int size) {

	int index = getGlobalIdx_1D_1D();
	
	if (index >= size) {
		return;
	}
	
	assignToVec3d(&normal[index]);
	
	int hash_list[27];
	bool skip = false;
	int count = 0;
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {
				vec3d BB;
				BB.x = points[index].x + i * h;
				BB.y = points[index].y + j * h;
				BB.z = points[index].z + k * h;

				int hash_index = hash.hashFunction(BB, invh);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)d_hashtable + hash_index * pitch);
					for (int t = 0; t < Ncols; t++) {
						/*int* element = (int*)((char*)d_hashtable + hash_index * pitch + t * sizeof(int));*/
						//printf("%d\n",element[0]);
						if (row[t] != -1) {
							//printf("%g %g %g %d %d %d %d %g %g %g\n", points[index].x, points[index].y, points[index].z,i,j,k,hash_index, points[row[t]].x, points[row[t]].y, points[row[t]].z);
							float r = distance(points[index], points[row[t]]);
							if (r <= h && r > 0) {

								vec3d poly6_gradient = Poly6_Gradient(index, row[t], points, r, h, invh);
								float tmp;
								if (type[row[t]] == 0) {
									tmp = h * mass[row[t]] / density[row[t]];
								} else if (type[row[t]] == 1) {
									tmp = h * mass[row[t]] / rho_0;
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

	//printf("%d\n", index);
	return;
}

__global__ void nonPressureForces(vec3d* points,vec3d* viscosity_force, vec3d* st_force,float* mass,float* density, vec3d* velocity,vec3d* normal, vec3d gravity, int* type,const float h,const float invh, const float rho_0,const float visc_const, const float st_const,const int Ncols,size_t pitch,int* d_hashtable,Hash hash, int size) {

	int index = getGlobalIdx_1D_1D();

	if (index >= size) {
		return;
	}

	//reseting vec3d value to 0
	assignToVec3d(&viscosity_force[index]);
	assignToVec3d(&st_force[index]);

	vec3d BB;
	int hash_list[27];
	bool skip = false;
	int count = 0;
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {
				
				BB.x = points[index].x + i * h;
				BB.y = points[index].y + j * h;
				BB.z = points[index].z + k * h;

				int hash_index = hash.hashFunction(BB, invh);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)d_hashtable + hash_index * pitch);
					for (int t = 0; t < Ncols; t++) {
						if (row[t] != -1) {
							//printf("%g %g %g %d %d %d %d %g %g %g\n", points[index].x, points[index].y, points[index].z,i,j,k,hash_index, points[row[t]].x, points[row[t]].y, points[row[t]].z);
							float r = distance(points[index], points[row[t]]);
							if (r <= h && r > 0) {

								//Viscosity calculation

								vec3d visc = ViscosityForce(index, row[t], mass, density, velocity, type[row[t]], visc_const, rho_0, Viscosity_Laplacian(r, h, invh));

								//summation of calcualted value to main array
								viscosity_force[index].x += visc.x;
								viscosity_force[index].y += visc.y;
								viscosity_force[index].z += visc.z;

								//Surface tension calculation
								vec3d st = STForce(index, row[t], r, points, mass, density, normal, type[row[t]], st_const, rho_0, ST_Kernel(r, h, invh, type[row[t]]));

								//summation of calcualted value to main array
								st_force[index].x += st.x;
								st_force[index].y += st.y;
								st_force[index].z += st.z;

								
								//printf("[%g %g %g] -> [%g %g %g]\n", st.x,st.y,st.z, st_force[index].x, st_force[index].y, st_force[index].z);
							}
						}
					}
				}
			}
		}
	}
	//printf("[%g %g %g]\n", st_force[index].x, st_force[index].y, st_force[index].z);
	//printf("visc = [%.6f %.6f %.6f] st = [%.6f %.6f %.6f]\n", viscosity_force[index].x, viscosity_force[index].y, viscosity_force[index].z, st_force[index].x, st_force[index].y, st_force[index].z);
	return;
}

__global__ void positionAndVelocity(vec3d* points1,vec3d* velocities1, vec3d* points2, vec3d* velocities2, vec3d* pressure_force, vec3d* viscosity_force, vec3d* st_force,vec3d gravity,float* mass,float delta_t,int size) {

	// array 1 -> Will be changed by this kernel
	// array 2 -> Wont be changed by this kernel

	int index = getGlobalIdx_1D_1D();

	if (index >= size) {
		return;
	}

	float tmp = delta_t / mass[index];
	//printf("%g/%g = %g\n", delta_t, mass[index],tmp);
	//calculating velocity
	velocities1[index].x = velocities2[index].x + (pressure_force[index].x + viscosity_force[index].x + st_force[index].x + gravity.x * mass[index]) * (tmp);
	velocities1[index].y = velocities2[index].y + (pressure_force[index].y + viscosity_force[index].y + st_force[index].y + gravity.y * mass[index]) * (tmp);
	velocities1[index].z = velocities2[index].z + (pressure_force[index].z + viscosity_force[index].z + st_force[index].z + gravity.z * mass[index]) * (tmp);

	//calculating position
	points1[index].x = points2[index].x + delta_t * velocities1[index].x;
	points1[index].y = points2[index].y + delta_t * velocities1[index].y;
	points1[index].z = points2[index].z + delta_t * velocities1[index].z;


	//printf("[%g %g %g] [%g %g %g]", velocities1[index].x, velocities1[index].y, velocities1[index].z, points1[index].x, points1[index].y, points1[index].z);
	

	return;
}

__global__ void collisionHandler(vec3d* points, vec3d* velocities,vec3d* normal,int* type,int* d_hashtable,float h,float invh,size_t pitch, Hash hash,int Ncols,float boundary_diameter,float epsilon,int size) {

	int index = getGlobalIdx_1D_1D();

	if (index >= size) {
		return;
	}

	vec3d n_c_i;
	assignToVec3d(&n_c_i);
	float w_c_ib_sum = 0.f;
	float w_c_ib_second_sum = 0.f;

	vec3d BB;
	int count = 0;
	bool skip = false;
	int hash_list[27];
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {

				BB.x = points[index].x + i * h;
				BB.y = points[index].y + j * h;
				BB.z = points[index].z + k * h;

				int hash_index = hash.hashFunction(BB, invh);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)d_hashtable + hash_index * pitch);
					for (int t = 0; t < Ncols; t++) {
						if (row[t] != -1 && type[row[t]] == 1) {
							float r = distance(points[index], points[row[t]]);
							float w_c_ib = fmaxf((boundary_diameter - r) / boundary_diameter,0.f);
							vec3d n_b = normal[row[t]];
							
							n_c_i.x += n_b.x * w_c_ib;
							n_c_i.y += n_b.y * w_c_ib;
							n_c_i.z += n_b.z * w_c_ib;

							w_c_ib_sum += w_c_ib;
							w_c_ib_second_sum += w_c_ib * (boundary_diameter - r);
						}
					}
				}
			}
		}
	}
	
	if (w_c_ib_sum == 0) {
		return;
	}

	//printf("[%g %g %g] %g %g\n", n_c_i.x, n_c_i.y, n_c_i.z, w_c_ib_sum, w_c_ib_second_sum);
	//calculating new position
	float inv_norm_normal = 1 / norm3df(n_c_i.x, n_c_i.y, n_c_i.z);
	float inv_w = 1 / w_c_ib_sum;
	//printf("[%g %g %g] * %g = [%g %g %g]\n", sum.x, sum.y, sum.z, inv_w, inv_w * sum.x, inv_w * sum.y, inv_w * sum.z);
	points[index].x += n_c_i.x * inv_norm_normal * w_c_ib_second_sum * inv_w;
	points[index].y += n_c_i.y * inv_norm_normal * w_c_ib_second_sum * inv_w;
	points[index].z += n_c_i.z * inv_norm_normal * w_c_ib_second_sum * inv_w;

	//calculating new velocity
	float dot = dot_product(velocities[index], normal[index]);
	vec3d v_n;
	v_n.x = dot * n_c_i.x;
	v_n.y = dot * n_c_i.y;
	v_n.z = dot * n_c_i.z;

	velocities[index].x = epsilon * (velocities[index].x - v_n.x);
	velocities[index].y = epsilon * (velocities[index].x - v_n.y);
	velocities[index].z = epsilon * (velocities[index].x - v_n.z);

	return;
}

__global__ void DensityCalc(vec3d* points, float* mass, float* density, const float h, const float invh, const float rho_0, const int Ncols, size_t pitch, int* d_hashtable, Hash hash, int size) {

	int index = getGlobalIdx_1D_1D();

	if (index >= size) {
		return;
	}

	density[index] = 0.f;
	
	vec3d BB;
	int hash_list[27];
	bool skip = false;
	int count = 0;
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {

				BB.x = points[index].x + i * h;
				BB.y = points[index].y + j * h;
				BB.z = points[index].z + k * h;

				int hash_index = hash.hashFunction(BB, invh);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)d_hashtable + hash_index * pitch);
					for (int t = 0; t < Ncols; t++) {
						if (row[t] != -1) {
							float r = distance(points[index], points[row[t]]);
							if (r <= h) {
								density[index] += mass[row[t]] * Poly6_Kernel(r, h, invh);
								//printf("row[t] = %d density = %g mass = % r = %g\n", row[t], density[index], mass[row[t]], r);
								
							}
						}
					}
				}
			}
		}
	}

	return;
}

__global__ void PressureCalc(float* pressure, float* density,float rho_0,float pressure_coeff,int size) {
	
	int index = getGlobalIdx_1D_1D();

	if (index >= size || (density[index] - rho_0) <= 0) {
		return;
	}

	pressure[index] += (density[index] - rho_0) * pressure_coeff;
	

	return;
}

__global__ void PressureForceCalc(vec3d* points, vec3d* pressure_force,float* pressure, float* mass, float* density,int* type, const float h, const float invh, const int Ncols, size_t pitch, int* d_hashtable, Hash hash, int size) {

	int index = getGlobalIdx_1D_1D();

	if (index >= size) {
		return;
	}

	//reseting vec3d value to 0
	assignToVec3d(&pressure_force[index]);
	
	vec3d BB;
	int hash_list[27];
	bool skip = false;
	int count = 0;
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {

				BB.x = points[index].x + i * h;
				BB.y = points[index].y + j * h;
				BB.z = points[index].z + k * h;

				int hash_index = hash.hashFunction(BB, invh);
				hash_list[count] = hash_index;
				skip = false;
				for (int t = 0; t < count; t++) {
					if (hash_index == hash_list[t]) {
						skip = true;
					}
				}
				count = count + 1;
				if (hash_index >= 0 && skip == false) {
					int* row = (int*)((char*)d_hashtable + hash_index * pitch);
					for (int t = 0; t < Ncols; t++) {
						if (row[t] != -1) {
							
							float r = distance(points[index], points[row[t]]);
							if (r <= h && r > 0) {

								vec3d spiky_grad = Spiky_Gradient(index, row[t], points, r,  h,  invh);
								vec3d p = PressureForce(index, row[t], pressure, mass, density, type[row[t]], spiky_grad);
								sum2Vec3d(&pressure_force[index],&p);

							}
						}
					}
				}
			}
		}
	}

	return;
}

__global__ void getMaxVandF(float* max_velocity, float* max_force, vec3d* velocities, vec3d* pressure_force, vec3d* viscosity_force, vec3d* st_force, vec3d gravity,float* mass,float* density,float* sum_rho_error,float* max_rho_err,float rho_0,int size) {

	int index = getGlobalIdx_1D_1D();

	if (index >= size) {
		return;
	}
	
	float max_p = maxValueInVec3D(pressure_force[index]);
	float max_v = maxValueInVec3D(viscosity_force[index]);
	float max_st = maxValueInVec3D(st_force[index]);

	vec3d g;
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

__global__ void hashtableReset(int* d_hashtable,int Ncols,size_t pitch, int size) {

	int index = getGlobalIdx_1D_1D();

	if (index >= size) {
		return;
	}

	int* row = (int*)((char*)d_hashtable + index * pitch);
	for (int t = 0; t < Ncols; t++) {
		row[t] = -1;
	}

	return;

}

__global__ void resetValues(float* max_velocity, float* max_force, float* sum_rho_err,float* max_rho_err) {
	max_velocity[0] = 0.f;
	max_force[0] = 0.f;
	sum_rho_err[0] = 0.f;
	max_rho_err[0] = 0.f;
	return;
}

__global__ void resetPressure(float* pressure, int size) {

	int index = getGlobalIdx_1D_1D();

	if (index >= size) {
		return;
	}

	pressure[index] = 0.f;
	return;
}