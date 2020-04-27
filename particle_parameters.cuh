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



}

__global__ void fluidNormal(vec3d* normal, vec3d* points, float* mass, float* density, float h, float invh, Hash hash, int* d_hashtable, int Ncols,size_t pitch, int size) {

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
							if (r <= h) {

								vec3d poly6_gradient = Poly6_Gradient(index, row[t], points, r, h, invh);
								float tmp = h * mass[row[t]] / density[row[t]];
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

__global__ void nonPressureForces(vec3d* points,vec3d* viscosity_force, vec3d* st_force,float* mass,float* density, vec3d* velocity,vec3d* normal, vec3d gravity,const float h,const float invh, const float rho_0,const float visc_const, const float st_const,const int Ncols,size_t pitch,int* d_hashtable,Hash hash, int size) {

	int index = getGlobalIdx_1D_1D();

	if (index >= size) {
		return;
	}

	//reseting vec3d valus to 0
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
								vec3d visc = Viscosity(index, row[t], mass, density, velocity, visc_const, Viscosity_Laplacian(r, h, invh));

								//summation of calcualted value to main array
								viscosity_force[index].x += visc.x;
								viscosity_force[index].y += visc.y;
								viscosity_force[index].z += visc.z;

								//Surface tension calculation
								vec3d st = ST(index, row[t], r, points, mass, density, normal, st_const, rho_0, ST_Kernel(r, h, invh));

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

__global__ void positionAndVelocity(vec3d* points,vec3d* velocities,vec3d* pressure_force, vec3d* viscosity_force, vec3d* st_force,vec3d gravity,vec3d* normal,float* w,int* type,float* mass,float delta_t,float boundary_diameter,float h,float invh,Hash hash,int Ncols,int* d_hashtable,size_t pitch,int size) {

	int index = getGlobalIdx_1D_1D();

	if (index >= size) {
		return;
	}

	float tmp = delta_t / mass[index];
	//printf("%g/%g = %g\n", delta_t, mass[index],tmp);
	//calculating velocity
	velocities[index].x = velocities[index].x + (pressure_force[index].x + viscosity_force[index].x + st_force[index].x + gravity.x) * (tmp);
	velocities[index].y = velocities[index].y + (pressure_force[index].y + viscosity_force[index].y + st_force[index].y + gravity.y) * (tmp);
	velocities[index].z = velocities[index].z + (pressure_force[index].z + viscosity_force[index].z + st_force[index].z + gravity.z) * (tmp);

	//calculating position
	points[index].x = points[index].x + delta_t * velocities[index].x;
	points[index].y = points[index].y + delta_t * velocities[index].y;
	points[index].z = points[index].z + delta_t * velocities[index].z;

	//calculating normal for collisions between fluid and boundary particles
	w[index] = 0.f;
	assignToVec3d(&normal[index]);
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
						
							//printf("%d\n", row[t]);
						
						
						if (row[t] != -1  && type[row[t]] == 1) {
							float r = distance(points[index], points[row[t]]);
							float inst_w = (boundary_diameter - r) / boundary_diameter;
							/*printf("(%g - %r)/%g = %g", boundary_diameter,r, boundary_diameter,inst_w);*/
							if (inst_w > 0) {
								
								w[index] += inst_w;

								normal[index].x += inst_w * normal[row[t]].x;
								normal[index].y += inst_w * normal[row[t]].y;
								normal[index].z += inst_w * normal[row[t]].z;

							}
							
						}
					}
				}
			}
		}
	}

	return;
}

__global__ void collisionHandler(vec3d* points, vec3d* velocities,vec3d* normal, float* w,int* type,int* d_hashtable,float h,float invh,size_t pitch, Hash hash,int Ncols,float boundary_diameter,float epsilon,int size) {

	int index = getGlobalIdx_1D_1D();

	if (index >= size || w[index] == 0.f) {
		return;
	}

	vec3d BB;
	int count = 0;
	bool skip = false;
	int hash_list[27];
	vec3d sum;
	assignToVec3d(&sum);

	float norm_normal = norm3df(normal[index].x, normal[index].y, normal[index].z);

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
							float inst_w = (boundary_diameter - r) / boundary_diameter;

							float tmp = inst_w * (boundary_diameter - r) / norm_normal;

							sum.x += tmp * normal[index].x;
							sum.y += tmp * normal[index].y;
							sum.z += tmp * normal[index].z;

						}
					}
				}
			}
		}
	}


	//calculating new position
	float tmp = 1 / w[index];
	points[index].x = points[index].x + tmp * sum.x;
	points[index].y = points[index].x + tmp * sum.y;
	points[index].z = points[index].x + tmp * sum.z;

	//calculating new velocity
	float dot = dot_product(velocities[index], normal[index]);
	vec3d v_n;
	v_n.x = dot * normal[index].x;
	v_n.y = dot * normal[index].y;
	v_n.z = dot * normal[index].z;

	velocities[index].x = epsilon * (velocities[index].x - v_n.x);
	velocities[index].y = epsilon * (velocities[index].x - v_n.y);
	velocities[index].z = epsilon * (velocities[index].x - v_n.z);

	return;
}

__global__ void return_hashvalue(vec3d av_point,vec3d* points,Hash hash, float invh, float h, size_t pitch, int* d_hashtable,int Ncols,vec3d* d_tmp_points, int* d_tmp_count) {

	vec3d BB;
	int hash_list[27];
	bool skip = false;
	int count = 0;
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {

				BB.x = av_point.x + i * h;
				BB.y = av_point.y + j * h;
				BB.z = av_point.z + k * h;

				int hash_index = hash.hashFunction(BB, invh);
				printf("%d - > [", hash_index);
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
						if (row[t] != -1 ) {
							printf("%d ", row[t]);
							atomicAdd(d_tmp_count, 1);
							d_tmp_points[d_tmp_count[0]].x = points[row[t]].x;
							d_tmp_points[d_tmp_count[0]].y = points[row[t]].y;
							d_tmp_points[d_tmp_count[0]].z = points[row[t]].z;

						}
					}
				}
				printf("]\n");
			}
		}
	}

}