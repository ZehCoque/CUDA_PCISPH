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
	
	
	//for (int i = 0; i < 5000; i++) {
	//	if (possible_neighbors[i] != -1) {
	//		//printf("%d\n", index);
	//		//printf("%d %g %g %g %d %g %g %g\n", 
	//			//index, points[index].x, points[index].y, points[index].z, possible_neighbors[i],points[possible_neighbors[i]].x, points[possible_neighbors[i]].y, points[possible_neighbors[i]].z);
	//		float r = distance(points[index], points[possible_neighbors[i]]);
	//		if (r <= h) {
	//			psi[index] += Poly6_Kernel(r, h,invh);
	//		}
	//			
	//	}
	//}
	////printf("%d %g\n", index, psi[index]);
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
	//assignToVec3d(&viscosity_force[index]);
	//assignToVec3d(&st_force[index]);

	viscosity_force[index].x = 0.f;
	viscosity_force[index].y = 0.f;
	viscosity_force[index].z = 0.f;
	st_force[index].x = 0.f;
	st_force[index].y = 0.f;
	st_force[index].z = 0.f;

	vec3d BB;
	vec3d visc;
	vec3d st;

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
								visc = Viscosity(index, row[t], mass, density, velocity, visc_const, Viscosity_Laplacian(r, h, invh));

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