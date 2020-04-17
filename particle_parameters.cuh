#pragma once
#include "device_functions.cuh"
#include "hashing.cuh"
#include "kernels.cuh"
#include "common.cuh"
#include "helper.cuh"
#include "forces.cuh"

__global__ void boundaryPsi(float* psi, int* d_hashtable, const float rho_0, vec3d* points, float h,float invh, int Ncols, size_t pitch, Hash hash, int size, const int n_p_neighbors) {
	
	int index = getGlobalIdx_1D_1D();

	if (index >= size) {
		return;
	}

	int* possible_neighbors = new int[n_p_neighbors];
	possible_neighbors[n_p_neighbors - 1] = 80000;
	printf("%d\n", possible_neighbors[n_p_neighbors - 1]);
	hash.getPossibleNeighbors(possible_neighbors,d_hashtable, points[index], h, invh, Ncols, pitch, n_p_neighbors);
	
	psi[index] = 0.f;
	for (int i = 0; i < n_p_neighbors; i++) {
		if (possible_neighbors[i] != -1) {
			//printf("%d\n", index);
			//printf("%d %g %g %g %d %g %g %g\n", 
				//index, points[index].x, points[index].y, points[index].z, possible_neighbors[i],points[possible_neighbors[i]].x, points[possible_neighbors[i]].y, points[possible_neighbors[i]].z);
			float r = distance(points[index], points[possible_neighbors[i]]);
			if (r <= h) {
				psi[index] += Poly6_Kernel(r, h,invh);
			}
				
		}
	}
	//printf("%d %g\n", index, psi[index]);
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

__global__ void fluidNormal(vec3d* normal, vec3d* points, float* mass, float* density, float h, float invh, Hash hash, int* d_hashtable, int Ncols,size_t pitch, int size,const int n_p_neighbors) {

	int index = getGlobalIdx_1D_1D();
	
	if (index >= size) {
		return;
	}
	
	vec3d point = points[index];

	int* possible_neighbors = new int[n_p_neighbors];
	
	hash.getPossibleNeighbors(possible_neighbors, d_hashtable, point, h, invh, Ncols, pitch, n_p_neighbors);
	
	assignToVec3d(&normal[index]);
	
	for (int i = 0; i < n_p_neighbors; i++) {
		if (possible_neighbors[i] != -1) {
			float r = distance(points[index], points[possible_neighbors[i]]);
			if (r <= h) {
				int i = index;
				int j = possible_neighbors[i];

				vec3d poly6_gradient = Poly6_Gradient(i, j, points, r, h, invh);
				float tmp = h * mass[j] / density[j];
				normal[index].x += tmp * poly6_gradient.x;
				normal[index].y += tmp * poly6_gradient.y;
				normal[index].z += tmp * poly6_gradient.z;
			}
		}
	}
	//printf("%d\n", index);
	return;
}

__global__ void nonPressureForces(vec3d* points,vec3d* viscosity_force, vec3d* st_force,float* mass,float* density, vec3d* velocity,vec3d* normal, vec3d gravity,const float h,const float invh, const float rho_0,const float visc_const, const float st_const,const int Ncols,size_t pitch,int* d_hashtable,Hash hash, int size, int n_p_neighbors) {

	int index = getGlobalIdx_1D_1D();

	printf("%d\n", index);

	if (index >= size) {
		return;
	}

	
	vec3d point = points[index];

	int* possible_neighbors = new int[n_p_neighbors];
	hash.getPossibleNeighbors(possible_neighbors, d_hashtable, point, h, invh, Ncols, pitch, n_p_neighbors);

	//reseting vec3d valus to 0
	assignToVec3d(&viscosity_force[index]);
	assignToVec3d(&st_force[index]);

	//Calculating forces
	for (int i = 0; i < n_p_neighbors; i++) {
		if (possible_neighbors[i] != -1) {
			float r = distance(points[index], points[possible_neighbors[i]]);
			if (r <= h && r > 0) {
				int i = index;
				int j = possible_neighbors[i];

				//Viscosity calculation
				float visc_laplacian = Viscosity_Laplacian(r, h, invh);
				vec3d visc = Viscosity(i, j, mass, density, velocity, visc_const, visc_laplacian);

				//summation of calcualted value to main array
				sum2Vec3d(&viscosity_force[index], &visc);

				//Surface tension calculation
				float st_kernel = ST_Kernel(r,h,invh);
				vec3d st = ST(i, j, r, points, mass, density, normal, st_const, rho_0, st_kernel);

				//summation of calcualted value to main array
				
				sum2Vec3d(&st_force[index], &st);
				
			}
		}
	}

	//printf("visc = [%.6f %.6f %.6f] st = [%.6f %.6f %.6f]\n", viscosity_force[index].x, viscosity_force[index].y, viscosity_force[index].z, st_force[index].x, st_force[index].y, st_force[index].z);
	return;
}