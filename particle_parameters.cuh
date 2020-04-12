#pragma once
#include "device_functions.cuh"
#include "hashing.cuh"
#include "kernels.cuh"
#include "common.cuh"

__global__ void boundaryPsi(float* psi, int* d_hashtable, const float rho_0, vec3d* points, float h,float invh, int Ncols, size_t pitch, Hash hash, int size) {
	
	int index = getGlobalIdx_1D_1D();

	if (index >= size) {
		return;
	}

	int possible_neighbors[800];
	hash.getPossibleNeighbors(possible_neighbors,d_hashtable, points[index], h, invh, Ncols, pitch);

	psi[index] = 0.f;
	for (int i = 0; i < 800; i++) {
		if (possible_neighbors[i] != -1) {
			//printf("%d\n", index);
			//printf("%d %g %g %g %d %g %g %g\n", 
				//index, points[index].x, points[index].y, points[index].z, possible_neighbors[i],points[possible_neighbors[i]].x, points[possible_neighbors[i]].y, points[possible_neighbors[i]].z);
			float r = distance(points[index], points[possible_neighbors[i]]);
			if (r <= h) {
				psi[index] += Poly6_Kernel(r, h);
			}
				
		}
	}
	//printf("%d %g\n", index, psi[index]);
	psi[index] = rho_0 / psi[index];
	
	return;

}