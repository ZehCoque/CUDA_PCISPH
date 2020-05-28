#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <windows.h>
#include <cctype>
#include <cstring>
#include <cmath>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include "dirent.h"
#include <sstream>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

typedef unsigned int uint;

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

// simulation parameters
struct SimParams
{
	float h;
	float invh;

	uint N; //number of fluid particles
	uint B; //number of bondary particles
	uint T; //total number of particles

	//variables for hashtable
	size_t pitch; //this variable is defined by the GPU when the cudaMallocPitch runs
	uint particles_per_row; //this is the maximum number of neighbors a particle can have due to memory allocation
	uint hashtable_size; //this is the size of the hashtable. Must be a power of 2.

	float boundary_diameter;

	//physical constants
	float rho_0; //rest density
	float visc_const; //viscosity constant
	float st_const; // surface tension constant
	float epsilon; // dumping coefficient for collision

	float3* d_POSITION; //stores the pointer to the position data in the GPU
	float3* d_PRED_POSITION; //stores the pointer to the predicted position data in the GPU
	float3* d_VELOCITY; //stores the pointer to the velocity data in the GPU
	float3* d_PRED_VELOCITY; //stores the pointer to the predicted data in the GPU
	float3* d_ST_FORCE; //stores the pointer to the surface tension force data in the GPU
	float3* d_VISCOSITY_FORCE; //stores the pointer to the viscosity force data in the GPU
	float3* d_PRESSURE_FORCE; //stores the pointer to the pressure force data in the GPU
	float3* d_NORMAL; //stores the pointer to the normal data in the GPU
	float* DENSITY; //stores the pointer to the density data in the CPU
	float* d_DENSITY; //stores the pointer to the density data in the GPU
	float* d_PRESSURE; //stores the pointer to the pressure data in the GPU
	float* d_MASS; //stores the pointer to the mass data in the GPU
	int* d_TYPE; //stores the pointer to the type data in the GPU
	int* d_hashtable; //stores the pointer to the hashtable data in the GPU
	float3 gravity; //stores the pointer to the gravity data in the CPU

};

__constant__ SimParams d_params; //parameters stored in the constant memory of GPU