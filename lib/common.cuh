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
	//PCISPH parameters
	float h;
	float invh;
	float boundary_diameter;
	float pressure_delta;
	float mass;

	//particle counter
	uint N; //number of fluid particles
	uint B; //number of bondary particles
	uint T; //total number of particles

	//variables for hashtable
	size_t pitch; //this variable is defined by the GPU when the cudaMallocPitch runs
	uint particles_per_row; //this is the maximum number of neighbors a particle can have due to memory allocation
	uint hashtable_size; //this is the size of the hashtable. Must be a power of 2.

	//physical constants
	float rho_0; //rest density
	float visc_const; //viscosity constant
	float st_const; // surface tension constant
	float epsilon; // damping coefficient for collision

	float3 gravity; //stores the pointer to the gravity data in the CPU

};

__constant__ SimParams d_params; //parameters stored in the constant memory of GPU