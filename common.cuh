#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <windows.h>
#include <cctype>
#include <cstring>
#include <cmath>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include "include/dirent.h"
#include <sstream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

struct vec3d
{
	float x, y, z;
};

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}