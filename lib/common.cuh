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