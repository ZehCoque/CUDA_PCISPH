#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ int getGlobalIdx_3D_3D() {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
		+ (threadIdx.z * (blockDim.x * blockDim.y))
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
};

__device__ int getGlobalIdx_2D_2D() {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y)
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}

__device__ int getGlobalIdx_1D_1D() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int getGlobalIdx_3D_1D() {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * blockDim.x + threadIdx.x;
	return threadId;
}