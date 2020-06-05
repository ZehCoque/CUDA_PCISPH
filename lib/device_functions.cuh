#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ uint getGlobalIdx_3D_3D() {
	uint blockId = blockIdx.x + blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
	uint threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
		+ (threadIdx.z * (blockDim.x * blockDim.y))
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
};

__device__ uint getGlobalIdx_2D_2D() {
	uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
	uint threadId = blockId * (blockDim.x * blockDim.y)
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}

__device__ uint getGlobalIdx_1D_1D() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ uint getGlobalIdx_3D_1D() {
	uint blockId = blockIdx.x + blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
	uint threadId = blockId * blockDim.x + threadIdx.x;
	return threadId;
}