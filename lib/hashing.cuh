#pragma once
#include <device_launch_parameters.h>
#include "device_functions.cuh"
#include "helper.cuh"
#include "common.cuh"

#include "thrust/device_ptr.h"
#include "thrust/sort.h"

__device__ int3 calcGridPos(float3 point) {

    int3 gridPos;

    gridPos.x = static_cast<unsigned int>(floor(point.x * d_params.invh));
    gridPos.y = static_cast<unsigned int>(floor(point.y * d_params.invh));
    gridPos.z = static_cast<unsigned int>(floor(point.z * d_params.invh));

    return gridPos;
}

__device__ uint calcGridHash(int3 gridPos) {

    uint p1 = 73856093;
    uint p2 = 19349669;
    uint p3 = 83492791;
    
    int3 __gridPos = make_int3(gridPos.x * p1, gridPos.y * p2, gridPos.z * p3);

    return ((__gridPos.x ^ __gridPos.y ^ __gridPos.z) & (d_params.hashtable_size - 1));

}

__global__ void hashParticlePositions(float3* position, uint* gridParticleHash, uint* gridParticleIndex, uint size = d_params.T) {

    uint index = getGlobalIdx_1D_1D();

    if (index >= size) {
        return;
    }

    float3 p = position[index];

    int3 gridPos = calcGridPos(p);
    uint hash = calcGridHash(gridPos);
    
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;

    return;
}

void sortParticles(uint* dGridParticleHash, uint* dGridParticleIndex, uint numParticles)
{
    thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
        thrust::device_ptr<uint>(dGridParticleHash + numParticles),
        thrust::device_ptr<uint>(dGridParticleIndex));
}

__global__ void getCellAndStartEnd( uint* cellStart,        
                                    uint* cellEnd,                 
                                    uint* gridParticleHash,
                                    uint size = d_params.T
                                    ) {

    extern __shared__ uint sharedHash[];

    uint index = getGlobalIdx_1D_1D();

    if (index >= size) {
        return;
    }

    uint hash = gridParticleHash[index];

    sharedHash[threadIdx.x + 1] = hash;

    if (index > 0 && threadIdx.x == 0) {

        sharedHash[0] = gridParticleHash[index - 1];

    }

    __syncthreads();

    if (index == 0 || hash != sharedHash[threadIdx.x])
    {
        cellStart[hash] = index;

        if (index > 0)
            cellEnd[sharedHash[threadIdx.x]] = index;
    }

    if (index == size - 1)
    {
        cellEnd[hash] = index + 1;
    }

}

__global__ void sortArrays_float3(float3* sortedArray, float3* oldArray, uint* gridParticleIndex, uint size = d_params.T) {
    
    uint index = getGlobalIdx_1D_1D();

    if (index >= size) {
        return;
    }
    
    uint sortedIndex = gridParticleIndex[index];

    float3 sorted = oldArray[sortedIndex];

    sortedArray[index] = sorted;

}

__global__ void sortArrays_float(float* sortedArray, float* oldArray, uint* gridParticleIndex) {
    uint index = getGlobalIdx_1D_1D();

    if (index >= d_params.T) {
        return;
    }

    uint sortedIndex = gridParticleIndex[index];

    sortedArray[index] = oldArray[sortedIndex];
}

__global__ void sortArrays_int(uint* sortedArray, uint* oldArray, uint* gridParticleIndex) {
    uint index = getGlobalIdx_1D_1D();

    if (index >= d_params.T) {
        return;
    }

    uint sortedIndex = gridParticleIndex[index];

    sortedArray[index] = oldArray[sortedIndex];
}