#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "device_functions.cuh"
#include "helper.cuh"
#include "common.cuh"

#include "thrust/device_ptr.h"
#include "thrust/sort.h"

__device__ int3 calcGridPos(float3 point) {

    uint p1 = 73856093;
    uint p2 = 19349669;
    uint p3 = 83492791;

    int3 gridPos;

    gridPos.x = static_cast<unsigned int>(floor(point.x * d_params.invh)) * p1;
    gridPos.y = static_cast<unsigned int>(floor(point.y * d_params.invh)) * p2;
    gridPos.z = static_cast<unsigned int>(floor(point.z * d_params.invh)) * p3;

    return gridPos;
}

__device__ uint calcGridHash(int3 gridPos) {

    
    return ((gridPos.x ^ gridPos.y ^ gridPos.z) & (d_params.hashtable_size - 1));

}

__global__ void hashParticlePositions(uint *gridParticleHash, uint * gridParticleIndex, float3 *position) {

    uint index = getGlobalIdx_1D_1D();

    if (index >= d_params.T) {
        return;
    }

    float3 p = position[index];

    int3 gridPos = calcGridPos(p);
    uint hash = calcGridHash(gridPos);
    
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;

    return;
}

__global__ void hashParticlePositionsBoundary(uint* gridParticleHash, uint* gridParticleIndex, float3* position) {

    uint index = getGlobalIdx_1D_1D();

    if (index >= d_params.B) {
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

__global__ void sortAndGetCellStartEnd( uint* cellStart,        // output: cell start index
                                        uint* cellEnd,          // output: cell end index
                                        float4* sortedPos,        // output: sorted positions
                                        float4* sortedVel,        // output: sorted velocities
                                        uint* gridParticleHash, // input: sorted grid hashes
                                        uint* gridParticleIndex,// input: sorted particle indices
                                        float4* oldPos,           // input: sorted position array
                                        float4* oldVel           // input: sorted velocity array
                                        ) {

    extern __shared__ uint sharedHash[];

    uint index = getGlobalIdx_1D_1D();

    if (index >= d_params.T) {
        return;
    }

    uint hash = gridParticleHash[index];

    sharedHash[threadIdx.x + 1] = hash;

    if (index > 0 && threadIdx.x == 0) {

        sharedHash[0] = gridParticleHash[index - 1];

    }

    __syncthreads;

    if (index == 0 || hash != sharedHash[threadIdx.x])
    {
        cellStart[hash] = index;

        if (index > 0)
            cellEnd[sharedHash[threadIdx.x]] = index;
    }

    if (index == d_params.N - 1)
    {
        cellEnd[hash] = index + 1;
    }

    // Now use the sorted index to reorder the pos and vel data
    uint sortedIndex = gridParticleIndex[index];
    float4 pos = oldPos[sortedIndex];
    float4 vel = oldVel[sortedIndex];

    sortedPos[index] = pos;
    sortedVel[index] = vel;
}

//int main(){
//
//    float h = 2.5;
//    int size = 3;
//    float3* points = new float3[size];
//    points[0].x = 0.252;
//    points[0].y = 1.524;
//    points[0].z = 5.45;
//
//    points[1].x = 6.545;
//    points[1].y = 0;
//    points[1].z = 1.7;
//
//    points[2].x = 6.545;
//    points[2].y = 0;
//    points[2].z = 1.7;
//
//    const int hashtable_size = nextPrime(200);
//
//    float3* d_points;
//    gpuErrchk(cudaMalloc((void**)&d_points,  3*size*sizeof(float)));
//    gpuErrchk(cudaMemcpy(d_points, points, 3 *size* sizeof(float), cudaMemcpyHostToDevice));
//
//    Hash hash(hashtable_size);
//    const int particles_per_row = 200;
//    size_t pitch = 0;
//    int* hashtable = new int[hashtable_size * particles_per_row];
//    for (int i = 0; i < hashtable_size; ++i) {
//        for (int j = 0; j < particles_per_row; j++) {
//            hashtable[i * particles_per_row + j] = -1;
//        }
//    }
//
//    int *d_hashtable;
//    
//
//    size_t width = particles_per_row * sizeof(int);
//    size_t height = hashtable_size;
//
//    gpuErrchk(cudaMallocPitch(&d_hashtable, &pitch, particles_per_row * sizeof(int), hashtable_size));
//    gpuErrchk(cudaMemcpy2D(d_hashtable, pitch, hashtable, particles_per_row * sizeof(int), width, height, cudaMemcpyHostToDevice));
//
//    int block_size = 1024;
//    int grid_size = size / block_size + 1;
//    hashParticlePositions << <grid_size, block_size >> > (d_hashtable, d_points, h, hash, size, pitch, particles_per_row);
//
//
//    gpuErrchk(cudaMemcpy2D(hashtable, particles_per_row * sizeof(int), d_hashtable,pitch , width, height, cudaMemcpyDeviceToHost));
//    cudaDeviceSynchronize();
//
//
//
//    cudaFree(d_points);
//    cudaFree(d_hashtable);
//
//}