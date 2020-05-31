#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "device_functions.cuh"
#include "helper.cuh"
#include "common.cuh"

bool isPrime(int n)
{
    // Corner cases
    if (n <= 1)  return false;
    if (n <= 3)  return true;

    // This is checked so that we can skip
    // middle five numbers in below loop
    if (n%2 == 0 || n%3 == 0) return false;

    for (int i=5; i*i<=n; i=i+6)
        if (n%i == 0 || n%(i+2) == 0)
           return false;

    return true;
}

// Function to return the smallest
// prime number greater than N
int nextPrime(int N)
{

    // Base case
    if (N <= 1)
        return 2;

    int prime = N;
    bool found = false;

    // Loop continuously until isPrime returns
    // true for a number greater than n
    while (!found) {
        prime++;

        if (isPrime(prime))
            found = true;
    }

    return prime;
}

__device__ int hashFunction(float3 point) {

    int p1 = 73856093;
    int p2 = 19349669;
    int p3 = 83492791;

    int r_x, r_y, r_z;

    r_x = static_cast<int>(floor(point.x * d_params.invh)) * p1;
    r_y = static_cast<int>(floor(point.y * d_params.invh)) * p2;
    r_z = static_cast<int>(floor(point.z * d_params.invh)) * p3;

    return ((r_x ^ r_y ^ r_z) & (d_params.hashtable_size - 1));
}

__device__ void insertItem(float3 point, int point_id, int* hashtable)
{
    int hash_index = hashFunction(point);

    int* row_a = (int*)((char*)hashtable + hash_index * d_params.pitch);
    for (int i = 0; i < d_params.particles_per_row; i++) {
        atomicCAS(&row_a[i], -1, point_id);
        if (row_a[i] == point_id) {
            return;
        }
    }
}

__global__ void hashParticlePositions(float3* position, uint size, int* hashtable) {

    int index = getGlobalIdx_1D_1D();

    if (index >= size) {
        return;
    }
    printf("%g\n", d_params.invh);
    insertItem(position[index], index, hashtable);
    
    return;
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