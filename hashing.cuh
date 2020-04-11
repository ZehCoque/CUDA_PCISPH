#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "device_functions.cuh"
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

class Hash
{
private:
    int hashtable_size;    // No. of buckets
    int p1 = 73856093;
    int p2 = 19349669;
    int p3 = 83492791;
             
public:

    Hash(int);  // Constructor

    __device__ void insertItem(int*,vec3d,int,float, size_t,int);

    __device__ int hashFunction(vec3d,float);
    
    __device__ int* getPossibleNeighbors(int*, vec3d, float,int, size_t);

};

Hash::Hash(int b)
{
    this->hashtable_size = b;
    
}


__device__ int Hash::hashFunction(vec3d point,float h) {

    int r_x,r_y,r_z;

    r_x = static_cast<int>(floor(point.x/h)) * this->p1;
    r_y = static_cast<int>(floor(point.y/h)) * this->p2;
    r_z = static_cast<int>(floor(point.z/h)) * this->p3;
    //printf("[%g %g %g] -> %d\n", point.x, point.y, point.z, (r_x ^ r_y ^ r_z) & this->hashtable_size);
    //printf("%d %d %d\n", (r_x ^ r_y ^ r_z), this->hashtable_size,(r_x ^ r_y ^ r_z) % this->hashtable_size);
    return ((r_x ^ r_y ^ r_z) & (this->hashtable_size - 1));
}

__device__ void Hash::insertItem(int* hashtable, vec3d point, int point_id, float h, size_t pitch, int Ncols)
{
    int hash_index = hashFunction(point, h);
    /*printf("[%g %g %g] -> %d\n", point.x, point.y, point.z, hash_index);*/
    int* row_a = (int*)((char*)hashtable + hash_index * pitch);
    for (int i = 0; i < Ncols; i++) {
        atomicCAS(&row_a[i], -1, point_id);
        if (row_a[i] == point_id) {
            return;
        }
    }
}

__device__ int* Hash::getPossibleNeighbors(int* hashtable, vec3d point,float h,int Ncols,size_t pitch) {
    
    vec3d BB;
    int possible_neighbors[800];
    int count = 0;
    for (int i = 0; i < 800; i++) {
        possible_neighbors[i] = -1;
    }
  
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            for (int k = -1; k <= 1; k++) {
                BB.x = point.x + i * h;
                BB.y = point.y + j * h;
                BB.z = point.z + k * h;
                int hash_index = hashFunction(BB, h);
                //printf("[%g %g %g] + [%g %g %g] = [%g %g %g] -> %d\n", point.x, point.y, point.z,h*i,h*j,h*k,BB.x,BB.y,BB.z, hash_index);
                if (hash_index >= 0) {
                    int* row_a = (int*)((char*)hashtable + hash_index * pitch);
                    for (int t = 0; t < Ncols; t++) {
                        if (row_a[t] != -1) {
                            possible_neighbors[count] = row_a[t];
                            count++;
                        }
                    }
                    
                }
            }
        }
    } 
    for (int t = 0; t < 800; t++) {
        int check_value = possible_neighbors[t];
        if (check_value != -1) {
            for (int g = t + 1; g < 800; g++) {
                if (possible_neighbors[g] == check_value) {
                    possible_neighbors[g] = -1;
                }
            }
        }
    }

    return possible_neighbors;
}

__global__ void hashParticlePositions(int * d_hashtable,vec3d* points, float h,Hash hash,int size,size_t pitch,int Ncols){

    int index = getGlobalIdx_1D_1D();

    if (index >= size) {
        return;
    }

    hash.insertItem(d_hashtable,points[index],index,h, pitch, Ncols);
    return;
}

//int main(){
//
//    float h = 2.5;
//    int size = 3;
//    vec3d* points = new vec3d[size];
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
//    vec3d* d_points;
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