//defining 3 prime numbers

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
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

};

Hash::Hash(int b)
{
    this->hashtable_size = b;
    //cudaPointerAttributes* atributes = new cudaPointerAttributes;
    //cudaPointerGetAttributes(atributes, this->hashtable);
    //printf("%d %d %d hptr = %p dptr = %p\n", static_cast<int>(atributes->memoryType), atributes->device, atributes->isManaged, atributes->devicePointer, atributes->hostPointer);

}

//void initializeHashtable(int* hashtable[],int* d_hashtable,int hashtable_size,int particles_per_row,size_t pitch) {
//   
//    int count = 0;
//    for (int i = 0; i < hashtable_size; i++) {
//        for (int j = 0; j < particles_per_row; j++) {
//            hashtable[i][j] = count;
//            count++;
//        }
//    }
//
//    printf("hashtable at host position [2][15] = %d\n", hashtable[2][15]);
//
//    size_t width = particles_per_row * sizeof(int);
//    size_t height = hashtable_size;
// 
//    gpuErrchk(cudaMallocPitch(&d_hashtable, &pitch, particles_per_row * sizeof(int), hashtable_size));
//    gpuErrchk(cudaMemcpy2D(d_hashtable,pitch, hashtable, particles_per_row*sizeof(int), width, height,cudaMemcpyHostToDevice));
//    cudaPointerAttributes* atributes = new cudaPointerAttributes;
//    cudaPointerGetAttributes(atributes, d_hashtable);
//    printf("%d %d %d hptr = %p dptr = %p\n", static_cast<int>(atributes->memoryType), atributes->device, atributes->isManaged, atributes->devicePointer, atributes->hostPointer);
//    return;
//}

__device__ int Hash::hashFunction(vec3d point,float h) {

    int r_x,r_y,r_z;

    r_x = static_cast<int>(floor(point.x/h));
    r_y = static_cast<int>(floor(point.y/h));
    r_z = static_cast<int>(floor(point.z/h));
    //printf("%d\n",(r_x ^ r_y ^ r_z) % hashtable_size);
    return ((r_x ^ r_y ^ r_z) % hashtable_size);
}

__device__ void Hash::insertItem(int* hashtable,vec3d point,int point_id,float h,size_t pitch,int Ncols)
{

    int hash_index = hashFunction(point,h);
    printf("[%g %g %g] -> %d\n", point.x, point.y, point.z, hash_index);
    int* row_a = (int*)((char*)hashtable + hash_index * pitch);
    for (int i = 0; i < Ncols; i++) {
        atomicCAS(&row_a[i], -1, point_id);
        if (row_a[i] == point_id) {
            return;
        }
    }


    //
    //printf("hashtable at device position [2][15] = %d\n", hashtable[2][15]);
    //printf("%p\n", hashtable);
}

// __device__ void Hash::deleteItem(int key)
// {
//   // get the hash index of key
//   int index = hashFunction(key);

//   // find the key in (inex)th list
//   std::list <int> :: iterator i;
//   for (i = table[index].begin();
//            i != table[index].end(); i++) {
//     if (*i == key)
//       break;
//   }

//   // if key is found in hash table, remove it
//   if (i != table[index].end())
//     table[index].erase(i);
// }

// function to display hash table
//void Hash::displayHash() {
//   printf("display Hash\n");
//   int** points;
//   points = (int**)malloc(200 * 3 * sizeof(float));
// 
//   gpuErrchk(cudaMemcpy2D(points,this->pitch,hashtable,this->pitch, 200 * sizeof(int),this->hashtable_size, cudaMemcpyDeviceToHost));
//   for (int i = 0; i < hashtable_size; i++) {
//       printf("table index: d% -> particle indexes: {", i);
//       for (int j = 0; j < 200; j++) {
//           printf("%d,", points[i][j]);
//     }
//       printf("}\n");
//   }
// }


__global__ void hashParticlePositions_firstTime(int * d_hashtable,vec3d* points, float h,Hash hash,int size,size_t pitch,int Ncols){

    int index = getGlobalIdx_1D_1D();

    if (index >= size) {
        return;
    }

    hash.insertItem(d_hashtable,points[index],index,h, pitch, Ncols);
    return;
}

//__global__ void test_kernel_2D(int* devPtr, size_t pitch,int Nrows, int Ncols)
//{
//
//    int row = threadIdx.x + blockIdx.x * blockDim.x;
//    int col = threadIdx.y + blockIdx.y * blockDim.y;
//
//    if (row < Nrows && col < Ncols) {
//        int* row_a = (int*)((char*)devPtr + row * pitch);
//        row_a[col] = 9;
//    }
//
//
//
//    return;
//
//}

int main(){

    float h = 2.5;
    int size = 3;
    vec3d* points = new vec3d[size];
    points[0].x = 0.252;
    points[0].y = 1.524;
    points[0].z = 5.45;

    points[1].x = 6.545;
    points[1].y = 0;
    points[1].z = 1.7;

    points[2].x = 6.545;
    points[2].y = 0;
    points[2].z = 1.7;

    const int hashtable_size = nextPrime(200);

    vec3d* d_points;
    gpuErrchk(cudaMalloc((void**)&d_points,  3*size*sizeof(float)));
    gpuErrchk(cudaMemcpy(d_points, points, 3 *size* sizeof(float), cudaMemcpyHostToDevice));

    Hash hash(hashtable_size);
    const int particles_per_row = 200;
    size_t pitch = 0;
    int* hashtable = new int[hashtable_size * particles_per_row];
    for (int i = 0; i < hashtable_size; ++i) {
        for (int j = 0; j < particles_per_row; j++) {
            hashtable[i * particles_per_row + j] = -1;
        }
    }
    //    hashtable[i] = new int[particles_per_row];

    int *d_hashtable;
    
 /*   int count = 0;
    for (int i = 0; i < hashtable_size; i++) {
        for (int j = 0; j < particles_per_row; j++) {
            hashtable[i][j] = count;
            count++;
        }
    }*/

    //printf("hashtable at host position [2] = %d\n", hashtable[2]);

    size_t width = particles_per_row * sizeof(int);
    size_t height = hashtable_size;

    gpuErrchk(cudaMallocPitch(&d_hashtable, &pitch, particles_per_row * sizeof(int), hashtable_size));
    gpuErrchk(cudaMemcpy2D(d_hashtable, pitch, hashtable, particles_per_row * sizeof(int), width, height, cudaMemcpyHostToDevice));

    //int block_size = 1024;
    //int grid_size = size / block_size + 1;

    //hashParticlePositions<<< grid_size, block_size >>>(d_hashtable,d_points,h,hash,size);

    int block_size = 1024;
    int grid_size = size / block_size + 1;
    hashParticlePositions_firstTime << <grid_size, block_size >> > (d_hashtable, d_points, h, hash, size, pitch, particles_per_row);
    //test_kernel_2D <<<grid_size, block_size>>> (d_hashtable, pitch,hashtable_size,particles_per_row);

    gpuErrchk(cudaMemcpy2D(hashtable, particles_per_row * sizeof(int), d_hashtable,pitch , width, height, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    //hash.displayHash();
    //cudaDeviceReset();

    cudaFree(d_points);
    cudaFree(d_hashtable);

}