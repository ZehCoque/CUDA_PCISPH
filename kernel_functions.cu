//All functions only callable from the host. 
//The "kernel" in the title refers to how CUDA calls 
//the functions running in the device called from the 
//host (processor).

#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct vec3d
{
    float x, y, z;
};

__global__ void getPositions(vec3d* POSITIONS,float diameter, int SIMULATION_DIMENSION,int total) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= total || j >= total || k >= total) {
        return;
    }
        
    int index = getGlobalIdx_3D_3D();

    POSITIONS[index].x = i * diameter;
    POSITIONS[index].y = j * diameter;
    POSITIONS[index].z = k * diameter;

 
};