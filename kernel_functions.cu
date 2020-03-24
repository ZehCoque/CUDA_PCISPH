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

__global__ void makePrism(vec3d* position_arr,float diameter) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
        
    int index = getGlobalIdx_3D_3D();

    // Hexagonal packing
    if (j%2==0){
        position_arr[index].x = i * diameter;
        position_arr[index].y = j * diameter*powf(3,1/2.f)/2.f;
        position_arr[index].z = k * diameter;
    } else {
        position_arr[index].x = i * diameter + diameter/2.f;
        position_arr[index].y = j * diameter*powf(3.f,1/2.f)/2.f;
        position_arr[index].z = k * diameter + diameter/2.f;
    }

};

__global__ void makeBox(vec3d* position_arr,float diameter,vec3d* initial_pos,vec3d* final_pos) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
        
    int index = getGlobalIdx_3D_3D();

    // Hexagonal packing
    if (j%2==0){

        if (i == initial_pos.x || i == final_pos.x){
            position_arr[index].x = i * diameter;
        }
        
        if (j == initial_pos.y || j == final_pos.y){
            position_arr[index].y = j * diameter*powf(3,1/2.f)/2.f;
        }

        if (k == initial_pos.z || k == final_pos.z){
            position_arr[index].z = k * diameter;
        }

    } else {
        if (i == initial_pos.x || i == final_pos.x){
            position_arr[index].x = i * diameter + diameter/2.f;
        }
        
        if (j == initial_pos.y || j == final_pos.y){
            position_arr[index].y = j * diameter*powf(3,1/2.f)/2.f;
        }

        if (k == initial_pos.z || k == final_pos.z){
            position_arr[index].z = k * diameter + diameter/2.f;
        }
    }

};