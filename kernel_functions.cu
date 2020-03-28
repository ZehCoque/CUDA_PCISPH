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

__global__ void makePrism(vec3d* position_arr,float diameter,vec3d initial_pos) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int index = getGlobalIdx_3D_3D();

    // Hexagonal packing
    if (j%2==0){
        position_arr[index].x = initial_pos.x + i * diameter;
        position_arr[index].y = initial_pos.y + j * diameter*powf(3,1/2.f)/2.f;
        position_arr[index].z = initial_pos.z + k * diameter;
    } else {
        position_arr[index].x = initial_pos.x + i * diameter + diameter/2.f;
        position_arr[index].y = initial_pos.y + j * diameter * powf(3.f,1/2.f)/2.f;
        position_arr[index].z = initial_pos.z + k * diameter + diameter/2.f;
    }

    return;
};

// __global__ void makeBox(vec3d* position_arr,float diameter, vec3d initial_pos,vec3d final_pos) {

//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;
//     int k = blockIdx.z * blockDim.z + threadIdx.z;
    
//     int index = getGlobalIdx_3D_3D();

//     position_arr[index].x = inf
//     position_arr[index].y = inf
//     position_arr[index].z = inf

//     float inf = std::numeric_limits<float>::infinity();
    
//     // Hexagonal packing
//     if (j%2==0){

//         if (initial_pos.x + i * diameter == initial_pos.x || i * diameter == final_pos.x || 
//             initial_pos.y + j * diameter*powf(3,1/2.f)/2.f == initial_pos.y || j * diameter*powf(3,1/2.f)/2.f == final_pos.y || 
//             initial_pos.z + k * diameter == initial_pos.z || k * diameter == final_pos.z ){
//             position_arr[index].x = initial_pos.x + i * diameter;
//             position_arr[index].y = initial_pos.y + j * diameter*powf(3,1/2.f)/2.f;
//             position_arr[index].z = initial_pos.z + k * diameter;
//         }

//     } else {
        
//         if (initial_pos.x + i * diameter == initial_pos.x || i * diameter == final_pos.x || 
//             initial_pos.y + j * diameter*powf(3,1/2.f)/2.f == initial_pos.y || j * diameter*powf(3,1/2.f)/2.f == final_pos.y || 
//             initial_pos.z + k * diameter == initial_pos.z || k * diameter == final_pos.z ){
//             position_arr[index].x = initial_pos.x + i * diameter + 3.f/2.f*diameter;
//             position_arr[index].y = initial_pos.y + j * diameter*powf(3,1/2.f)/2.f;
//             position_arr[index].z = initial_pos.z + k * diameter + 3.f/2.f*diameter;
//         }
//     }

// };

__global__ void makePlane(vec3d* position_arr,float diameter, vec3d initial_pos,int offset,int orientation){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    int index = getGlobalIdx_3D_3D() + offset;

    // 1 for xy orientation
    // 2 for xz orientation
    // 3 for yz orientation

    if (orientation == 1){

            position_arr[index].x = initial_pos.x + i * diameter;
            position_arr[index].y = initial_pos.y + j * diameter;
            position_arr[index].z = initial_pos.z;
        
    } else if (orientation == 2){
            position_arr[index].x = initial_pos.x + i * diameter;
            position_arr[index].y = initial_pos.y;
            position_arr[index].z = initial_pos.z + k * diameter;

    } else if (orientation == 3){
            position_arr[index].x = initial_pos.x;
            position_arr[index].y = initial_pos.y + j * diameter;
            position_arr[index].z = initial_pos.z + k * diameter;

    }
    return;
}

void makeBox(vec3d* position_arr,float diameter, vec3d initial_pos,vec3d final_pos){
    // char** orientation_order = {"xy","xy","yz","yz","xz","xz"};
    int offset = 0;
    dim3 block(1,1,1);
    int num_x,num_y,num_z;

    //first wall
    num_x = static_cast<int>(ceil((final_pos.x - initial_pos.x)/diameter)) + 1;
    num_y = static_cast<int>(ceil((final_pos.y - initial_pos.y)/diameter)) + 1;
    num_z = 1;
    dim3 grid(num_x,num_y,num_z);
    makePlane<<<grid,block>>>(position_arr,diameter,initial_pos,offset,1);

    offset = offset + num_x*num_y*num_z;
    //printf("%d\n",offset);
    //second wall
    vec3d tmp_initial_pos;
    tmp_initial_pos.x = initial_pos.x;
    tmp_initial_pos.y = initial_pos.y;
    tmp_initial_pos.z = final_pos.z;

    makePlane<<<grid,block>>>(position_arr,diameter,tmp_initial_pos,offset,1);

    offset = offset + num_x*num_y*num_z;
     //printf("%d\n",offset);
    //third wall
    tmp_initial_pos.x = initial_pos.x;
    tmp_initial_pos.y = initial_pos.y;
    tmp_initial_pos.z = initial_pos.z + diameter;

    num_x = static_cast<int>(ceil((final_pos.x - initial_pos.x)/diameter)) + 1;
    num_y = 1;
    num_z = static_cast<int>(ceil((final_pos.z - initial_pos.z)/diameter)) - 1;

    dim3 grid2(num_x,num_y,num_z);

    makePlane<<<grid2,block>>>(position_arr,diameter,tmp_initial_pos,offset,2);

    offset = offset + num_x*num_y*num_z;
    //printf("%d\n",offset);
    //forth wall
    tmp_initial_pos.x = initial_pos.x;
    tmp_initial_pos.y = final_pos.y;
    tmp_initial_pos.z = initial_pos.z + diameter;

    // num_x = static_cast<int>(ceil((final_pos.x - initial_pos.x)/diameter));
    // num_y = 1;
    // num_z = static_cast<int>(ceil((final_pos.z - initial_pos.z)/diameter)) - 1;

    makePlane<<<grid2,block>>>(position_arr,diameter,tmp_initial_pos,offset,2);

    offset = offset + num_x*num_y*num_z;
    //printf("%d\n",offset);
    //fifth wall
    num_x = 1;
    num_y = static_cast<int>(ceil((final_pos.y - initial_pos.y)/diameter)) - 1;
    num_z = static_cast<int>(ceil((final_pos.z - initial_pos.z)/diameter)) - 1;

    tmp_initial_pos.x = initial_pos.x;
    tmp_initial_pos.y = initial_pos.y + diameter;
    tmp_initial_pos.z = initial_pos.z + diameter;

    dim3 grid3(num_x,num_y,num_z);

    makePlane<<<grid3,block>>>(position_arr,diameter,tmp_initial_pos,offset,3);

    offset = offset + num_x*num_y*num_z;
    //printf("%d\n",offset);
    //sixth wall

    tmp_initial_pos.x = final_pos.x;
    tmp_initial_pos.y = initial_pos.y + diameter;
    tmp_initial_pos.z = initial_pos.z + diameter;

    makePlane<<<grid3,block>>>(position_arr,diameter,tmp_initial_pos,offset,3);

    return;
}
