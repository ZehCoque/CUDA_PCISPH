#define _USE_MATH_DEFINES

#include <iostream>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include "device_functions.cu"
#include "kernel_functions.cu"
#include <stdio.h>
#include "utilities.cu"
#include "VTK.cu"

// Initial conditions
const float PARTICLE_RADIUS = 1/100.f;
const float mass = M_PI * pow(PARTICLE_RADIUS,3)/3*4;
const float PARTICLE_DIAMETER = 2 * PARTICLE_RADIUS;
const float STARTING_POSITION[3] = { 0,0,0 };
const float FINAL_POSITION[3] = { 1,1,1 };
int NPD[3];
float VOLUME = 1;
const int SIMULATION_DIMENSION = 3;
const int x = 40; // Number of particles inside the smoothing length
vec3d gravity;
gravity.x = 0;
gravity.y = -9.81;
gravity.z = 0;

int iteration = 1;
float simulation_time = 0;

// Value for PI -> M_PI

// Algorithm variables
// int vtu_no_of_lines = 0;

int main(void)
{
    // get main path of simulation
    char main_path[1024];
    getMainPath(main_path);

    // write path for vtu files
    char vtu_path[1024];
    strcpy(vtu_path,main_path);
    strcat(vtu_path,"/vtu");

    // write path for vtk group file
    char vtk_group_path[1024];
    strcpy(vtk_group_path,main_path);
    strcat(vtk_group_path,"/PCISPH.pvd");

    // create directory for vtu files
    CreateDir(vtu_path);
    // Get number per dimension (NPD) of particles
    for (int i = 0; i < 3; i++) {
        NPD[i] = ceil((FINAL_POSITION[i] - STARTING_POSITION[i]) / PARTICLE_DIAMETER);
        VOLUME = VOLUME * (FINAL_POSITION[i] - STARTING_POSITION[i]);
    }
    
    int N = NPD[0] * NPD[1] * NPD[2];
    int SIM_SIZE = N * SIMULATION_DIMENSION;
    const float h = pow(3 * VOLUME * x/(4*M_PI*N),1/3.f);

    //const float boundary_radius = h/4;
    //const float boundary_diameter = h/2;

    //printf("%g\n",h);

    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);

    vec3d* POSITIONS;

    // Allocate Unified Memory accessible from CPU or GPU
    cudaMallocManaged(&POSITIONS, SIM_SIZE * sizeof(float));

    // Define grid and block allocations for CUDA kernel function
    dim3 block(1, 1, 1);
    dim3 grid(NPD[0], NPD[1], NPD[2]);
    
    //generate locations for each particle
    getPositions<<<grid,block>>>(POSITIONS, PARTICLE_DIAMETER, SIMULATION_DIMENSION, SIM_SIZE);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    cudaDeviceSynchronize();

    float* density = new float[N];
    for (int i = 0; i < N; i++){
        density[i] = 1000;
    }

    vec3d* velocity = new vec3d[N];
    for (int i = 0; i < N; i++){
        velocity[i].x = i;
        velocity[i].y = i;
        velocity[i].z = i;
    }

    float** pointData[] = {&density};
    int size_pointData = sizeof(pointData)/sizeof(double);
    vec3d** vectorData[] = {&velocity};
    int size_vectorData = sizeof(vectorData)/sizeof(double);
    // std::cout << sizeof(vectorData) << std::endl;
    // std::cout << typeid(vectorData).name() << std::endl;
    std::string pointDataNames[] = {"density"};
    std::string vectorDataNames[] = {"velocity"};

    char vtu_fullpath[1024];
    strcpy(vtu_fullpath,VTU_Writer(vtu_path,iteration,POSITIONS,N,pointData,vectorData,pointDataNames,vectorDataNames,size_pointData,size_vectorData,vtu_fullpath));

    VTK_Group(vtk_group_path,vtu_fullpath,simulation_time);

    // Free memory
    cudaFree(POSITIONS);

    return 0;
}