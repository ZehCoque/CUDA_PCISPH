#pragma once
#include "common.cuh"

#define _USE_MATH_DEFINES
#include <math.h>

//global variables for physical constants
extern const float rho_0;
extern const float visc_const;
extern const float st_const;
extern vec3d gravity;

//global variables for time and iteration number tracking
extern int iteration;
extern float simulation_time;
extern float final_time;

//global variables for initial conditions
extern const float PARTICLE_RADIUS;
extern const float MASS_calc;
extern const float PARTICLE_DIAMETER;
extern const float F_INITIAL_POSITION[3];
extern const float F_FINAL_POSITION[3];
extern const float B_INITIAL_POSITION[3];
extern const float B_FINAL_POSITION[3];

//global variables for CUDA
extern int block_size;

//variables for hashtable
extern size_t pitch;
extern const int particles_per_row;
extern int hashtable_size;
extern const int n_p_neighbors;

//number of particles
extern int N; //fluid particles
extern int B; //bondary particles
extern int T; //total particles

//simulation parameters
extern float h;
extern float invh;

//global variables fopr initialization
extern char main_path[1024];
extern char vtk_group_path[1024];
extern char vtu_fullpath[1024];
extern char vtu_path[1024];
extern float** pointData[2];
extern int size_pointData;
extern vec3d** vectorData[4];
extern int size_vectorData;
extern std::string pointDataNames[2];
extern std::string vectorDataNames[4];
extern vec3d* POSITION;
extern vec3d* d_POSITION;
extern vec3d* VELOCITY;
extern vec3d* d_VELOCITY;
extern vec3d* PRESSURE_FORCE;
extern vec3d* d_PRESSURE_FORCE;
extern vec3d* NORMAL;
extern vec3d* d_NORMAL;
extern float* DENSITY;
extern float* d_DENSITY;
extern float* PRESSURE;
extern float* d_PRESSURE;
extern float* MASS;
extern float* d_MASS;
extern int* TYPE;
extern int* d_TYPE;
extern int* hashtable;
extern int* d_hashtable;