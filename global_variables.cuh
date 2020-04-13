#pragma once
#include "common.cuh"

#define _USE_MATH_DEFINES
#include <math.h>

//global variables for physical constants
extern const float rho_0;
extern const vec3d gravity;

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

//gloval variables for CUDA
extern int block_size;

//global variables fopr initialization

extern char main_path[1024];
extern char vtk_group_path[1024];
extern vec3d* POSITION;
extern vec3d* d_POSITION;
extern vec3d* VELOCITY;
extern vec3d* d_VELOCITY;
extern vec3d* ST_FORCE;
extern vec3d* d_ST_FORCE;
extern vec3d* VISCOSITY_FORCE;
extern vec3d* d_VISCOSITY_FORCE;
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