#pragma once
#include "common.cuh"

#define _USE_MATH_DEFINES
#include <math.h>

extern const float rho_0;
extern const vec3d gravity;
extern int iteration;
extern float simulation_time;

// Initial conditions
extern const float PARTICLE_RADIUS;
extern const float MASS_calc;
extern const float PARTICLE_DIAMETER;
extern const float F_INITIAL_POSITION[3];
extern const float F_FINAL_POSITION[3];
extern const float B_INITIAL_POSITION[3];
extern const float B_FINAL_POSITION[3];

//Initialize global variables

extern int block_size;
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