#pragma once
#include "common.cuh"

#define _USE_MATH_DEFINES
#include <math.h>

const float rho_0 = 1000.f;
vec3d gravity;

// Initial conditions
const float PARTICLE_RADIUS = 0.01f;
const float MASS_calc = (float)M_PI * -pow(PARTICLE_RADIUS, 3.f) / 3.f * 4.f;
const float PARTICLE_DIAMETER = 2 * PARTICLE_RADIUS;
const float F_INITIAL_POSITION[3] = { -0.5,-0.5,-0.5 }; //Fluid particles initial position
const float F_FINAL_POSITION[3] = { 0.5,0.5,0.5 }; //Fluid particles final position
const float B_INITIAL_POSITION[3] = { -0.5,-0.5,-0.5 }; //Boundary particles final position
const float B_FINAL_POSITION[3] = { 0.5,0.5,0.5 }; //Boundary particles final position