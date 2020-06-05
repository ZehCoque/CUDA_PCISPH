#pragma once
#include <device_launch_parameters.h>
#include "common.cuh"
#include "helper.cuh"

__device__ float3 ViscosityForce(float* mass_i, float* mass_j, float* density_i, float* density_j, float3* velocity_i, float3* velocity_j ,int* type, float Laplacian) {

	float3 viscosity;

	if (*type == 1) {
		float tmp = *mass_i / *density_i * *mass_j / d_params.rho_0 * d_params.visc_const * Laplacian;

		viscosity = make_float3(tmp * (velocity_j->x - velocity_i->x),
			tmp * (velocity_j->y - velocity_i->y),
			tmp * (velocity_j->z - velocity_i->z));

	}
	else {

		float tmp = *mass_i / *density_i * *mass_j / *density_j * d_params.visc_const * Laplacian;
		
		viscosity = make_float3(tmp * (velocity_j->x - velocity_i->x),
			tmp * (velocity_j->y - velocity_i->y),
			tmp * (velocity_j->z - velocity_i->z));
	}

	return viscosity;

}

__device__ float3 STForce(float3* position_i, float3* position_j, float* mass_i, float* mass_j, float* density_i, float* density_j, float3* normal_i, float3* normal_j,int *type, float *r, float ST_Kernel)
{
	float3 st;
	float tmp;

	if (*type == 0) {
		float3 cohesion;
		float3 curvature;

		tmp = -d_params.st_const * *mass_i * *mass_j * ST_Kernel / *r;

		cohesion = make_float3(tmp * (position_i->x - position_j->x), 
		tmp * (position_i->y - position_j->y), 
		tmp * (position_i->z - position_j->z));

		tmp = -d_params.st_const * *mass_i;

		curvature = make_float3(tmp * (normal_i->x - normal_j->x),
			tmp * (normal_i->y - normal_j->y),
			tmp * (normal_i->z - normal_j->z));

		tmp = 2 * d_params.rho_0 / (*density_i + *density_j);

		st = make_float3(tmp * (cohesion.x + curvature.x),
			 tmp * (cohesion.y + curvature.y),
			 tmp * (cohesion.z + curvature.z));

		
	}
	else if (*type == 1) {

		tmp = -d_params.st_const * *mass_i * *mass_j * ST_Kernel / *r;

		st = make_float3(tmp * (position_i->x - position_j->x),
			tmp * (position_i->y - position_j->y),
			tmp * (position_i->z - position_j->z));

	}
	
	return st;

}

__device__ float3 PressureForce(float* pressure_i, float* pressure_j, float* mass_i, float* mass_j, float* density_i, float* density_j, int *type, float3 Spiky_Gradient) {

	float3 p;

	float tmp;

	if (*type == 0) {
		tmp = -*mass_i * *mass_j * (*pressure_i / powf(*density_i, 2) + *pressure_j / powf(*density_j, 2));
	}
	else {
		tmp = -*mass_i * *mass_j * (*pressure_i / powf(*density_i, 2));
	}

	p.x = tmp * Spiky_Gradient.x;
	p.y = tmp * Spiky_Gradient.y;
	p.z = tmp * Spiky_Gradient.z;

	return p;
}