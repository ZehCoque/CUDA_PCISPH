#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "common.cuh"

__device__ vec3d Viscosity(int i, int j,float* mass, float* density,vec3d* velocity, float visc_const, float Laplacian) {

	vec3d viscosity;

	//mass_i / density_i * visc_const * mass_j / density_j * (v_j - v_i) * Laplacian;

	float tmp = mass[i] / density[i] * visc_const * mass[j] / density[j] * Laplacian;

	viscosity.x = tmp * (velocity[j].x - velocity[i].x);
	viscosity.y = tmp * (velocity[j].y - velocity[i].y);
	viscosity.z = tmp * (velocity[j].z - velocity[i].z);

	return viscosity;

}

__device__ vec3d ST(int i, int j,float r, vec3d* points, float* mass, float* density, vec3d* normal,float st_const,float rho_0,float ST_Kernel)
{
	vec3d st;
	vec3d cohesion;
	vec3d curvature;

	float tmp;

	tmp = -st_const * mass[i] * mass[j] * ST_Kernel / r;

	cohesion.x = tmp * (points[i].x - points[j].x);
	cohesion.y = tmp * (points[i].y - points[j].y);
	cohesion.z = tmp * (points[i].z - points[j].z);

	tmp = -st_const * mass[i];

	curvature.x = tmp * (normal[i].x - normal[j].x);
	curvature.y = tmp * (normal[i].y - normal[j].y);
	curvature.z = tmp * (normal[i].z - normal[j].z);

	tmp = 2 * rho_0 / (density[i] + density[j]);

	st.x = tmp * (cohesion.x + curvature.x);
	st.y = tmp * (cohesion.y + curvature.y);
	st.z = tmp * (cohesion.z + curvature.z);

	return st;

}