#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "common.cuh"
#include "helper.cuh"

__device__ vec3d ViscosityForce(int i, int j,float* mass, float* density,vec3d* velocity, float visc_const,float rho_0, float Laplacian) {

	vec3d viscosity;


	float tmp = mass[i] / density[i] * mass[j] / density[j] * visc_const * Laplacian;

	viscosity.x = tmp * (velocity[j].x - velocity[i].x);
	viscosity.y = tmp * (velocity[j].y - velocity[i].y);
	viscosity.z = tmp * (velocity[j].z - velocity[i].z);


	return viscosity;

}

__device__ vec3d STForce(int i, int j,float r, vec3d* points, float* mass, float* density, vec3d* normal,int type, float st_const,float rho_0,float ST_Kernel)
{
	vec3d st;
	float tmp;

	if (type == 0) {
		vec3d cohesion;
		vec3d curvature;

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

		
	}
	else if (type == 1) {

		tmp = -st_const * mass[i] * mass[j] * ST_Kernel / r;

		st.x = tmp * (points[i].x - points[j].x);
		st.y = tmp * (points[i].y - points[j].y);
		st.z = tmp * (points[i].z - points[j].z);

	}
	
	return st;

}

__device__ vec3d PressureForce(int i, int j, float* pressure, float* mass, float* density, vec3d Spiky_Gradient) {

	vec3d p;

	float tmp;

	tmp = -mass[i] * mass[j] * (pressure[i] / powf(density[i], 2) + pressure[j] / powf(density[j], 2));

	
	p.x = tmp * Spiky_Gradient.x;
	p.y = tmp * Spiky_Gradient.y;
	p.z = tmp * Spiky_Gradient.z;

	return p;
}