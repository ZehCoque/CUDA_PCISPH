#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "common.cuh"
#include "helper.cuh"

__device__ float3 ViscosityForce(int i, int j,float* mass, float* density,float3* velocity,int type, float visc_const,float rho_0, float Laplacian) {

	float3 viscosity;

	if (type == 1) {
		float tmp = mass[i] / density[i] * mass[j] / rho_0 * visc_const * Laplacian;

		viscosity.x = tmp * (velocity[j].x - velocity[i].x);
		viscosity.y = tmp * (velocity[j].y - velocity[i].y);
		viscosity.z = tmp * (velocity[j].z - velocity[i].z);

	}
	else {

		float tmp = mass[i] / density[i] * mass[j] / density[j] * visc_const * Laplacian;

		viscosity.x = tmp * (velocity[j].x - velocity[i].x);
		viscosity.y = tmp * (velocity[j].y - velocity[i].y);
		viscosity.z = tmp * (velocity[j].z - velocity[i].z);
	}

	return viscosity;

}

__device__ float3 STForce(int i, int j,float r, float3* points, float* mass, float* density, float3* normal,int type, float st_const,float rho_0,float ST_Kernel)
{
	float3 st;
	float tmp;

	if (type == 0) {
		float3 cohesion;
		float3 curvature;

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

		//st.x = 0.f;
		//st.y = 0.f;
		//st.z = 0.f;

	}
	
	return st;

}

__device__ float3 PressureForce(int i, int j, float* pressure, float* mass, float* density, int type, float3 Spiky_Gradient) {

	float3 p;

	float tmp;

	if (type == 0) {
		tmp = -mass[i] * mass[j] * (pressure[i] / powf(density[i], 2) + pressure[j] / powf(density[j], 2));
	}
	else {
		tmp = -mass[i] * mass[j] * (pressure[i] / powf(density[i], 2));
	}

	p.x = tmp * Spiky_Gradient.x;
	p.y = tmp * Spiky_Gradient.y;
	p.z = tmp * Spiky_Gradient.z;

	return p;
}