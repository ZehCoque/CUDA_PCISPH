#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "common.cuh"
#include "helper.cuh"

__device__ vec3d ViscosityForce(int i, int j,float* mass, float* density,vec3d* velocity, int type, float visc_const, float Laplacian) {

	vec3d viscosity;

	if (type == 1) {
		viscosity.x = 0.f;
		viscosity.y = 0.f;
		viscosity.z = 0.f;
	}
	else {
		//mass_i / density_i * visc_const * mass_j / density_j * (v_j - v_i) * Laplacian;

		float tmp = mass[i] / density[i] * mass[j] / density[j] * visc_const * Laplacian;

		viscosity.x = tmp * (velocity[j].x - velocity[i].x);
		viscosity.y = tmp * (velocity[j].y - velocity[i].y);
		viscosity.z = tmp * (velocity[j].z - velocity[i].z);
	}

	return viscosity;

}

//__device__ vec3d ViscosityForce(int i, int j, float* mass, float* density, vec3d* points, vec3d* velocity, int type, float cs, float h,float r, float visc_const, vec3d Visc_Grad) {
//
//	vec3d viscosity;
//	float nu;
//
//	if (type == 0) {
//		nu = 2 * visc_const * h * cs / (density[i] + density[j]);
//	}
//	else {
//		nu = visc_const * h * cs / (density[i]);
//	}
//
//	vec3d delta_v;
//	delta_v.x = velocity[i].x - velocity[j].x;
//	delta_v.y = velocity[i].y - velocity[j].y;
//	delta_v.z = velocity[i].z - velocity[j].z;
//
//	vec3d delta_pos;
//	delta_pos.x = points[i].x - points[j].x;
//	delta_pos.y = points[i].y - points[j].y;
//	delta_pos.z = points[i].z - points[j].z;
//
//	float PI = nu * fmaxf(0, dot_product(delta_v, delta_pos)) / (r + 0.01 * h);
//
//	float tmp = - mass[i] * mass[j] * PI;
//
//	viscosity.x = tmp * Visc_Grad.x;
//	viscosity.y = tmp * Visc_Grad.y;
//	viscosity.z = tmp * Visc_Grad.z;
//
//	return viscosity;
//
//}

__device__ vec3d STForce(int i, int j,float r, vec3d* points, float* mass, float* density, vec3d* normal,int type, float st_const,float rho_0,float ST_Kernel)
{
	vec3d st;

	if (type == 0) {
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
	else if (type == 1) {

		vec3d adhesion;

		float tmp = -st_const * mass[i] * mass[j] * ST_Kernel / r;

		adhesion.x = tmp * (points[i].x - points[j].x);
		adhesion.y = tmp * (points[i].y - points[j].y);
		adhesion.z = tmp * (points[i].z - points[j].z);
		//printf("%g %g %g\n", adhesion.x, adhesion.y, adhesion.z);
		return adhesion;
	}
	
	

}

__device__ vec3d PressureForce(int i, int j, float* pressure, float* mass, float* density, int type, vec3d Spiky_Gradient) {

	vec3d p;

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