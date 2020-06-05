#pragma once
#include <device_launch_parameters.h>
#include "common.cuh"

__device__ float Poly6_Kernel(float *r)
{
	const float invh9 = powf(d_params.invh, 9);
	const float tmp = powf(d_params.h, 2) - powf(*r, 2);
	
	return 1.5666814710f * invh9 * tmp * tmp * tmp;
}

__host__ __device__ float3 Poly6_Gradient(float3* position_i, float3* position_j, float *r, float* invh, float* h) {

	float3 poly6_grad;

	float tmp = 9.4000888263f * powf(*invh, 9) * powf(powf(*h,2)-powf(*r,2),2);

	poly6_grad =  make_float3(tmp * (position_i->x - position_j->x), 
		tmp * (position_i->y - position_j->y), 
		tmp * (position_i->z - position_j->z));

	return poly6_grad;

}

__device__ float3 Viscosity_Gradient(float3* position_i, float3* position_j, float* r)
{

	float3 visc_grad;

	float invr = 1 / *r;
	
	float tmp1 = 2.38732414637f * powf(d_params.invh, 3);
	float tmp2 = -1.5f * *r * powf(d_params.invh, 3) + 2 * powf(d_params.invh, 2) - 0.5f * d_params.h * powf(invr, 3);

	visc_grad = make_float3(tmp1 * tmp2 * (position_i->x - position_j->x), 
		tmp1 * tmp2 * (position_i->y - position_j->y), 
		tmp1 * tmp2 * (position_i->z - position_j->z));

	return visc_grad;
}

__device__ float Viscosity_Laplacian(float *r)
{
	return 14.32394487f * powf(d_params.invh,6) * (d_params.h-*r);
}

__device__ float ST_Kernel(float *r, int *type)
{
	if (*type == 0) {
		float tmp = 10.1859163578f * powf(d_params.invh, 9);

		if (2.f * *r > d_params.h && *r <= d_params.h) {
			return tmp * (d_params.h - *r) * (d_params.h - *r) * (d_params.h - *r) * *r * *r * *r;
		}
		else if (*r > 0.f && 2.f * *r <= d_params.h) {
			return tmp * (2.f * (d_params.h - *r) * (d_params.h - *r) * (d_params.h - *r) * *r * *r * *r - powf(d_params.h, 6.f) * 0.015625f);
		}
		else {
			return 0.f;
		}
	}
	else if (*type == 1) {
		if (2.f * *r > d_params.h && *r <= d_params.h) {
			return 0.007f * powf(d_params.invh, 3.25f) * powf(-4.f * powf(*r, 2.f) * d_params.invh + 6.f * *r - 2.f * d_params.h, 0.25f);
		}
		else {
			return 0.f;
		}

	}

	return 0.f;
}

__host__ __device__ float3 Spiky_Gradient(float3* position_i, float3* position_j, float *r, float* invh, float* h) {

	float3 spiky;

	float tmp = -14.323944878f * powf(*invh, 6) * powf(*h - *r, 2);

	spiky =  make_float3(tmp * (position_i->x - position_j->x), 
		tmp * (position_i->y - position_j->y), 
		tmp * (position_i->z - position_j->z));

	return spiky;
}
