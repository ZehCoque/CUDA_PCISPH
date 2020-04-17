#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ float Poly6_Kernel(float r, float h, float invh)
{
	const float invh9 = powf(invh, 9);
	const float tmp = powf(h, 2) - powf(r, 2);
	
	return 1.5666814710f * invh9 * tmp * tmp * tmp;
}

__device__ vec3d Poly6_Gradient(int i, int j,vec3d* points, float r, float h, float invh) {

	vec3d poly6_grad;

	float tmp = 9.4000888263f * powf(invh, 9) * powf(powf(h,2)-powf(r,2),2);

	poly6_grad.x = tmp * (points[i].x - points[j].x);
	poly6_grad.y = tmp * (points[i].y - points[j].y);
	poly6_grad.z = tmp * (points[i].z - points[j].z);

	return poly6_grad;

}

__device__ vec3d Viscosity_Gradient(vec3d point, float r, float h, float invh)
{

	vec3d visc_grad;

	float invr = 1 / r;

	float tmp = -1.5f * r * invh * invh * invh + 2 * invh * invh - 0.5f * h * invr * invr * invr;

	visc_grad.x = 2.387324146f * invh * invh * invh * point.x * tmp;
	visc_grad.y = 2.387324146f * invh * invh * invh * point.y * tmp;
	visc_grad.z = 2.387324146f * invh * invh * invh * point.z * tmp;

	return visc_grad;
}

__device__ float Viscosity_Laplacian(float r, float h, float invh)
{
	return 14.32394487f * powf(invh,6) * (h-r);
}

__device__ float ST_Kernel(float r, float h,float invh)
{
	float tmp = 10.1859163578f * powf(invh, 9);

	if (2.f * r > h && r <= h) {
		return tmp * (h - r) * (h - r) * (h - r) * r * r * r;
	}
	else if (r > 0.f && 2.f * r <= h) {
		return tmp * (2.f * (h - r) * (h - r) * (h - r) * r * r * r - powf(h,6) * 0.015625f);
	}
	else {
		return 0.f;
	}

}
