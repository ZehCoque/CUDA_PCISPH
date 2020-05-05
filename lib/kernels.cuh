#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ float Poly6_Kernel(float r, float h, float invh)
{
	const float invh9 = powf(invh, 9);
	const float tmp = powf(h, 2) - powf(r, 2);
	
	return 1.5666814710f * invh9 * tmp * tmp * tmp;
}

__host__ __device__ vec3d Poly6_Gradient(int i, int j,vec3d* points, float r, float h, float invh) {

	vec3d poly6_grad;

	float tmp = 9.4000888263f * powf(invh, 9) * powf(powf(h,2)-powf(r,2),2);

	poly6_grad.x = tmp * (points[i].x - points[j].x);
	poly6_grad.y = tmp * (points[i].y - points[j].y);
	poly6_grad.z = tmp * (points[i].z - points[j].z);

	return poly6_grad;

}

__device__ vec3d Viscosity_Gradient(int i, int j, vec3d* points, float r, float h, float invh)
{

	vec3d visc_grad;

	float invr = 1 / r;
	
	float tmp1 = 2.38732414637f * powf(invh, 3);
	float tmp2 = -1.5f * r * powf(invh, 3) + 2 * powf(invh, 2) - 0.5f * h * powf(invr, 3);

	visc_grad.x = tmp1 * (points[i].x - points[j].x) * tmp2;
	visc_grad.y = tmp1 * (points[i].y - points[j].y) * tmp2;
	visc_grad.z = tmp1 * (points[i].z - points[j].z) * tmp2;

	return visc_grad;
}

__device__ float Viscosity_Laplacian(float r, float h, float invh)
{
	return 14.32394487f * powf(invh,6) * (h-r);
}

__device__ float ST_Kernel(float r, float h,float invh, int type)
{
	if (type == 0) {
		float tmp = 10.1859163578f * powf(invh, 9);

		if (2.f * r > h && r <= h) {
			return tmp * (h - r) * (h - r) * (h - r) * r * r * r;
		}
		else if (r > 0.f && 2.f * r <= h) {
			return tmp * (2.f * (h - r) * (h - r) * (h - r) * r * r * r - powf(h, 6.f) * 0.015625f);
		}
		else {
			return 0.f;
		}
	}
	else if (type == 1) {
		if (2.f * r > h || r <= h) {
			return 0.007f * powf(invh, 3.25f) * powf(-4.f * powf(r, 2.f) * invh + 6.f * r - 2.f * h, 0.25f);
		}
		else {
			return 0.f;
		}
	}


}

__host__ __device__ vec3d Spiky_Gradient(int i, int j, vec3d* points, float r, float h, float invh) {

	vec3d spiky;

	float tmp = -14.323944878f * powf(invh, 6) * powf(h - r, 2);

	spiky.x = tmp * (points[i].x - points[j].x);
	spiky.y = tmp * (points[i].y - points[j].y);
	spiky.z = tmp * (points[i].z - points[j].z);

	return spiky;
}
