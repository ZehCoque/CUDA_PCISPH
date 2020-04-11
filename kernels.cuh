#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ float distance(vec3d point1, vec3d point2) {
	//printf("point1 = [%g %g %g]|point2 = [%g %g %g]|point1 - point2 = [%g %g %g]|norm3df = %g\n", point1.x, point1.y, point1.z, point2.x, point2.y, point2.z, point1.x - point2.x, point1.y - point2.y, point1.z - point2.z, norm3df(point1.x - point2.x, point1.y - point2.y, point1.z - point2.z));
	return norm3df(point1.x - point2.x, point1.y- point2.y, point1.z - point2.z);
}

__device__ float Poly6_Kernel(float r, float h)
{
	const float invh = 1 / h;
	const float invh9 = powf(invh, 9);
	const float tmp = powf(h, 2) - powf(r, 2);
	
	return 1.5666814710f * invh9 * tmp * tmp * tmp;
}
