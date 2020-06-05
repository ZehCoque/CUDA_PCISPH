#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "common.cuh"

__device__ float distance(float3 point1, float3 point2) {
	//printf("point1 = [%g %g %g]|point2 = [%g %g %g]|point1 - point2 = [%g %g %g]|norm3df = %g\n", point1.x, point1.y, point1.z, point2.x, point2.y, point2.z, point1.x - point2.x, point1.y - point2.y, point1.z - point2.z, norm3df(point1.x - point2.x, point1.y - point2.y, point1.z - point2.z));
	return norm3df(point1.x - point2.x, point1.y - point2.y, point1.z - point2.z);
}

__device__ char* device_strcpy(char* dest, const char* src) {
	int i = 0;
	do {
		dest[i] = src[i];
	} while (src[i++] != 0);
	return dest;
}

__device__ char* device_strcat(char* dest, const char* src) {
	int i = 0;
	while (dest[i] != 0) i++;
	device_strcpy(dest + i, src);
	return dest;
}

__host__ __device__ float dot_product(float3 vec1, float3 vec2) {

	return vec1.x * vec2.x + vec1.y + vec2.y + vec1.z * vec2.z;

}

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
	float old;
	old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
		__uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

	return old;
}

__device__ float maxValueInVec3D(float3 vec) {

	return fmaxf(fabs(vec.x), fmaxf(fabs(vec.y), fabs(vec.z)));

}

__device__ __forceinline__ float atomicAddFloat(float* address, float val)
{
	// Doing it all as longlongs cuts one __longlong_as_double from the inner loop
	unsigned int* ptr = (unsigned int*)address;
	unsigned int old, newint, ret = *ptr;
	do {
		old = ret;
		newint = __float_as_int(__int_as_float(old) + val);
	} while ((ret = atomicCAS(ptr, old, newint)) != old);

	return __int_as_float(ret);
}