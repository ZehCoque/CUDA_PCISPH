#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "common.cuh"

__device__ float distance(vec3d point1, vec3d point2) {
	//printf("point1 = [%g %g %g]|point2 = [%g %g %g]|point1 - point2 = [%g %g %g]|norm3df = %g\n", point1.x, point1.y, point1.z, point2.x, point2.y, point2.z, point1.x - point2.x, point1.y - point2.y, point1.z - point2.z, norm3df(point1.x - point2.x, point1.y - point2.y, point1.z - point2.z));
	return norm3df(point1.x - point2.x, point1.y - point2.y, point1.z - point2.z);
}

__device__ void assignToVec3d(vec3d* point, float x = 0.f, float y = 0.f, float z = 0.f ) {

	point->x = x;
	point->y = y;
	point->z = z;

	return;
}

__device__ void sumToVec3d(vec3d* point, float x = 0.f, float y = 0.f, float z = 0.f) {

	point->x += x;
	point->y += y;
	point->z += z;

	return;
}

__device__ void sum2Vec3d(vec3d* vec1, vec3d* vec2) {

	vec1->x += vec2->x;
	vec1->y += vec2->y;
	vec1->z += vec2->z;

	return;
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

