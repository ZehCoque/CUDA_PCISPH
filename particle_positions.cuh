#pragma once
#include "device_functions.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "common.cuh"

__global__ void makePrism(vec3d* position_arr, const float diameter, const vec3d initial_pos, const int NPD[], const int size) {

	int index = getGlobalIdx_1D_1D();
	
	int i = index % NPD[0];
	int j = (index / NPD[0]) % NPD[1];
	int k = index / (NPD[0]*NPD[1]);

	if (index > size) {
		return;
	}

	// Hexagonal packing
	if (j % 2 == 0) {
		position_arr[index].x = initial_pos.x + i * diameter;
		position_arr[index].y = initial_pos.y + j * diameter * powf(3, 1 / 2.f) / 2.f;
		position_arr[index].z = initial_pos.z + k * diameter;
	}
	else {
		position_arr[index].x = initial_pos.x + i * diameter + diameter / 2.f;
		position_arr[index].y = initial_pos.y + j * diameter * powf(3.f, 1 / 2.f) / 2.f;
		position_arr[index].z = initial_pos.z + k * diameter + diameter / 2.f;
	}

	return;
};

__global__ void makePlane(vec3d* position_arr, const float diameter,const vec3d initial_pos,const int offset,const int orientation,const int size,const int NPD[]) {

	int index = getGlobalIdx_1D_1D() + offset;

	int i = index % NPD[0];
	int j = (index / NPD[0]) % NPD[1];
	int k = index / (NPD[0] * NPD[1]);

	if (index >= size) {
		return;
	}

	//printf("%d\n", index);

	

	// 1 for xy orientation
	// 2 for xz orientation
	// 3 for yz orientation

	if (orientation == 1) {
		position_arr[index].x = initial_pos.x + i * diameter;
		position_arr[index].y = initial_pos.y + j * diameter;
		position_arr[index].z = initial_pos.z;
	}
	else if (orientation == 2) {
		printf("%g %g %g\n", initial_pos.x + i * diameter, initial_pos.y, initial_pos.z + k * diameter);
		position_arr[index].x = initial_pos.x + i * diameter;
		position_arr[index].y = initial_pos.y;
		position_arr[index].z = initial_pos.z + k * diameter;
	}
	else if (orientation == 3) {
		position_arr[index].x = initial_pos.x;
		position_arr[index].y = initial_pos.y + j * diameter;
		position_arr[index].z = initial_pos.z + k * diameter;
	}
	return;
}

void makeBox(vec3d* position_arr, float diameter, vec3d initial_pos, vec3d final_pos, int block_size, int D_NPD[]) {
	int offset = 0;
	
	int num_x, num_y, num_z, size,grid_size;


	//first wall
	num_x = static_cast<int>(ceil((final_pos.x - initial_pos.x) / diameter)) + 1;
	num_y = static_cast<int>(ceil((final_pos.y - initial_pos.y) / diameter)) + 1;
	num_z = 1;

	size = offset + num_x * num_y * num_z;
	grid_size = (num_x * num_y * num_z) / block_size + 1;
	
	makePlane << <grid_size, block_size >> > (position_arr, diameter, initial_pos, offset, 1, size,D_NPD);

	offset = offset + num_x * num_y * num_z;
	printf("%d\n",offset);
	//second wall
	vec3d tmp_initial_pos;
	tmp_initial_pos.x = initial_pos.x;
	tmp_initial_pos.y = initial_pos.y;
	tmp_initial_pos.z = final_pos.z;

	size = offset + num_x * num_y * num_z;
	grid_size = (num_x * num_y * num_z) / block_size + 1;

	makePlane << <grid_size, block_size >> > (position_arr, diameter, tmp_initial_pos, offset, 1, size,D_NPD);

	offset = offset + num_x * num_y * num_z;
	printf("%d\n",offset);
	//third wall
	tmp_initial_pos.x = initial_pos.x;
	tmp_initial_pos.y = initial_pos.y;
	tmp_initial_pos.z = initial_pos.z + diameter;

	num_x = static_cast<int>(ceil((final_pos.x - initial_pos.x) / diameter)) + 1;
	num_y = 1;
	num_z = static_cast<int>(ceil((final_pos.z - initial_pos.z) / diameter)) - 1;

	size = offset + num_x * num_y * num_z;
	grid_size = (num_x * num_y * num_z) / block_size + 1;

	makePlane << <grid_size, block_size >> > (position_arr, diameter, tmp_initial_pos, offset, 2, size,D_NPD);

	offset = offset + num_x * num_y * num_z;
	printf("%d\n",offset);
	////forth wall
	//tmp_initial_pos.x = initial_pos.x;
	//tmp_initial_pos.y = final_pos.y;
	//tmp_initial_pos.z = initial_pos.z + diameter;

	//size = offset + num_x * num_y * num_z;
	//grid_size = (num_x * num_y * num_z / block_size) + 1;

	//makePlane << <grid_size, block_size >> > (position_arr, diameter, tmp_initial_pos, offset, 2, size,D_NPD);

	//offset = offset + num_x * num_y * num_z;
	//printf("%d\n",offset);
	////fifth wall
	//num_x = 1;
	//num_y = static_cast<int>(ceil((final_pos.y - initial_pos.y) / diameter)) - 1;
	//num_z = static_cast<int>(ceil((final_pos.z - initial_pos.z) / diameter)) - 1;

	//tmp_initial_pos.x = initial_pos.x;
	//tmp_initial_pos.y = initial_pos.y + diameter;
	//tmp_initial_pos.z = initial_pos.z + diameter;

	//size = offset + num_x * num_y * num_z;

	//grid_size = (num_x * num_y * num_z / block_size) + 1;

	//makePlane << <grid_size, block_size >> > (position_arr, diameter, tmp_initial_pos, offset, 3, size,D_NPD);

	//offset = offset + num_x * num_y * num_z;
	//printf("%d\n",offset);
	////sixth wall

	//tmp_initial_pos.x = final_pos.x;
	//tmp_initial_pos.y = initial_pos.y + diameter;
	//tmp_initial_pos.z = initial_pos.z + diameter;

	//size = offset + num_x * num_y * num_z;
	//grid_size = (num_x * num_y * num_z) / block_size + 1;

	//makePlane << <grid_size, block_size >> > (position_arr, diameter, tmp_initial_pos, offset, 3, size,D_NPD);

	return;
}
