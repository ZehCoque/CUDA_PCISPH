#pragma once
#include "device_functions.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "common.cuh"

//IMPORTANT NOTE: In the "main_function.cu" script, the initial and final positions of boundary and fluids particles are tweaked.
// The user is defining the entire size of the prims and the box containing the fluid, but these positions must be corrected with the particle radii since the particles are ploted according to their central position.
// See this link for a visual explanation -> https://media.giphy.com/media/lSDYIhgp7bO56xRCvZ/giphy.gif

// This kernel makes a completely full prism with particles inside separated by the particle radii on the X and Z axis. 
// For the Y axis, the distance is calculated using the height of an equilateral triangle (check this link https://i.imgur.com/EElCDGP.png for a visual explanation and this link https://i.imgur.com/ys0Abpn.jpg for an example).
// This is called hexagonal packing and it reduces the empty space between particles.
__global__ void makePrism(float3* position_arr, const float diameter, const float3 initial_pos, const int NPD[], const int size) {

	int index = getGlobalIdx_1D_1D();
	
	int i = index % NPD[0];
	int j = (index / NPD[0]) % NPD[1];
	int k = index / (NPD[0]*NPD[1]);

	if (index >= size) {
		return;
	}

	// Hexagonal packing
	if (j % 2 == 0) {
		position_arr[index].x = initial_pos.x + i * diameter;
		position_arr[index].y = initial_pos.y + j * diameter * powf(3.f, 1.f / 2.f) / 2.f;
		position_arr[index].z = initial_pos.z + k * diameter;
	}
	else {
		position_arr[index].x = initial_pos.x + i * diameter + diameter / 2.f;
		position_arr[index].y = initial_pos.y + j * diameter * powf(3.f, 1.f / 2.f) / 2.f;
		position_arr[index].z = initial_pos.z + k * diameter + diameter / 2.f;
	}

	return;
};

// This kernel makes a plane with the indicated initial, final and orientation
__global__ void makePlane(float3* position_arr, const float diameter,const float3 initial_pos,const int offset,const int orientation,const int size,const int NPD[]) {

	int index = getGlobalIdx_1D_1D();

	if (index >= size) {
		return;
	}

	// 1 for xy orientation
	// 2 for xz orientation
	// 3 for yz orientation

	if (orientation == 1) {
		int i = index % NPD[0];
		int j = (index / NPD[0]) % NPD[1];
		index = index + offset;
		position_arr[index].x = initial_pos.x + i * diameter;
		position_arr[index].y = initial_pos.y + j * diameter;
		position_arr[index].z = initial_pos.z;
	}
	else if (orientation == 2) {
		int i = index % NPD[0];
		int j = (index / NPD[0]) % NPD[2];
		index = index + offset;
		position_arr[index].x = initial_pos.x + i * diameter;
		position_arr[index].y = initial_pos.y;
		position_arr[index].z = initial_pos.z + j * diameter;
	}
	else if (orientation == 3) {
		int i = index % (NPD[1]);
		int j = (index / (NPD[1])) % (NPD[2]);
		index = index + offset;
		position_arr[index].x = initial_pos.x;
		position_arr[index].y = initial_pos.y + i * diameter;
		position_arr[index].z = initial_pos.z + j * diameter;
	}
	return;
}

// This is a function that uses the makePlane kernel to make an empty box with walls with 1 particle of thickness
void makeBox(float3* position_arr, float diameter, float3 initial_pos, float3 final_pos, int block_size, int* D_NPD,int* NPD, int SIMULATION_DIMENSION) {
	int offset = 0;
	
	int num_x, num_y, num_z, size,grid_size;
	int _NPD[3];
	//first wall
	num_x = NPD[0];
	num_y = NPD[1];
	num_z = 1;

	_NPD[0] = num_x;
	_NPD[1] = num_y;
	_NPD[2] = num_z;
	gpuErrchk(cudaMemcpy(D_NPD, _NPD, SIMULATION_DIMENSION * sizeof(float), cudaMemcpyHostToDevice));

	size = num_x * num_y * num_z;
	grid_size = (num_x * num_y * num_z) / block_size + 1;
	
	makePlane << <grid_size, block_size >> > (position_arr, diameter, initial_pos, offset, 1, size,D_NPD);

	offset = offset + num_x * num_y * num_z;

	//second wall
	float3 tmp_initial_pos;
	tmp_initial_pos.x = initial_pos.x;
	tmp_initial_pos.y = initial_pos.y;
	tmp_initial_pos.z = final_pos.z;

	size = num_x * num_y * num_z;
	grid_size = (num_x * num_y * num_z) / block_size + 1;

	makePlane << <grid_size, block_size >> > (position_arr, diameter, tmp_initial_pos, offset, 1, size,D_NPD);

	offset = offset + num_x * num_y * num_z;

	//third wall
	tmp_initial_pos.x = initial_pos.x;
	tmp_initial_pos.y = initial_pos.y;
	tmp_initial_pos.z = initial_pos.z + diameter;

	num_x = NPD[0];
	num_y = 1;
	num_z = NPD[2] - 2;

	_NPD[0] = num_x;
	_NPD[1] = num_y;
	_NPD[2] = num_z;
	gpuErrchk(cudaMemcpy(D_NPD, _NPD, SIMULATION_DIMENSION * sizeof(float), cudaMemcpyHostToDevice));

	size = num_x * num_y * num_z;
	grid_size = (num_x * num_y * num_z) / block_size + 1;

	makePlane << <grid_size, block_size >> > (position_arr, diameter, tmp_initial_pos, offset, 2, size,D_NPD);

	offset = offset + num_x * num_y * num_z;

	//forth wall
	tmp_initial_pos.x = initial_pos.x;
	tmp_initial_pos.y = final_pos.y;
	tmp_initial_pos.z = initial_pos.z + diameter;

	size = num_x * num_y * num_z;
	grid_size = (num_x * num_y * num_z / block_size) + 1;

	makePlane << <grid_size, block_size >> > (position_arr, diameter, tmp_initial_pos, offset, 2, size,D_NPD);

	offset = offset + num_x * num_y * num_z;

	//fifth wall
	num_x = 1;
	num_y = NPD[1] - 2;
	num_z = NPD[2] - 2;

	_NPD[0] = num_x;
	_NPD[1] = num_y;
	_NPD[2] = num_z;
	gpuErrchk(cudaMemcpy(D_NPD, _NPD, SIMULATION_DIMENSION * sizeof(float), cudaMemcpyHostToDevice));

	tmp_initial_pos.x = initial_pos.x;
	tmp_initial_pos.y = initial_pos.y + diameter;
	tmp_initial_pos.z = initial_pos.z + diameter;

	size = num_x * num_y * num_z;

	grid_size = (num_x * num_y * num_z / block_size) + 1;

	makePlane << <grid_size, block_size >> > (position_arr, diameter, tmp_initial_pos, offset, 3, size,D_NPD);

	offset = offset + num_x * num_y * num_z;

	//sixth wall
	tmp_initial_pos.x = final_pos.x;
	tmp_initial_pos.y = initial_pos.y + diameter;
	tmp_initial_pos.z = initial_pos.z + diameter;

	size = num_x * num_y * num_z;
	grid_size = (num_x * num_y * num_z) / block_size + 1;

	makePlane << <grid_size, block_size >> > (position_arr, diameter, tmp_initial_pos, offset, 3, size,D_NPD);

	return;
}
