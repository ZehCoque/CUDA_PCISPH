#pragma once
#include "common.cuh"

void VTU_Writer(char path[], int iteration, float3* points, int numberOfPoints, float** pointData[], float3** vectorData[], std::string pointDataNames[], std::string vectorDataNames[], int size_pointData, int size_vectorData, char* fullpath, int type = 0);

void VTK_Group(char vtk_group_path[], char vtu_path[], float time);

void readVTU(char* iter_path, float3* position, float3* velocity);
