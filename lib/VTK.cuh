#pragma once
#include "common.cuh"

void VTU_Writer(char path[], int iteration, vec3d* points, int numberOfPoints, float** pointData[], vec3d** vectorData[], std::string pointDataNames[], std::string vectorDataNames[], int size_pointData, int size_vectorData, char* fullpath, int type = 0);

void VTK_Group(char vtk_group_path[], char vtu_path[], float time);

void readVTU(char* iter_path, vec3d* position, vec3d* velocity);