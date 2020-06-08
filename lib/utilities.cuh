#pragma once
#include "common.cuh"
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "global_variables.cuh"
#include <conio.h>
//struct stat info;
int fileExists(const char* const path);

int dirExists(const char* const path);

void CreateDir(char* path);

int count_lines(char path[]);

int extractIntegers(char* str);

char* getMainPath(char* main_path);

char* clearAddressArray(char* buffer, char* s1, char* s2);

double dround(double val, int dp);

uint unsignedIntPow(int x, int p);

void write_txt_file(const char* main_path, const char* filename, uint* array1, uint* array2, int size);

void write_txt_file_float3(const char* main_path, const char* filename, float3* array1, float3* array2, int size);