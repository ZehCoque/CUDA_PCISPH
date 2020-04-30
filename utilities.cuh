#pragma once
#include "common.cuh"
#include <cuda_runtime.h>
//struct stat info;
int fileExists(const char* const path);

int dirExists(const char* const path);

void CreateDir(char* path);

int count_lines(char path[]);

int extractIntegers(char* str);

char* getMainPath(char* main_path);

char* clearAddressArray(char* buffer, char* s1, char* s2);

void writeTimeKeeper(char* main_path, float simulation_time, int iteration);

void getNewSimTime(char* main_path, float *simulation_time, int iteration);