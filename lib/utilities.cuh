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