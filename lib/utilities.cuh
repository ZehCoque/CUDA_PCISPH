#pragma once
#include "common.cuh"
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

void writeTimeKeeper(char* main_path,float max_rho_err);

void getNewSimTime(char* main_path);

double dround(double val, int dp);

void displayProgress(std::chrono::high_resolution_clock::time_point start);

void rewritePVD(char* main_path);

int getLastIter(char* main_path);