#pragma once
#include <cstring>
#include <time.h>
#include <math.h>
#include "../../general.h"

//float INFINITY = -numeric_limits<float>::infinity();

vec3 getDims(vec3 a_dim, size_t k_size, size_t padding, size_t stride, size_t out_depth);


float* convolve(vec3 a_dim, float* a, vec3 k_dim, int out_depth, float* kernel, int padding, int stride);

float* convolve_volume(vec3 a_dim, float* a, vec3 k_dim, size_t out_depth, float* kernel, size_t padding, size_t stride);

void convolve_volume(vec3 a_dim, vector<float>& a, vec3 k_dim, vector<vector<vector<vector<float>>>>& kernel, vec3 out_dim, vector<float>& out, size_t padding, size_t stride);

void backVolve(vec3 a_dim, vector<float>& terms, vec3 k_dim, vector<vector<vector<vector<float>>>>& kernel, vec3 out_dim, vector<float>& for_terms, size_t padding, size_t stride);

void gradVolve(vec3 a_dim, vector<float>& in_layer, vec3 k_dim, vector<vector<vector<vector<float>>>>& gradient, vec3 out_dim, vector<float>& out_terms, size_t padding, size_t stride);

float* max_pool(float* a, vec3 a_dim, size_t pool_size, size_t padding, size_t stride);

void max_pool(vec3 a_dim, vector<float>& a, vec3 out_dim, vector<float>& out, size_t pool_size, size_t padding, size_t stride);

void backPool(vec3 a_dim, vector<float>& a, vector<float>& terms, vec3 out_dim, vector<float>& for_terms, size_t pool_size, size_t padding, size_t stride);