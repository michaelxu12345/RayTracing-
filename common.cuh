#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <curand_kernel.h>

const float infinity = 3.402823466e+38F;
const float pi = 3.1415926535897932385;

__device__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0f;
}


__device__ inline float random_double(curandState* rand_state) {
    // Returns a random real in [0,1).
    return curand_uniform(rand_state);
}

__device__ inline float random_double(float min, float max, curandState* rand_state) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_double(rand_state);
}

__host__ __device__ void checkError(const char* f, int linen) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel launch failed file %s, line %d: %s\n", f, linen,
            cudaGetErrorString(error));
    }
}
// Common Headers

#include "interval.cuh"
#include "color.cuh"
#include "ray.cuh"
#include "vec3.cuh"

