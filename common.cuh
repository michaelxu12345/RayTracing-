#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

const float infinity = 3.402823466e+38F;
const float pi = 3.1415926535897932385;

__device__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0f;
}


inline double random_double() {
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_double();
}

// Common Headers

#include "interval.cuh"
#include "color.cuh"
#include "ray.cuh"
#include "vec3.cuh"

