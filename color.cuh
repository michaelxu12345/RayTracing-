#pragma once
#include "vec3.cuh"
#include "interval.cuh"

using color = vec3;

__device__ inline float linear_to_gamma(float linear_component) {
	return sqrtf(linear_component);
}

__device__ void write_color(unsigned char* d_image, color pixel_color, int x, int y, int width) {
	auto r = pixel_color.x();
	auto g = pixel_color.y();
	auto b = pixel_color.z();

	interval intensity(0.000, 0.999);
	int rbyte = int(255.999f * intensity.clamp(r));
	int gbyte = int(255.999f * intensity.clamp(g));
	int bbyte = int(255.999f * intensity.clamp(b));

	int idx = (y * width + x) * 3;

	d_image[idx] = rbyte;
	d_image[idx + 1] = gbyte;
	d_image[idx + 2] = bbyte;
}