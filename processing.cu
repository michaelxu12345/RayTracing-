#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//#include "./processing.cuh"
#include "color.cuh"
#include "ray.cuh"
#include "hittable.cuh"
#include "hittable_list.cuh""

// don't forget to manual-enforce floats by adding f to every decimal


__device__ color ray_color(const ray& r, hittable** world) {
	hit_record rec;
	if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
		vec3 N = rec.normal;
		return 0.5f * color(N.x() + 1, N.y() + 1, N.z() + 1);
	}

	vec3 unit_direction = unit_vector(r.direction());
	float a = 0.5f * (unit_direction.y() + 1.0f);
	return (1.0f - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0);
}

__global__ void processImageKernel(unsigned char* d_image, int width, int height,
	point3 camera_center, point3 pixel_upper_left, vec3 pixel_du, vec3 pixel_dv,
	hittable** world
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= width) || (y >= height)) {
		return;
	}

	int idx = (y * width + x) * 3;
	point3 pixel_center = pixel_upper_left + (x * pixel_du) + (y * pixel_dv);
	vec3 ray_direction = pixel_center - camera_center;
	ray r(camera_center, ray_direction);
	color pixel_color = ray_color(r, world);
	// write color
	write_color(d_image, pixel_color, x, y, width);

}



/*
* entry point for a function called from clicking a button
*
* 1. allocate memory on GPU
* 2. copy memory on CPU to GPU
* 3. launch kernel: call a function inputting a pointer on GPU
* 4. after GPU pointer is filled correctly, put stuff back onto CPU.
*
*/
void processImage(unsigned char* h_image, int width, int height, hittable** world) {
	/*processImageNotKernel(h_image, width, height);
	return;*/


	// this stuff will prob need to be moved and changed
	float aspect_ratio = float(width) / float(height);

	float focal_length = 1.0f;
	float viewport_height = 2.0f;
	float viewport_width = viewport_height * (float(width) / height);
	point3 camera_center = point3(0, 0, 0);

	vec3 viewport_u = vec3(viewport_width, 0, 0);
	vec3 viewport_v = vec3(0, -viewport_height, 0);

	vec3 pixel_du = viewport_u / width;
	vec3 pixel_dv = viewport_v / height;

	point3 viewport_upper_left = camera_center - vec3(0, 0, focal_length) -
		viewport_u / 2 - viewport_v / 2;
	point3 pixel_upper_left = viewport_upper_left + 0.5f * (pixel_du + pixel_dv);


	unsigned char* d_image;
	size_t imageSize = width * height * 3 * sizeof(unsigned char);

	// device memory
	cudaMalloc((void**)&d_image, imageSize);

	// copy image to device mem
	cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice);

	// block and grid sizes
	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	// launch kernel
	processImageKernel<<<gridSize, blockSize>>>(d_image, width, height, camera_center,
		pixel_upper_left, pixel_du, pixel_dv, world);

	// copy processed device to host
	cudaMemcpy(h_image, d_image, imageSize, cudaMemcpyDeviceToHost);

	cudaFree(d_image);
}