#pragma once
#include "common.cuh"
#include "hittable.cuh"
#include "camera.cuh"


__device__ color ray_color(const ray& r, hittable** world) {
	hit_record rec;
	if ((*world)->hit(r, interval(0.0, 3.402823466e+38F), rec)) {
		return 0.5 * (rec.normal + color(1, 1, 1));
	}

	vec3 unit_direction = unit_vector(r.direction());
	float a = 0.5f * (unit_direction.y() + 1.0f);
	return (1.0f - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0);
}

__global__ void processImageKernel(unsigned char* d_image,
	camera cam, 
	hittable** world
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= cam.image_width) || (y >= cam.image_height)) {
		return;
	}

	//int idx = (y * width + x) * 3;
	point3 pixel_center = cam.pixel_upper_left + (x * cam.pixel_du) + (y * cam.pixel_dv);
	vec3 ray_direction = pixel_center - cam.center;
	ray r(cam.center, ray_direction);
	color pixel_color = ray_color(r, world);
	// write color
	write_color(d_image, pixel_color, x, y, cam.image_width);

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
void processImage(unsigned char* h_image, int image_width, int image_height, hittable** world, camera& cam) {
	

	unsigned char* d_image;
	size_t imageSize = image_width * image_height * 3 * sizeof(unsigned char);

	// device memory
	cudaMalloc((void**)&d_image, imageSize);

	// copy image to device mem
	cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice);

	// block and grid sizes
	dim3 blockSize(16, 16);
	dim3 gridSize((image_width + blockSize.x - 1) / blockSize.x, (image_height + blockSize.y - 1) / blockSize.y);

	// launch kernel
	processImageKernel << <gridSize, blockSize >> > (d_image, cam, world);

	// copy processed device to host
	cudaMemcpy(h_image, d_image, imageSize, cudaMemcpyDeviceToHost);

	cudaFree(d_image);
}