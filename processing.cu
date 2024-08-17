#pragma once
#include "common.cuh"
#include "hittable.cuh"
#include "camera.cuh"
#include "material.cuh"
#include <curand_kernel.h>


__device__ color ray_color(const ray& r, hittable** world, curandState* rand_state) {
	

	ray my_ray = r;
	color my_color(1.0, 1.0, 1.0);
	for (int depth = 0; depth < 10; depth++) {
		hit_record rec;
		if ((*world)->hit(my_ray, interval(0.001f, FLT_MAX), rec)) {
			ray scattered = my_ray;
			color attenuation(1.0, 1.0, 1.0);

			if ( rec.mat->scatter(my_ray, rec, attenuation, scattered, rand_state)) {
				//printf("rec stats: %d %d %d\n", rec.front_face, rec.mat, rec.)
				my_ray = scattered;
				my_color = my_color * attenuation;
			}
			else {
				return color(0.0, 0.0, 0.0);
			}
		}
		else {
			vec3 unit_direction = unit_vector(my_ray.direction());
			float a = 0.5f * (unit_direction.y() + 1.0f);
			color sky = (1.0f - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0);
			return my_color * sky;
		}
	}
	/*if ((*world)->hit(r, interval(0.0, 3.402823466e+38F), rec)) {
		return 0.5 * (rec.normal + color(1, 1, 1));
	}*/

	return color(0, 0, 0);

	/*vec3 unit_direction = unit_vector(r.direction());
	float a = 0.5f * (unit_direction.y() + 1.0f);
	return (1.0f - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0);*/
}

__global__ void processImageKernel(unsigned char* d_image,
	camera cam,
	hittable** world,
	curandState* rand_state
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= cam.image_width) || (y >= cam.image_height)) {
		return;
	}

	int idx = (y * cam.image_width + x);
	curandState local_rand_state = rand_state[idx];

	color pixel_color = color(0, 0, 0);

	if (cam.num_samples == 1) {
		point3 pixel_center = cam.pixel_upper_left + (x * cam.pixel_du) + (y * cam.pixel_dv);
		vec3 ray_direction = pixel_center - cam.center;
		ray r(cam.center, ray_direction);
		ray myray = cam.get_ray(x, y);
		pixel_color = ray_color(r, world, &local_rand_state);
	}
	else {
		for (int samp = 0; samp < cam.num_samples; samp++) {
			float u = float(x + curand_uniform(&local_rand_state)-0.5);// float(cam.image_width);
			float v = float(y + curand_uniform(&local_rand_state)-0.5);// float(cam.image_height);
			pixel_color += ray_color(cam.get_ray(u, v), world, &local_rand_state);
		}
	}
	
	// write color
	write_color(d_image, pixel_color / cam.num_samples, x, y, cam.image_width);

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
void processImage(unsigned char* h_image,  hittable** world, camera& cam,
	curandState* d_rand_state
	) {
	
	int image_width = cam.image_width;
	int image_height = cam.image_height;
	unsigned char* d_image;
	size_t imageSize = image_width * image_height * 3 * sizeof(unsigned char);

	// device memory
	cudaMalloc((void**)&d_image, imageSize);
	checkError(__FILE__, __LINE__);

	// copy image to device mem
	cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice);
	checkError(__FILE__, __LINE__);

	// block and grid sizes
	dim3 blockSize(16, 16);
	dim3 gridSize((image_width + blockSize.x - 1) / blockSize.x, (image_height + blockSize.y - 1) / blockSize.y);

	// launch kernel
	processImageKernel << <gridSize, blockSize >> > (d_image, cam, world, d_rand_state);
	checkError(__FILE__, __LINE__);
	cudaDeviceSynchronize();
	checkError(__FILE__, __LINE__);
	// copy processed device to host
	cudaMemcpy(h_image, d_image, imageSize, cudaMemcpyDeviceToHost);

	cudaFree(d_image);
}