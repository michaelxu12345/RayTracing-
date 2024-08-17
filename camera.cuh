#pragma once
#include "common.cuh"

#include "hittable.cuh"
#include "hittable_list.cuh"


//__global__ void processImageKernel(unsigned char* d_image, hittable** world, camera& cam) {
//	int x = blockIdx.x * blockDim.x + threadIdx.x;
//	int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//	if ((x >= cam.image_width) || (y >= cam.image_height)) {
//		return;
//	}
//
//	//int idx = (y * width + x) * 3;
//	point3 pixel_center = cam.pixel_upper_left + (x * cam.pixel_du) + (y * cam.pixel_dv);
//	vec3 ray_direction = pixel_center - cam.center;
//	ray r(cam.center, ray_direction);
//	color pixel_color = cam.ray_color(r, world);
//	// write color
//	write_color(d_image, pixel_color, x, y, cam.image_width);
//
//}


class camera {
public:
	float aspect_ratio = 1.0;
	int image_width = 100;
	int image_height;
	point3 center = point3(0, 0, 0);
	point3 pixel_upper_left;
	vec3 pixel_du;
	vec3 pixel_dv;
	int num_samples;

	__host__ __device__ void initialize() {
		image_height = int(image_width / aspect_ratio);
		image_height = (image_height < 1) ? 1 : image_height;

		float focal_length = 1.0f;
		float viewport_height = 2.0f;
		float viewport_width = viewport_height * (float(image_width) / image_height);

		vec3 viewport_u = vec3(viewport_width, 0, 0);
		vec3 viewport_v = vec3(0, -viewport_height, 0);

		pixel_du = viewport_u / image_width;
		pixel_dv = viewport_v / image_height;

		point3 viewport_upper_left = center - vec3(0, 0, focal_length) -
			viewport_u / 2 - viewport_v / 2;
		pixel_upper_left = viewport_upper_left + 0.5f * (pixel_du + pixel_dv);
	}

	__device__ ray get_ray(float u, float v) {
		return ray(center,
			pixel_upper_left + u * pixel_du + v * pixel_dv);
	}

	__device__ vec3 sample_square(curandState* rand_state) {
		return vec3(curand_uniform(rand_state) - 0.5, curand_uniform(rand_state) - 0.5
			, 0);
	}


	/*__host__ __device__ void render(unsigned char* h_image, hittable** world) {
		initialize();

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
		processImageKernel << <gridSize, blockSize >> > (d_image, world);

		// copy processed device to host
		cudaMemcpy(h_image, d_image, imageSize, cudaMemcpyDeviceToHost);

		cudaFree(d_image);
	}*/


	
};


