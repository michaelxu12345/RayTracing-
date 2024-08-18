#pragma once
#include "common.cuh"

#include "hittable.cuh"
#include "hittable_list.cuh"



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

	float vfov = 90;
	point3 lookfrom = point3(0, 0, 0);
	point3 lookat = point3(0, 0, -1);
	vec3 vup = vec3(0, 1, 0);
	vec3 u, v, w;

	float defocus_angle = 0;
	float focus_dist = 10;
	vec3 defocus_disk_u;
	vec3 defocus_disk_v;

	__host__ __device__ void initialize() {
		image_height = int(image_width / aspect_ratio);
		image_height = (image_height < 1) ? 1 : image_height;

		center = lookfrom;

		// Determine viewport dimensions.
		float theta = degrees_to_radians(vfov);
		float h = tanf(theta / 2);
		float viewport_height = 2.0f;
		float viewport_width = viewport_height * (float(image_width) / image_height);

		// calculate u, v, w unit basis vectors
		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup, w));
		v = cross(w, u);

		// vectors across horizontal and down vertical viewport edges
		vec3 viewport_u = viewport_width * u;
		vec3 viewport_v = viewport_height * -v;

		// horizontal and vertical delta vectors pixel to pixel
		pixel_du = viewport_u / image_width;
		pixel_dv = viewport_v / image_height;

		// location of upper left pixel
		point3 viewport_upper_left = center - (focus_dist * w) -
			viewport_u / 2 - viewport_v / 2;
		pixel_upper_left = viewport_upper_left + 0.5f * (pixel_du + pixel_dv);

		float defocus_radius = focus_dist * tanf(degrees_to_radians(defocus_angle / 2));
		defocus_disk_u = u * defocus_radius;
		defocus_disk_v = v * defocus_radius;
	}

	__device__ ray get_ray(float u, float v, curandState* rand_state) {

		vec3 pixel_center = pixel_upper_left + u * pixel_du + v * pixel_dv;
		vec3 pixel_sample = pixel_center + pixel_sample_square(rand_state);

		vec3 ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample(rand_state);
		vec3 ray_direction = pixel_sample - ray_origin;

		return ray(ray_origin, ray_direction);
	}

	__device__ vec3 sample_square(curandState* rand_state) {
		return vec3(curand_uniform(rand_state) - 0.5, curand_uniform(rand_state) - 0.5
			, 0);
	}

	__device__ vec3 pixel_sample_square(curandState* rand_state) const {
		float px = -0.5 + random_double(rand_state);
		float py = -0.5 + random_double(rand_state);
		return (px * pixel_du) + (py * pixel_dv);
	}

	__device__ point3 defocus_disk_sample(curandState* rand_state) const {
		point3 p = random_in_unit_disk(rand_state);
		return center + p[0] * defocus_disk_u + p[1] * defocus_disk_v;
	}
	
};


