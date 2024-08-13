#pragma once

#include "common.cuh"

class hit_record {
public:
	point3 p;
	vec3 normal;
	float t;
	bool front_face;

	__device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class hittable {
public:
	__device__ virtual ~hittable() = default;

	__device__ virtual bool hit(const ray&, interval ray_t, hit_record& rec) const = 0;
};