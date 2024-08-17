#pragma once
#include "common.cuh"
#include "hittable.cuh"

class sphere : public hittable {
public:
	__device__ sphere(const point3& center, float radius, material* mat) 
		: center(center), radius(fmaxf(0, radius)), mat(mat) {}
	__device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
		vec3 oc = center - r.origin();
		float a = dot(r.direction(), r.direction());
		float h = dot(r.direction(), oc);
		float c = oc.length_squared() - radius * radius;

		float discriminant = h * h - a * c;
		if (discriminant < 0) {
			return false;
		}

		float sqrtd = sqrtf(discriminant);

		// nearest root
		float root = (h - sqrtd) / a;
		if (!ray_t.surrounds(root)) {
			root = (h + sqrtd) / a;
			if (!ray_t.surrounds(root)) {
				return false;
			}
		}

		rec.t = root;
		rec.p = r.at(rec.t);
		vec3 outward_normal = (rec.p - center) / radius;

		rec.set_face_normal(r, outward_normal);
		rec.mat = mat;

		return true;
	}

private:
	point3 center;
	float radius;
	material* mat;
};