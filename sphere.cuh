#pragma once

#include "hittable.cuh"
#include "vec3.cuh"

class sphere : public hittable {
public:
	__device__ sphere(const point3& center, float radius) : center(center), radius(fmaxf(0, radius)) {}
	__device__ bool hit(const ray& r, float ray_tmin, float ray_tmax, hit_record& rec) const override {
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
		if (root <= ray_tmin || ray_tmax <= root) {
			root = (h + sqrtd) / a;
			if (root <= ray_tmin || ray_tmax <= root) {
				return false;
			}
		}

		rec.t = root;
		rec.p = r.at(rec.t);
		vec3 outward_normal = (rec.p - center) / radius;

		rec.set_face_normal(r, outward_normal);

		return true;
	}

private:
	point3 center;
	float radius;
};