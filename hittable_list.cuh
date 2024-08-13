#pragma once

#include "common.cuh"


class hittable_list : public hittable {
public:
	hittable** objects;
	int size;

	__device__ hittable_list() {}
	__device__ hittable_list(hittable** list, int n) { objects = list; size = n; }
	__device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const;
	
};

__device__ bool hittable_list::hit(const ray& r, interval ray_t, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = ray_t.max;
    for (int i = 0; i < size; i++) {
        if (objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}