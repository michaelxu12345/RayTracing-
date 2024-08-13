#pragma once

#include "hittable.cuh"

int MAX_OBJECTS = 10;

class hittable_list : public hittable {
public:
	hittable** objects;
	int size = 0;

	hittable_list() {}
	hittable_list(hittable* object) { add(object); }
	
	void clear() {

	}
	void add(hittable* object) {
		objects[size] = object;
		size++;
	}

};