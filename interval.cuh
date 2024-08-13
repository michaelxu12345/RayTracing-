#pragma once

class interval {
public:
	float min, max;
	__device__ interval() : min(+infinity), max(-infinity) {}
	__device__ interval(float _min, float _max) : min(_min), max(_max) {}

	__device__ double size() const {
		return max - min;
	}

	__device__ interval expand(float delta) const {
		float padding = delta / 2;
		return interval(min - padding, max + padding);
	}

	__device__ bool contains(float x) const {
		return min <= x && x <= max;
	}

	__device__ bool surrounds(float x) const {
		return min < x && x < max;
	}

	__device__ float clamp(float x) const {
		if (x < min) return min;
		if (x > max) return max;
		return x;
	}

	static const interval empty, universe;
};

const interval interval::empty = interval(+infinity, -infinity);
const interval interval::universe = interval(-infinity, +infinity);