#include "common.cuh"
#include "SDL.h"
#include "processing.cu"
#include "hittable.cuh"
#include "hittable_list.cuh"
#include "sphere.cuh"
#include <curand_kernel.h>


__global__ void create_world(hittable** d_list, hittable** d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(vec3(0, 0, -1), 0.5,
                        new lambertian(vec3(0.1, 0.2, 0.5)));
        d_list[1] = new sphere(vec3(0, -100.5, -1), 100,
                        new lambertian(vec3(0.1, 0.2, 0.5)));
        *d_world = new hittable_list(d_list, 2);
    }
}

__global__ void render_init(int width, int height, curandState* rand_state) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    /*if ((x >= width) || (y >= height)) {
        return;
    }*/

    int pixel_index = y * width + x;
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);

    //int idx = (y * width + x) * 3;
}

int main(int argc, char* argv[]) {

    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
        return -1;
    }

    SDL_Window* window = SDL_CreateWindow("CUDA Image Renderer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720, SDL_WINDOW_SHOWN);
    if (!window) {
        std::cerr << "Failed to create window: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return -1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        std::cerr << "Failed to create renderer: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    // Create an SDL texture
    int width = 1280;
    int height = 720;
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, width, height);

    // Allocate memory for image
    unsigned char* image = new unsigned char[width * height * 3]; // Assuming RGB format
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int idx = 3 * (j * width + i);
            image[idx] = unsigned char(int(255.9999 * ((double(i)) / (width - 1))));
            image[idx + 1] = unsigned char(int(255.9999 * ((double(j)) / (height - 1))));
            image[idx + 2] = 0;
        }
    }

    // Setting up cam
    camera cam;

    cam.aspect_ratio = float(width) / float(height);
    cam.image_width = width;
    cam.image_height = height;
    cam.initialize();
    cam.num_samples = 10; // 1 for no antialiasing

    // Setting random numbers
    curandState* d_rand_state;
    cudaMalloc((void**)&d_rand_state, width * height * sizeof(curandState));

    dim3 blockSize(16, 16);
    dim3 gridSize((cam.image_width + blockSize.x - 1) / blockSize.x, (cam.image_height + blockSize.y - 1) / blockSize.y);

    render_init << <gridSize, blockSize >> > (width, height, d_rand_state);
    checkError(__FILE__, __LINE__);
    cudaDeviceSynchronize();

    // Creating world
    hittable** d_list;
    hittable** d_world;
    cudaMalloc((void**)&d_list, 2 * sizeof(hittable*));
    cudaMalloc((void**)&d_world, sizeof(hittable*));

    create_world <<<1, 1 >>> (d_list, d_world);
    checkError(__FILE__, __LINE__);
    cudaDeviceSynchronize();

    

    processImage(image, d_world, cam, d_rand_state);

    //cudaDeviceSynchronize();
    
    std::cout << "processed image" << std::endl;

    // Update texture with the processed image
    SDL_UpdateTexture(texture, NULL, image, width * 3);

    std::cout << "updated texture" << std::endl;
    // Define button area
    SDL_Rect buttonRectRight = { 1100, 590, 100, 50 }; // x, y, width, height
    SDL_Rect buttonRectLeft = { 20, 590, 100, 50 }; // x, y, width, height
    // Main loop
    bool running = true;
    SDL_Event event;
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
            else if (event.type == SDL_MOUSEBUTTONDOWN) {
                int x, y;
                SDL_GetMouseState(&x, &y);
                if (x >= buttonRectRight.x && x <= (buttonRectRight.x + buttonRectRight.w) &&
                    y >= buttonRectRight.y && y <= (buttonRectRight.y + buttonRectRight.h)) {
                    // Button clicked, reprocess image
                    cam.center += vec3(0, 0, -0.1);
                    cam.initialize();
                    processImage(image,d_world, cam, d_rand_state);
                    
                    SDL_UpdateTexture(texture, NULL, image, width * 3);
                }
                else if (x >= buttonRectLeft.x && x <= (buttonRectLeft.x + buttonRectLeft.w) &&
                    y >= buttonRectLeft.y && y <= (buttonRectLeft.y + buttonRectLeft.h)) {
                    // Button clicked, reprocess image
                    cam.center += vec3(0, -0.1, 0);
                    cam.initialize();
                    processImage(image, d_world, cam, d_rand_state);
                    
                    SDL_UpdateTexture(texture, NULL, image, width * 3);
                }
            }
        }

        SDL_RenderClear(renderer);

        // Render the image
        SDL_RenderCopy(renderer, texture, NULL, NULL);

        // Render the button
        SDL_SetRenderDrawColor(renderer, 0, 0, 255, 255); // Blue button
        SDL_RenderFillRect(renderer, &buttonRectRight);

        SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255); // Blue button
        SDL_RenderFillRect(renderer, &buttonRectLeft);

        // Optionally, render button text (requires SDL_ttf for text rendering, not covered here)

        SDL_RenderPresent(renderer);
    }

    // Clean up
    delete[] image;
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}