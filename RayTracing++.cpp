#include "SDL.h"
#include <iostream>
#include "./processing.cuh"

int main(int argc, char* argv[]) {



    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
        return -1;
    }

    SDL_Window* window = SDL_CreateWindow("CUDA Image Renderer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 600, SDL_WINDOW_SHOWN);
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
    int width = 800;
    int height = 600;
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

    processImage(image, width, height);

    // Update texture with the processed image
    SDL_UpdateTexture(texture, NULL, image, width * 3);

    // Define button area
    SDL_Rect buttonRect = { 650, 500, 100, 50 }; // x, y, width, height

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
                if (x >= buttonRect.x && x <= (buttonRect.x + buttonRect.w) &&
                    y >= buttonRect.y && y <= (buttonRect.y + buttonRect.h)) {
                    // Button clicked, reprocess image
                    processImage(image, width, height);
                    SDL_UpdateTexture(texture, NULL, image, width * 3);
                }
            }
        }

        SDL_RenderClear(renderer);

        // Render the image
        SDL_RenderCopy(renderer, texture, NULL, NULL);

        // Render the button
        SDL_SetRenderDrawColor(renderer, 0, 0, 255, 255); // Blue button
        SDL_RenderFillRect(renderer, &buttonRect);

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