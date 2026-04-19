#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define SEED 27
#define H 500
#define W 500
#define SCALE 20
#define SAMPLE 500
#define RUNS 10000
#define BIAS 5.0f
#define LEARNING_RATE 0.5f
#define TEST 1


typedef float Layer[H][W];

Layer model;
Layer* rect_samples;
Layer* circ_samples;

void set_layer(Layer input, float v){
    for(int i = 0; i < H; i++){
        for(int j = 0; j < W; j++){
            model[i][j] = v;
        }
    }
}

float get_max(Layer input){
    float max = input[0][0];
    for(int i = 0; i < H; i++){
        for(int j = 0; j < W; j++){
            if(input[i][j] > max){
                max = input[i][j];
            }
        }
    }
    return max;
}

float get_min(Layer input){
    float min = input[0][0];
    for(int i = 0; i < H; i++){
        for(int j = 0; j < W; j++){
            if(input[i][j] < min){
                min = input[i][j];
            }
        }
    }
    return min;
}

void save_as_ppm(Layer input, char* file){
    FILE *fp = fopen(file, "wb");
    if(fp == NULL){
        fprintf(stderr, "[ERROR] Failed to open file for writing: %s\n", file);
        perror("[ERROR] fopen");
        exit(1);
    }
    fprintf(fp, "P6\n%d %d 255\n", W*SCALE, H * SCALE);
    float max = get_max(input);
    float min = get_min(input);
    for(int i = 0; i < H*SCALE; i++){
        for(int j = 0; j < W*SCALE; j++){
            float v = input[i/SCALE][j/SCALE];
            unsigned char r = (unsigned char)((v-min)/(max-min)*255);
            unsigned char g = (unsigned char)((max-v)/(max-min)*255);
            unsigned char b = 0;
            if(fwrite(&r, 1, 1, fp) != 1 || fwrite(&g, 1, 1, fp) != 1 || fwrite(&b, 1, 1, fp) != 1){
                fprintf(stderr, "[ERROR] Failed to write pixel at [%d][%d] to file: %s\n", i, j, file);
                perror("[ERROR] fwrite");
                fclose(fp);
                exit(1);
            }
        }
    }
    fclose(fp);
}

void save_as_bin(Layer input, char* file){
    FILE *fp = fopen(file, "wb");
    for(int i = 0; i < H; i++){
        for(int j = 0; j < W; j++){
            float v = input[i][j];
            fwrite(&v, sizeof(float), 1, fp);
        }
    }
    fclose(fp);
}

int save_model_bin(char* file){
    FILE *fp = fopen(file, "wb");
    if(fp == NULL){
        fprintf(stderr, "[ERROR] Failed to open file for writing: %s\n", file);
        perror("[ERROR] fopen");
        exit(1);
    }
    for(int i = 0; i < H; i++){
        for(int j = 0; j < W; j++){
            float v = model[i][j];
            if(fwrite(&v, sizeof(float), 1, fp) != 1){
                fprintf(stderr, "[ERROR] Failed to write float at [%d][%d] to file: %s\n", i, j, file);
                perror("[ERROR] fwrite");
                fclose(fp);
                exit(1);
            }
        }
    }
    fclose(fp);
    return 0;
}

int load_model_bin(char* file){
    FILE *fp = fopen(file, "rb");
    if(fp == NULL){
        fprintf(stderr, "[ERROR] Failed to open file for reading: %s\n", file);
        perror("[ERROR] fopen");
        exit(1);
    }
    for(int i = 0; i < H; i++){
        for(int j = 0; j < W; j++){
            float v;
            if(fread(&v, sizeof(float), 1, fp) != 1){
                fprintf(stderr, "[ERROR] Failed to read float at [%d][%d] from file: %s\n", i, j, file);
                perror("[ERROR] fread");
                fclose(fp);
                exit(1);
            }
            model[i][j] = v;
        }
    }
    fclose(fp);
    return 0;
}

void gene_rect(Layer output, int x, int y, int w, int h, float v){
    assert(x >= 0 && y >= 0 && w > 0 && h > 0 && x+w <= W && y+h <= H);
    set_layer(output, 0.0);
    for(int i = y; i < y+h; i++){
        for(int j = x; j < x+w; j++){
            output[i][j] = v;
        }
    }
}

void gene_circ(Layer output, int x, int y, int r, float v){
    assert(x >= 0 && y >= 0 && r > 0 && x-r >= 0 && y-r >= 0 && x+r <= W && y+r <= H);
    set_layer(output, 0.0);
    for(int i = y-r; i <= y+r; i++){
        for(int j = x-r; j <= x+r; j++){
            if((i-y)*(i-y) + (j-x)*(j-x) <= r*r){
                output[i][j] = v;
            }
        }
    }
}

float rand_range(int min, int max){
    return (float)rand()/RAND_MAX * (max - min) + min;
}

float gaussian_rand(float mean, float stddev){
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    while(u1 == 0) u1 = (float)rand() / RAND_MAX;
    return mean + stddev * sqrt(-2.0 * log(u1)) * cos(2.0 * 3.1416 * u2);
}

void init_model(){
    for(int i = 0; i < H; i++){
        for(int j = 0; j < W; j++){
            model[i][j] = rand_range(0, 1);
        }
    }
}

int perceptron(Layer input){
    float sum = 0.0;
    for(int i = 0; i < H; i++){
        for(int j = 0; j < W; j++){
            sum += input[i][j] * model[i][j];
        }
    }
    return sum > BIAS ? 1 : 0;
}

void sub(Layer input){
    for(int i = 0; i < H; i++){
        for(int j = 0; j < W; j++){
            model[i][j] -= input[i][j] * LEARNING_RATE;
        }
    }
}

void add(Layer input){
    for(int i = 0; i < H; i++){
        for(int j = 0; j < W; j++){
            model[i][j] += input[i][j] * LEARNING_RATE;
        }
    }
}

void train(){
    char __filepath[100];
    int runs = 0;
    int last_correct = 0;
    int last_percent = 0;
    int last_save_threshold = 0;
    while(1){
        int correct = 0;
        for(int i = 0; i < SAMPLE; ++i){
            if(perceptron(rect_samples[i])){
                correct++;
            }else{
                add(rect_samples[i]);
            }
            if(!perceptron(circ_samples[i])){
                correct++;
            }else{
                sub(circ_samples[i]);
            }
        }
        int correct_percent = correct * 100 / (SAMPLE * 2);
        if(correct_percent >= last_percent + 8 || correct == SAMPLE * 2){
            last_percent = last_correct * 100 / (SAMPLE * 2);
            snprintf(__filepath, sizeof(__filepath), "model%d\\model_%d.ppm", SEED, runs);
            save_as_ppm(model, __filepath);
            printf("runs: %d, correct: %d / %d (saved)\n", runs, correct, SAMPLE*2);
        }else{
            // printf("runs: %d, correct: %d / %d\n", runs, correct, SAMPLE*2);
        }
        if(correct == SAMPLE * 2){
            save_model_bin("final_model.bin");
            printf("Full accuracy reached!\n");
            break;
        }
        if(runs >= RUNS){
            save_model_bin("final_model.bin");
            printf("Max runs reached, giving up\n");
            break;
        }
        last_correct = correct;
        runs++;
    }
}

void test(int n){
    int correct = 0;
    Layer rect, circ;
    if(n)save_as_ppm(model, "final_model\\final_model.ppm");
    else{save_as_ppm(model, "first_model\\first_model.ppm");}
        for(int i = 0; i < SAMPLE; ++i){
            int x = rand_range(1, W-2);
            int y = rand_range(1, H-2);
            int h = rand_range(1, H-y-1);
            int w = rand_range(1, W-x-1);
            float v = rand_range(0, 1);
            gene_rect(rect, x, y, w, h, v);

            x = gaussian_rand(W/2, W/6);
            if(x < 2) x = 2;
            if(x > W-3) x = W-3;
            y = gaussian_rand(H/2, H/6);
            if(y < 2) y = 2;
            if(y > H-3) y = H-3;
            int r = (int)(gaussian_rand(W/6, W/12) + 0.5);
            if(r < 1) r = 1;
            if(r > W/4) r = W/4;
            if(r > x) r = x;
            if(r > y) r = y;
            if(r > W-x) r = W-x;
            if(r > H-y) r = H-y;
            v = rand_range(0, 1);
            gene_circ(circ, x, y, r, v);

            if(perceptron(rect)){
                correct++;
            }
            if(!perceptron(circ)){
                correct++;
            }
        }
    printf("Correct: %d / %d\n", correct, SAMPLE*2);
}


int main(void){
    if(TEST){
        srand(SEED);
        init_model();

        rect_samples = (Layer*)malloc(SAMPLE * sizeof(Layer));
        circ_samples = (Layer*)malloc(SAMPLE * sizeof(Layer));
        for(int i = 0; i < SAMPLE; ++i){
            int x = rand_range(1, W-2);
            int y = rand_range(1, H-2);
            int h = rand_range(1, H-y-1);
            int w = rand_range(1, W-x-1);
            float v = rand_range(0, 1);
            gene_rect(rect_samples[i], x, y, w, h, v);

            x = gaussian_rand(W/2, W/6);
            if(x < 2) x = 2;
            if(x > W-3) x = W-3;
            y = gaussian_rand(H/2, H/6);
            if(y < 2) y = 2;
            if(y > H-3) y = H-3;
            int r = (int)(gaussian_rand(W/6, W/12) + 0.5);
            if(r < 1) r = 1;
            if(r > W/4) r = W/4;
            if(r > x) r = x;
            if(r > y) r = y;
            if(r > W-x) r = W-x;
            if(r > H-y) r = H-y;
            v = rand_range(0, 1);
            gene_circ(circ_samples[i], x, y, r, v);
        }

        srand(SEED);
        printf("Test mode enabled (before training)\n");
        test(0);

        srand(SEED);
        train();

        srand(SEED);
        printf("Test mode enabled (after training)\n");
        test(1);
    }else{        
        srand(SEED);
        init_model();
        train();
    }
    return 0;
}