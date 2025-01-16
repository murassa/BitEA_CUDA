#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <chrono>
#include <memory>

#include "BitEA.h"
#include "stdgraph.h"

struct test_return
{
    block_t *solution;
    int color_count;
    int fitness;
    int uncolored;
    float best_time;
    char summary[512];
};

struct test_param
{
    int size;
    int target_color;
    int iteration_count;
    int test_count;
    int population_size;
    char graph_filename[128];
    char weight_filename[128];
    char result_filename[128];
    char time_filename[128];
    FILE *summary_file;
    struct test_return result;
};

void print_times(char *filename, float total_time, float k_time, int k, int population_size, int color_count)
{
    // open file and write the time at the end of the file
    FILE *time_file = fopen(filename, "a+");
    if (time_file == NULL)
    {
        printf("time file could not be opened\n");
        return;
    }

    fprintf(time_file, "%f %f %d %d %d\n", total_time, k_time, k, population_size, color_count);
    fclose(time_file);
}

void test_graph(const void *param, int *best_result)
{
    int size = ((struct test_param *)param)->size;
    int iteration_count = ((struct test_param *)param)->iteration_count;
    int target_color = ((struct test_param *)param)->target_color;
    int population_size = ((struct test_param *)param)->population_size;
    char *graph_filename = ((struct test_param *)param)->graph_filename;
    char *weight_filename = ((struct test_param *)param)->weight_filename;
    char *result_filename = ((struct test_param *)param)->result_filename;
    char *time_filename = ((struct test_param *)param)->time_filename;

    auto edges_ptr = std::make_unique<block_t[]>(size * TOTAL_BLOCK_NUM(size));
    for (int i = 0; i < size * TOTAL_BLOCK_NUM(size); i++)
        edges_ptr[i] = 0;
    block_t *edges = edges_ptr.get();
    if (!read_graph(graph_filename, size, edges, 0))
    {
        printf("Could not initialize graph from %s, exiting ...\n", graph_filename);
        return;
    }

    auto edge_count_ptr = std::make_unique<int[]>(size);
    for (int i = 0; i < size; i++)
        edge_count_ptr[i] = 0;
    int *edge_count = edge_count_ptr.get();
    count_edges(size, edges, edge_count);

    auto weights_ptr = std::make_unique<int[]>(size);
    for (int i = 0; i < size; i++)
        weights_ptr[i] = 1;
    int *weights = weights_ptr.get();
    if (strncmp(weight_filename, "null", 4) != 0)
    {
        if (!read_weights(weight_filename, size, weights))
        {
            printf("Could not initialize graph weights from %s, exiting ...\n", weight_filename);
            return;
        }
    }
    else
    {
        for (int i = 0; i < size; i++)
            weights[i] = edge_count[i];
    }

    int max_edge_count = 0;
    for (int i = 0; i < size; i++)
        if (max_edge_count < edge_count[i])
            max_edge_count = edge_count[i];

    float temp_time = 0;
    int temp_fitness = 0, temp_color_count = 0, temp_uncolored = 0;
    block_t *temp_colors = (block_t *)malloc(size * sizeof(block_t));

    float total_execution_time = 0;

#ifdef _WIN32
    std::chrono::high_resolution_clock::time_point t1, t2;
    t1 = std::chrono::high_resolution_clock::now();
#endif
#ifdef __linux__
    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
#endif

    temp_color_count = BitEA(size, edges, weights, population_size, target_color, iteration_count, temp_colors, &temp_fitness, &temp_time, &temp_uncolored);

#ifdef _WIN32
    t2 = std::chrono::high_resolution_clock::now();
    total_execution_time += std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1).count();
#endif
#ifdef __linux__
    clock_gettime(CLOCK_MONOTONIC, &t2);
    total_execution_time += (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1000000000.0;
#endif

    if (temp_fitness == 0)
        is_valid(size, edges, temp_color_count, temp_colors);

    printf("|%s|%3d|%10.6lf|%3d|%5d|%3d|%10.6lf|\n",
           graph_filename,
           target_color,
           temp_time,
           temp_color_count,
           temp_fitness,
           temp_uncolored,
           total_execution_time);

    // fprintf(((struct test_param *)param)->summary_file, "|%s|%3d|%10.6lf|%3d|%5d|%3d|%10.6lf|\n",
    //         graph_filename,
    //         target_color,
    //         temp_time,
    //         temp_color_count,
    //         temp_fitness,
    //         temp_uncolored,
    //         total_execution_time);

    // printf("best fitness: %d\n", *best_result);
    // fflush(stdout);

    if (*best_result > temp_fitness)
    {
        *best_result = temp_fitness;

        char buffer[2048];
        sprintf(buffer, "|  graph name   | target color | k time | k | cost | uncolored | total time |\n|%s|%3d|%10.6lf|%3d|%5d|%3d|%10.6lf|\n",
                graph_filename,
                target_color,
                temp_time,
                temp_color_count,
                temp_fitness,
                temp_uncolored,
                total_execution_time);

        print_colors(result_filename, buffer, target_color, size, temp_colors);
    }

    // printf("filename: %s time: %f k: %d uncolored: %d population size: %d color count: %d\n", time_filename, total_execution_time, temp_time, temp_uncolored, population_size, temp_color_count);
    // fflush(stdout);

    print_times(time_filename, total_execution_time, temp_time, temp_uncolored, population_size, temp_color_count);

    if (temp_colors != nullptr)
        free(temp_colors);
}

int main(int argc, char *argv[])
{
    srand(time(NULL));

    printf("|  graph name   | target color | k time | k | cost | uncolored | total time |\n");

    if (argc == 2)
    {
        FILE *test_list_file = fopen(argv[1], "r");
        if (test_list_file == NULL)
        {
            printf("test file not found\n");
            return 0;
        }

        char buffer[512];
        struct test_param param;
        int test_count;
        while (fgets(buffer, 256, test_list_file) != NULL)
        {
            buffer[strcspn(buffer, "\n")] = 0;

            param.size = atoi(strtok(buffer, " "));
            param.target_color = atoi(strtok(NULL, " "));
            param.iteration_count = atoi(strtok(NULL, " "));
            param.population_size = atoi(strtok(NULL, " "));
            test_count = atoi(strtok(NULL, " "));
            strcpy(param.graph_filename, strtok(NULL, " "));
            strcpy(param.weight_filename, strtok(NULL, " "));
            strcpy(param.result_filename, strtok(NULL, " "));
            strcpy(param.time_filename, strtok(NULL, " "));

            int best_fitness = __INT_MAX__;
            for (; test_count > 0; test_count--)
                test_graph(&param, &best_fitness);
        }

        fclose(test_list_file);
    }
    else
    {
        int i = 0;
        if (argc == 3)
            i = atoi(argv[2]);
        // file names are tests_0.txt, tests_1.txt, tests_2.txt, ...
        for (; i < 73; i++)
        {
            char filename[32];
            sprintf(filename, "tests/tests_%d.txt", i);
            FILE *test_list_file = fopen(filename, "r");
            if (test_list_file == NULL)
            {
                printf("test file not found\n");
                return 0;
            }

            char buffer[512];
            struct test_param param;
            int test_count;
            while (fgets(buffer, 256, test_list_file) != NULL)
            {
                buffer[strcspn(buffer, "\n")] = 0;

                param.size = atoi(strtok(buffer, " "));
                param.target_color = atoi(strtok(NULL, " "));
                param.iteration_count = atoi(strtok(NULL, " "));
                param.population_size = atoi(strtok(NULL, " "));
                test_count = atoi(strtok(NULL, " "));
                strcpy(param.graph_filename, strtok(NULL, " "));
                strcpy(param.weight_filename, strtok(NULL, " "));
                strcpy(param.result_filename, strtok(NULL, " "));
                strcpy(param.time_filename, strtok(NULL, " "));

                int best_fitness = __INT_MAX__;
                for (; test_count > 0; test_count--)
                {
                    test_graph(&param, &best_fitness);
                }
            }

            printf("-----------------%d---------------\n", i);

            fclose(test_list_file);
        }
    }
}
