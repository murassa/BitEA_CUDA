#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include <sys/time.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <chrono>
#include <thrust/copy.h>

#include "BitEA.h"
#include "stdgraph.h"

#define CUDA_CHECK(call)                                                                                \
    {                                                                                                   \
        cudaError_t err = call;                                                                         \
        if (err != cudaSuccess)                                                                         \
        {                                                                                               \
            fprintf(stderr, "CUDA Error: %s, at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1);                                                                                    \
        }                                                                                               \
    }

// int num_threads = SIZE / 4;
// int num_blocks = 1;

// remaining gen count for each thread (multi-thread safe)
__device__ int remaining_gen_count;

__device__ curandState *curand_state;

__global__ void setup_kernel(unsigned long seed)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x; // Unique thread ID
    curand_init(seed + id, id, 0, &curand_state[id]);
}

__device__ inline int __rand()
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    // Get and advance state in one call
    int value = curand(&curand_state[id]);
    // Force another advance of the state
    curand(&curand_state[id]);
    if (value < 0)
        return -value;
    return value;
}

__device__ int lock_pair(int p1, int p2, int *mutexes, int population_size)
{
    int combined = min(p1, p2) * population_size + max(p1, p2);
    return atomicCAS(&mutexes[combined], 0, 1) == 0;
}

__device__ void unlock_pair(int p1, int p2, int *mutexes, int population_size)
{
    int combined = min(p1, p2) * population_size + max(p1, p2);
    atomicExch(&mutexes[combined], 0);
}

__device__ int __popcountl(uint64_t n)
{
    int cnt = 0;
    while (n)
    {
        n &= n - 1; // key point
        ++cnt;
    }
    return cnt;
}

__device__ int __count_conflicts(int graph_size, const block_t *color, const block_t *edges, int *conflict_count)
{
    block_t(*edges_p)[][TOTAL_BLOCK_COUNT] = (block_t(*)[][TOTAL_BLOCK_COUNT])edges;

    int i, j, total_conflicts = 0;
    for (i = 0; i < graph_size; i++)
    {
        if (CHECK_COLOR(color, i))
        {
            conflict_count[i] = 0;
            for (j = 0; j < TOTAL_BLOCK_NUM(graph_size); j++)
                conflict_count[i] += __popcountl(color[j] & (*edges_p)[i][j]);
            total_conflicts += conflict_count[i];
        }
    }

    return total_conflicts / 2;
}

__device__ void fix_conflicts(int graph_size, const block_t *edges, const int *weights, int *conflict_count, int *total_conflicts, block_t *color, block_t *pool, int *pool_total)
{
    block_t(*edges_p)[][TOTAL_BLOCK_COUNT] = (block_t(*)[][TOTAL_BLOCK_COUNT])edges;

    // Keep removing problematic vertices until all conflicts are gone.
    int i, worst_vert = 0, vert_block;
    block_t vert_mask;
    while (*total_conflicts > 0)
    {
        // Find the vertex with the most conflicts.
        for (i = 0; i < SIZE; i++)
        {
            if (CHECK_COLOR(color, i) &&
                (conflict_count[worst_vert] < conflict_count[i] ||
                 (conflict_count[worst_vert] == conflict_count[i] &&
                  (weights[worst_vert] > weights[i] || (weights[worst_vert] == weights[i] && __rand() % 2)))))
            {
                worst_vert = i;
            }
        }

        // Update other conflict counters.
        vert_mask = MASK(worst_vert);
        vert_block = BLOCK_INDEX(worst_vert);
        for (i = 0; i < SIZE; i++)
            if (CHECK_COLOR(color, i) && ((*edges_p)[i][vert_block] & vert_mask))
                conflict_count[i]--;

        // Remove the vertex.
        color[vert_block] &= ~vert_mask;
        pool[vert_block] |= vert_mask;
        (*pool_total)++;

        // Update the total number of conflicts.
        (*total_conflicts) -= conflict_count[worst_vert];
        conflict_count[worst_vert] = 0;
    }
}

__device__ void merge_and_fix(int graph_size, const block_t *edges, const int *weights, const block_t **parent_color, block_t *child_color, block_t *pool, int *pool_count, block_t *used_vertex_list, int *used_vertex_count, int *conflict_count)
{
    // Merge the two colors
    int temp_v_count = 0;
    if (parent_color[0] != NULL && parent_color[1] != NULL)
        for (int i = 0; i < (TOTAL_BLOCK_COUNT); i++)
        {
            child_color[i] = ((parent_color[0][i] | parent_color[1][i]) & ~(used_vertex_list[i]));
            temp_v_count += __popcountl(child_color[i]);

            // printf("child_color[%d]: %llu | parent_color[0][%d]: %llu | parent_color[1][%d]: %llu | used_vertex_list[%d]: %llu | temp_v_count: %d\n", i, child_color[i], i, parent_color[0][i], i, parent_color[1][i], i, used_vertex_list[i], temp_v_count);
        }

    else if (parent_color[0] != NULL)
        for (int i = 0; i < (TOTAL_BLOCK_COUNT); i++)
        {
            child_color[i] = (parent_color[0][i] & ~(used_vertex_list[i]));
            temp_v_count += __popcountl(child_color[i]);
        }

    else if (parent_color[1] != NULL)
        for (int i = 0; i < (TOTAL_BLOCK_COUNT); i++)
        {
            child_color[i] = (parent_color[1][i] & ~(used_vertex_list[i]));
            temp_v_count += __popcountl(child_color[i]);
        }

    (*used_vertex_count) += temp_v_count;

    // Merge the pool with the new color
    for (int i = 0; i < (TOTAL_BLOCK_COUNT); i++)
    {
        child_color[i] |= pool[i];
        used_vertex_list[i] |= child_color[i];
    }

    // memset(pool, 0, (TOTAL_BLOCK_COUNT) * sizeof(block_t));
    for (int i = 0; i < TOTAL_BLOCK_COUNT; i++)
        pool[i] = 0;
    (*pool_count) = 0;

    for (int i = 0; i < SIZE; i++)
        conflict_count[i] = 0;

    // Count conflicts.
    int total_conflicts = __count_conflicts(SIZE, child_color, edges, conflict_count);

    // Fix the conflicts.
    fix_conflicts(graph_size, edges, weights, conflict_count, &total_conflicts, child_color, pool, pool_count);
}

__device__ void search_back(int graph_size, const block_t *edges, const int *weights, block_t *child, int color_count, block_t *pool, int *pool_count)
{
    block_t(*edges_p)[][TOTAL_BLOCK_COUNT] = (block_t(*)[][TOTAL_BLOCK_COUNT])edges;
    block_t(*child_p)[][TOTAL_BLOCK_COUNT] = (block_t(*)[][TOTAL_BLOCK_COUNT])child;

    int conflict_count, last_conflict, last_conflict_block = 0;
    block_t i_mask, temp_mask, last_conflict_mask = 0;
    int i, j, k, i_block;

    // Search back and try placing vertices from the pool in previous colors.
    for (i = 0; i < SIZE && (*pool_count) > 0; i++)
    {
        i_block = BLOCK_INDEX(i);
        i_mask = MASK(i);

        // Check if the vertex is in the pool.
        if (pool[i_block] & i_mask)
        {
            // Loop through every previous color.
            for (j = 0; j < color_count; j++)
            {
                // Count the possible conflicts in this color.
                conflict_count = 0;
                for (k = 0; k < TOTAL_BLOCK_COUNT; k++)
                {
                    temp_mask = (*child_p)[j][k] & (*edges_p)[i][k];
                    if (temp_mask)
                    {
                        conflict_count += __popcountl(temp_mask);
                        if (conflict_count > 1)
                            break;
                        last_conflict = sizeof(block_t) * 8 * (k + 1) - 1 - __clzll(temp_mask);
                        last_conflict_mask = temp_mask;
                        last_conflict_block = k;
                    }
                }

                // Place immediately if there are no conflicts.
                if (conflict_count == 0)
                {
                    (*child_p)[j][i_block] |= i_mask;
                    pool[i_block] &= ~i_mask;
                    (*pool_count)--;
                    break;

                    // If only 1 conflict exists and its weight is smaller
                    // than that of the vertex in question, replace it.
                }
                else if (conflict_count == 1 && weights[last_conflict] < weights[i])
                {
                    (*child_p)[j][i_block] |= i_mask;
                    pool[i_block] &= ~i_mask;

                    (*child_p)[j][last_conflict_block] &= ~last_conflict_mask;
                    pool[last_conflict_block] |= last_conflict_mask;
                    break;
                }
            }
        }
    }
}

__device__ void local_search(int graph_size, const block_t *edges, const int *weights, block_t *child, int color_count, block_t *pool, int *pool_count)
{
    block_t(*edges_p)[][TOTAL_BLOCK_COUNT] = (block_t(*)[][TOTAL_BLOCK_COUNT])edges;
    block_t(*child_p)[][TOTAL_BLOCK_COUNT] = (block_t(*)[][TOTAL_BLOCK_COUNT])child;

    int i, j, k, h, i_block;
    block_t i_mask, temp_mask;
    int competition;
    int conflict_count;
    block_t conflict_array[TOTAL_BLOCK_COUNT];

    // Search back and try placing vertices from the pool in the colors.
    for (i = 0; i < SIZE && (*pool_count) > 0; i++)
    {
        i_block = BLOCK_INDEX(i);
        i_mask = MASK(i);

        // Check if the vertex is in the pool.
        if (pool[i_block] & i_mask)
        {
            // Loop through every color.
            for (j = 0; j < color_count; j++)
            {
                // Count conflicts and calculate competition
                conflict_count = 0;
                competition = 0;
                for (k = 0; k < TOTAL_BLOCK_COUNT; k++)
                {
                    conflict_array[k] = (*edges_p)[i][k] & (*child_p)[j][k];
                    if (conflict_array[k])
                    {
                        temp_mask = conflict_array[k];
                        conflict_count += __popcountl(temp_mask);
                        for (h = 0; h < sizeof(block_t) * 8; h++)
                            if ((temp_mask >> h) & (block_t)1)
                                competition += weights[k * 8 * sizeof(block_t) + h];
                    }
                }

                // Place immediately if there are no conflicts.
                if (competition == 0)
                {
                    (*child_p)[j][i_block] |= i_mask;
                    pool[i_block] &= ~i_mask;
                    (*pool_count) += conflict_count - 1;
                    break;

                    /**
                     * If the total competition is smaller than the weight
                     * of the vertex in question, move all the conflicts to the
                     * pool, and place the vertex in the color.
                     */
                }
                else if (competition < weights[i])
                {
                    for (k = 0; k < TOTAL_BLOCK_COUNT; k++)
                    {
                        (*child_p)[j][k] &= ~conflict_array[k];
                        pool[k] |= conflict_array[k];
                    }

                    (*child_p)[j][i_block] |= i_mask;
                    pool[i_block] &= ~i_mask;
                    (*pool_count) += conflict_count - 1;
                    break;
                }
            }
        }
    }
}

__device__ int get_rand_color(int max_color_num, int colors_used, block_t used_color_list[])
{
    // There are no available colors.
    if (colors_used >= max_color_num)
    {
        return -1;

        // There are only 2 colors available, search for them linearly.
    }
    else if (colors_used > max_color_num - 2)
    {
        for (int i = 0; i < max_color_num; i++)
        {
            if (!(used_color_list[BLOCK_INDEX(i)] & MASK(i)))
            {
                used_color_list[BLOCK_INDEX(i)] |= MASK(i);
                return i;
            }
        }
    }

    // Randomly try to select an available color.
    int temp;
    while (1)
    {
        temp = __rand() % max_color_num;
        if (!(used_color_list[BLOCK_INDEX(temp)] & MASK(temp)))
        {
            used_color_list[BLOCK_INDEX(temp)] |= MASK(temp);
            return temp;
        }
    }
}

__device__ int crossover(int graph_size, const block_t *edges, const int *weights, int color_num1, int color_num2, const block_t *parent1, const block_t *parent2,
                         int target_color_count, block_t *child, int *child_color_count, int *uncolored, int *conflict_count)
{
    const block_t(*parent1_p)[][TOTAL_BLOCK_COUNT] = (block_t(*)[][TOTAL_BLOCK_COUNT])parent1;
    const block_t(*parent2_p)[][TOTAL_BLOCK_COUNT] = (block_t(*)[][TOTAL_BLOCK_COUNT])parent2;
    block_t(*child_p)[][TOTAL_BLOCK_COUNT] = (block_t(*)[][TOTAL_BLOCK_COUNT])child;

    // max number of colors of the two parents.
    // int max_color_num = color_num1 > color_num2 ? color_num1 : color_num2;

    // list of used colors in the parents.
    block_t used_color_list[2][TOTAL_BLOCK_NUM(MAX_COLOR)];
    for (int i = 0; i < TOTAL_BLOCK_NUM(MAX_COLOR); i++)
    {
        used_color_list[0][i] = 0;
        used_color_list[1][i] = 0;
    }

    // list of used vertices in the parents.
    block_t used_vertex_list[TOTAL_BLOCK_COUNT];
    for (int i = 0; i < TOTAL_BLOCK_COUNT; i++)
    {
        used_vertex_list[i] = 0;
    }
    int used_vertex_count = 0;

    // Pool.
    block_t pool[TOTAL_BLOCK_COUNT];
    for (int i = 0; i < TOTAL_BLOCK_COUNT; i++)
    {
        pool[i] = 0;
    }
    int pool_count = 0;

    int color1, color2, last_color = 0;
    int i, j;
    const block_t *chosen_parent_colors[2];
    for (i = 0; i < target_color_count; i++)
    {
        // The child still has vertices that weren't used.
        if (used_vertex_count < SIZE)
        {
            // Pick 2 random colors.
            color1 = get_rand_color(color_num1, i, used_color_list[0]);
            color2 = get_rand_color(color_num2, i, used_color_list[1]);
            chosen_parent_colors[0] = color1 == -1 ? NULL : (*parent1_p)[color1];
            chosen_parent_colors[1] = color2 == -1 ? NULL : (*parent2_p)[color2];

            merge_and_fix(graph_size, edges, weights, chosen_parent_colors, (*child_p)[i], pool, &pool_count, used_vertex_list, &used_vertex_count, conflict_count);

            // If all of the vertices were used and the pool is empty, exit the loop.
        }
        else if (pool_count == 0)
        {
            break;
        }

        search_back(graph_size, edges, weights, child, i, pool, &pool_count);
    }

    // Record the last color.
    last_color = i;

    // If not all the vertices were visited, drop them in the pool.
    if (used_vertex_count < SIZE)
    {
        for (j = 0; j < (TOTAL_BLOCK_COUNT); j++)
            pool[j] |= ~used_vertex_list[j];
        pool[TOTAL_BLOCK_COUNT] &= ((0xFFFFFFFFFFFFFFFF) >> (TOTAL_BLOCK_COUNT * sizeof(block_t) * 8 - SIZE));

        pool_count += (SIZE - used_vertex_count);
        used_vertex_count = SIZE;
        // memset(used_vertex_list, 0xFF, (TOTAL_BLOCK_COUNT) * sizeof(block_t));
        for (i = 0; i < TOTAL_BLOCK_COUNT; i++)
            used_vertex_list[i] = 0xFF;
    }

    // local_search(graph_size, edges, weights, child, target_color_count, pool, &pool_count);

    // If the pool is not empty, randomly allocate the remaining vertices in the colors.
    int fitness = 0, temp_block;
    block_t temp_mask;
    // printf("Pool count: %d\n", pool_count);
    if (pool_count > 0)
    {
        int color_num;
        for (i = 0; i < SIZE; i++)
        {
            temp_block = BLOCK_INDEX(i);
            temp_mask = MASK(i);
            if (pool[temp_block] & temp_mask)
            {
                color_num = __rand() % target_color_count;
                (*child_p)[color_num][temp_block] |= temp_mask;

                if (color_num + 1 > last_color)
                    last_color = color_num + 1;

                fitness += weights[i];
            }
        }

        // All of the vertices were allocated and no conflicts were detected.
    }
    else
    {
        fitness = 0;
    }

    // printf("Fitness: %d | uncolored: %d | child_color_count: %d\n", fitness, pool_count, last_color);

    *uncolored = pool_count;
    *child_color_count = last_color;
    return fitness;
}

__global__ void d_BitEA(int graph_size, block_t **population, block_t **children, int *color_count, int *uncolored, int *fitness, const block_t *edges, int *weights, int population_size,
                        int base_color_count, block_t *best_solution, int *best_fitness, float *best_solution_time, int *uncolored_num, int **conflict_count, int *best_i_result, int *mutexes)
{
    // Get the thread ID
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    bool has_printed = false;

    int best_i = 0;
    int target_color = base_color_count;
    int temp_uncolored;
    int child_colors, temp_fitness;
    int bad_parent;

    long long start = 0;
    auto end = clock64();

    while (atomicSub(&remaining_gen_count, 1) > 0)
    {
        int parent1_locked = 0;
        int parent2_locked = 0;
        int parent1 = -1, parent2 = -1;

        // Select first parent
        do {           
            // Select the parents
            do
            {
                if (parent1 != -1)
                    parent1 = (parent1 + population_size / 3) % population_size;
                else
                    parent1 = __rand() % population_size;
            } while (mutexes[parent1] == 1);
            

            // Select the second parent, ensuring it's different from the first
            do {
                if (parent2 != -1)
                    parent2 = (parent2 + population_size / 3) % population_size;
                else
                    parent2 = __rand() % population_size;
            } while (parent2 == parent1 || mutexes[parent2] == 1); // Ensure the parents are different and not locked
            
            // Try to lock the pair
            if (lock_pair(parent1, parent2, mutexes, population_size))
            {
                parent1_locked = 1;
                parent2_locked = 1;
            }
            else
            {
                parent1_locked = 0;
                parent2_locked = 0;
            }

        } while ((!parent1_locked || !parent2_locked));

        for (int i = 0; i < base_color_count * TOTAL_BLOCK_COUNT; i++)
        {
            children[id][i] = 0;
        }

        temp_fitness = crossover(graph_size, edges, weights, color_count[parent1], color_count[parent2], population[parent1], population[parent2], target_color, children[id], &child_colors, &temp_uncolored, conflict_count[id]);

        // Choose the bad parent.
        if (fitness[parent1] <= fitness[parent2] && color_count[parent1] <= color_count[parent2])
            bad_parent = parent2;
        else
            bad_parent = parent1;

        // Replace the bad parent if needed.
        if (child_colors <= color_count[bad_parent] && temp_fitness <= fitness[bad_parent])
        {
            // Copy child to bad_parent both memory is in device
            // memmove(population[bad_parent], child, (TOTAL_BLOCK_NUM(graph_size))*base_color_count*sizeof(block_t));
            for (int i = 0; i < base_color_count * TOTAL_BLOCK_COUNT; i++)
            {
                population[bad_parent][i] = children[id][i];
            }
            atomicExch(&color_count[bad_parent], child_colors);
            atomicExch(&fitness[bad_parent], temp_fitness);
            atomicExch(&uncolored[bad_parent], temp_uncolored);

            if (temp_fitness < fitness[best_i] ||
                (temp_fitness == fitness[best_i] && child_colors < color_count[best_i]))
            {
                best_i = bad_parent;
            }
        }

        // Make the target harder if it was found.
        if (temp_fitness == 0)
        {
            // is_valid1(SIZE, edges, color_count[best_i], population[best_i]);
            target_color = child_colors - 1;
        }

        unlock_pair(parent1, parent2, mutexes, population_size);

        end = clock64();
        if (id == 0 && (end - start) > 2500000000)
        {
            if (has_printed)
                printf("\033[A\033[K");
            has_printed = true;
            printf("Thread ID: %d | Generation: %d | Fitness: %d | Uncolored: %d | Color Count: %d\n", id, remaining_gen_count, temp_fitness, temp_uncolored, child_colors);
            start = end;
        }
    }

    // Copy the best solution to the global memory
    if (id == 0)
    {
        // memcpy(best_solution, population[best_i], base_color_count * TOTAL_BLOCK_COUNT * sizeof(block_t));
        for (int i = 0; i < base_color_count * TOTAL_BLOCK_COUNT; i++)
        {
            best_solution[i] = population[best_i][i];
        }
        *best_fitness = fitness[best_i];
        *uncolored_num = uncolored[best_i];
        *best_solution_time = (float)(end - start) / 1000000;
        *best_i_result = best_i;

        if (has_printed)
            printf("\033[A\033[K");
    }
}

int BitEA(int graph_size, block_t *edges, int *weights, int population_size, int base_color_count, int max_gen_num, block_t *best_solution, int *best_fitness, float *best_solution_time, int *uncolored_num)
{
    // printf("BitEA\n");

    // int best_color_count = 0;

    int num_threads = 256; // Threads per block
    int num_blocks = ((population_size) + num_threads - 1) / num_threads;

    // // Create the random population.
    block_t **population = (block_t **)malloc(population_size * sizeof(block_t *));
    block_t **children = (block_t **)malloc(num_threads * sizeof(block_t *));
    int *color_count = (int *)malloc(population_size * sizeof(int));
    int *uncolored = (int *)malloc(population_size * sizeof(int));
    int *fitness = (int *)malloc(population_size * sizeof(int));
    int *best_i_result = (int *)malloc(sizeof(int));

    int **conflict_count = (int **)malloc(num_threads * sizeof(int *));

    for (int i = 0; i < num_threads; i++)
    {
        children[i] = (block_t *)malloc(base_color_count * TOTAL_BLOCK_COUNT * sizeof(block_t));
        memset(children[i], 0, base_color_count * TOTAL_BLOCK_COUNT * sizeof(block_t));
        conflict_count[i] = (int *)malloc(SIZE * sizeof(int));
        memset(conflict_count[i], 0, SIZE * sizeof(int));
    }
    for (int i = 0; i < population_size; i++)
    {
        population[i] = (block_t *)malloc(base_color_count * TOTAL_BLOCK_COUNT * sizeof(block_t));
        memset(population[i], 0, base_color_count * TOTAL_BLOCK_COUNT * sizeof(block_t));
        uncolored[i] = base_color_count;
        color_count[i] = base_color_count;
        fitness[i] = __INT_MAX__;
    }

    pop_complex_random(graph_size, edges, weights, population_size, population, base_color_count);

    // Device memory
    block_t **d_population;
    block_t **d_children;
    block_t *d_edges;
    int *d_weights;
    int *d_color_count;
    int *d_uncolored;
    int *d_fitness;
    int *d_best_i_result;

    int **d_conflict_count;

    // Best solution
    block_t *d_best_solution;
    int *d_best_fitness;
    float *d_total_execution_time;
    int *d_best_color_count;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Allocate device memor
    CUDA_CHECK(cudaMalloc(&d_population, population_size * sizeof(block_t *)));
    // Allocate and copy each row
    for (int i = 0; i < population_size; i++)
    {
        block_t *d_row;
        size_t row_size = base_color_count * TOTAL_BLOCK_NUM((size_t)SIZE) * sizeof(block_t);

        // Allocate memory for the row
        CUDA_CHECK(cudaMalloc(&d_row, row_size));

        // Copy data from host to device
        CUDA_CHECK(cudaMemcpy(d_row, population[i], row_size, cudaMemcpyHostToDevice));

        // Copy the device pointer to the device array of pointers
        CUDA_CHECK(cudaMemcpy(&d_population[i], &d_row, sizeof(block_t *), cudaMemcpyHostToDevice));
    }

    // Allocate device memory for children
    CUDA_CHECK(cudaMalloc(&d_children, num_threads * sizeof(block_t *));)
    // Allocate and copy each row
    for (int i = 0; i < num_threads; i++)
    {
        block_t *d_row;
        size_t row_size = base_color_count * TOTAL_BLOCK_NUM((size_t)SIZE) * sizeof(block_t);

        // Allocate memory for the row
        CUDA_CHECK(cudaMalloc(&d_row, row_size);)

        // Copy data from host to device
        CUDA_CHECK(cudaMemcpy(d_row, children[i], row_size, cudaMemcpyHostToDevice);)

        // Copy the device pointer to the device array of pointers
        CUDA_CHECK(cudaMemcpy(&d_children[i], &d_row, sizeof(block_t *), cudaMemcpyHostToDevice);)
    }

    CUDA_CHECK(cudaMalloc(&d_conflict_count, num_threads * sizeof(int *));)
    for (int i = 0; i < num_threads; i++)
    {
        int *d_row;
        size_t row_size = SIZE * sizeof(int);

        // Allocate memory for the row
        CUDA_CHECK(cudaMalloc(&d_row, row_size);)

        // Copy data from host to device
        CUDA_CHECK(cudaMemcpy(d_row, conflict_count[i], row_size, cudaMemcpyHostToDevice);)

        // Copy the device pointer to the device array of pointers
        CUDA_CHECK(cudaMemcpy(&d_conflict_count[i], &d_row, sizeof(int *), cudaMemcpyHostToDevice);)
    }

    CUDA_CHECK(cudaMalloc(&d_edges, SIZE * TOTAL_BLOCK_COUNT * sizeof(block_t));)
    CUDA_CHECK(cudaMemcpy(d_edges, edges, SIZE * TOTAL_BLOCK_COUNT * sizeof(block_t), cudaMemcpyHostToDevice);)

    CUDA_CHECK(cudaMalloc(&d_weights, SIZE * sizeof(int));)
    CUDA_CHECK(cudaMemcpy(d_weights, weights, SIZE * sizeof(int), cudaMemcpyHostToDevice);)

    CUDA_CHECK(cudaMalloc(&d_color_count, population_size * sizeof(int));)
    CUDA_CHECK(cudaMemcpy(d_color_count, color_count, population_size * sizeof(int), cudaMemcpyHostToDevice);)

    CUDA_CHECK(cudaMalloc(&d_uncolored, population_size * sizeof(int));)
    CUDA_CHECK(cudaMemcpy(d_uncolored, uncolored, population_size * sizeof(int), cudaMemcpyHostToDevice);)

    CUDA_CHECK(cudaMalloc(&d_fitness, population_size * sizeof(int));)
    CUDA_CHECK(cudaMemcpy(d_fitness, fitness, population_size * sizeof(int), cudaMemcpyHostToDevice);)

    CUDA_CHECK(cudaMalloc(&d_best_solution, base_color_count * TOTAL_BLOCK_NUM((size_t)SIZE) * sizeof(block_t));)
    CUDA_CHECK(cudaMalloc(&d_best_fitness, sizeof(int));)
    CUDA_CHECK(cudaMalloc(&d_total_execution_time, sizeof(float));)
    CUDA_CHECK(cudaMalloc(&d_best_color_count, sizeof(int));)
    CUDA_CHECK(cudaMalloc(&d_best_i_result, sizeof(int));)

    int *d_mutexes;

    // Allocate and initialize the mutex array on the device
    cudaMalloc(&d_mutexes, population_size * sizeof(int));
    cudaMemset(d_mutexes, 0, population_size * sizeof(int));

    // Allocate memory for curand_state
    curandState *d_curand_state;
    CUDA_CHECK(cudaMalloc(&d_curand_state, num_blocks * num_threads * sizeof(curandState)));
    CUDA_CHECK(cudaMemcpyToSymbol(curand_state, &d_curand_state, sizeof(curandState *)));

    // Initialize curand_state
    setup_kernel<<<num_blocks, num_threads>>>(time(NULL));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy host remaining_gen_count to device
    int host_remaining_gen_count = max_gen_num;
    CUDA_CHECK(cudaMemcpyToSymbol(remaining_gen_count, &host_remaining_gen_count, sizeof(int)));

    #ifdef _WIN32
    auto start_time = std::chrono::high_resolution_clock::now();
    #endif
    #ifdef __linux__
    struct timespec start_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    #endif

    // Run the algorithm.
    d_BitEA<<<num_blocks, num_threads>>>(graph_size, d_population, d_children, d_color_count, d_uncolored, d_fitness, d_edges, d_weights, population_size, base_color_count, d_best_solution,
                                         d_best_fitness, d_total_execution_time, d_best_color_count, d_conflict_count, d_best_i_result, d_mutexes);

    // get last cuda error
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Wait for the kernel to finish.
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the results back to the host.
    CUDA_CHECK(cudaMemcpy(best_solution, d_best_solution, base_color_count * TOTAL_BLOCK_NUM((size_t)SIZE) * sizeof(block_t), cudaMemcpyDeviceToHost);)
    CUDA_CHECK(cudaMemcpy(best_fitness, d_best_fitness, sizeof(int), cudaMemcpyDeviceToHost);)
    // CUDA_CHECK(cudaMemcpy(best_solution_time, d_total_execution_time, sizeof(float), cudaMemcpyDeviceToHost);)
    CUDA_CHECK(cudaMemcpy(uncolored_num, d_best_color_count, sizeof(int), cudaMemcpyDeviceToHost);)
    CUDA_CHECK(cudaMemcpy(best_i_result, d_best_i_result, sizeof(int), cudaMemcpyDeviceToHost);)
    CUDA_CHECK(cudaMemcpy(fitness, d_fitness, population_size * sizeof(int), cudaMemcpyDeviceToHost);)
    CUDA_CHECK(cudaMemcpy(color_count, d_color_count, population_size * sizeof(int), cudaMemcpyDeviceToHost);)
    CUDA_CHECK(cudaMemcpy(uncolored, d_uncolored, population_size * sizeof(int), cudaMemcpyDeviceToHost);)
    CUDA_CHECK(cudaMemcpy(edges, d_edges, SIZE * TOTAL_BLOCK_COUNT * sizeof(block_t), cudaMemcpyDeviceToHost);)
    CUDA_CHECK(cudaMemcpy(weights, d_weights, SIZE * sizeof(int), cudaMemcpyDeviceToHost);)

    // Free device memory
    CUDA_CHECK(cudaFree(d_population);)
    // for (int i = 0; i < population_size; i++)
    // {
    //     CUDA_CHECK(cudaFree(population[i]);)
    // }
    CUDA_CHECK(cudaFree(d_children);)
    // for (int i = 0; i < num_threads; i++)"
    // {
    //     CUDA_CHECK(cudaFree(children[i]);)
    // }
    CUDA_CHECK(cudaFree(d_conflict_count);)
    // for (int i = 0; i < num_threads; i++)
    // {
    //     CUDA_CHECK(cudaFree(conflict_count[i]);)
    // }
    CUDA_CHECK(cudaFree(d_edges);)
    CUDA_CHECK(cudaFree(d_weights);)
    CUDA_CHECK(cudaFree(d_color_count);)
    CUDA_CHECK(cudaFree(d_uncolored);)
    CUDA_CHECK(cudaFree(d_fitness);)
    CUDA_CHECK(cudaFree(d_best_solution);)
    CUDA_CHECK(cudaFree(d_best_fitness);)
    CUDA_CHECK(cudaFree(d_total_execution_time);)
    CUDA_CHECK(cudaFree(d_best_color_count);)
    CUDA_CHECK(cudaFree(d_best_i_result);)
    CUDA_CHECK(cudaFree(d_mutexes);)

    #ifdef _WIN32
    auto end_time = std::chrono::high_resolution_clock::now();
    *best_solution_time = std::chrono::duration_cast<std::chrono::duration<float>>(end_time - start_time).count();
    #endif
    #ifdef __linux__
    struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    *best_solution_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1000000000.0;
    #endif

    // Free host memory
    //

    return color_count[*best_i_result];
}