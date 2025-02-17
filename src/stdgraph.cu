#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include "stdgraph.h"

void graph_color_random(int graph_size, block_t **edges, block_t **colors, int max_color)
{
    for (int i = 0; i < graph_size; i++)
    {
        int index = rand() % max_color;
        SET_COLOR(colors[index], i);
    }
}

#ifdef _WIN32
int comp_crit_1(void *metrics, const void *a, const void *b)
{
    int **metric_array = (int **)metrics;
    int *weights = metric_array[0];
    int *degrees = metric_array[1];

    return (weights[*(int *)a] * degrees[*(int *)a]) - (weights[*(int *)b] * degrees[*(int *)b]);
}

int comp_crit_2(void *metrics, const void *a, const void *b)
{
    int **metric_array = (int **)metrics;
    int *weights = metric_array[0];
    int *degrees = metric_array[1];

    return (weights[*(int *)a] * degrees[*(int *)a] * degrees[*(int *)a]) -
           (weights[*(int *)b] * degrees[*(int *)b] * degrees[*(int *)b]);
}

int comp_crit_3(void *weights, const void *a, const void *b)
{
    return ((int *)weights)[*(int *)a] - ((int *)weights)[*(int *)b];
}
#endif

#ifdef __linux__
int comp_crit_1(const void *a, const void *b, void *metrics)
{
    int *weights = ((int **)metrics)[0];
    int *degrees = ((int **)metrics)[1];
    return (weights[*(int *)a] * degrees[*(int *)a]) - (weights[*(int *)b] * degrees[*(int *)b]);
}

int comp_crit_2(const void *a, const void *b, void *metrics)
{
    int *weights = ((int **)metrics)[0];
    int *degrees = ((int **)metrics)[1];
    return (weights[*(int *)a] * degrees[*(int *)a] * degrees[*(int *)a]) - (weights[*(int *)b] * degrees[*(int *)b] * degrees[*(int *)b]);
}

int comp_crit_3(const void *a, const void *b, void *weights)
{
    return (((int *)weights)[*(int *)a]) - (((int *)weights)[*(int *)b]);
}
#endif

int popcountl(uint64_t n)
{
    int cnt = 0;
    while (n)
    {
        n &= n - 1; // key point
        ++cnt;
    }
    return cnt;
}

void pop_complex_random(int graph_size, const block_t *edges, const int *weights, int pop_size, block_t **population, int max_color)
{
    int *criteria = (int *)malloc(3 * graph_size * sizeof(int));
    for (int i = 0; i < graph_size; i++)
    {
        criteria[0 * graph_size + i] = i;
        criteria[1 * graph_size + i] = i;
        criteria[2 * graph_size + i] = i;
    }

    int *degrees = (int *)malloc(graph_size * sizeof(int));
    for (int i = 0; i < graph_size; i++)
        degrees[i] = 0;
    if (degrees == nullptr)
    {
        printf("Could not allocate memory for degrees.\n");
        return;
    }
    count_edges(graph_size, edges, degrees);

    const int *metrics[2] = {weights, degrees};

#ifdef _WIN32
    qsort_s(&criteria[0 * graph_size], graph_size, sizeof(int), comp_crit_1, (void *)metrics);
    qsort_s(&criteria[1 * graph_size], graph_size, sizeof(int), comp_crit_2, (void *)metrics);
    qsort_s(&criteria[2 * graph_size], graph_size, sizeof(int), comp_crit_3, (void *)weights);
#endif

#ifdef __linux__
    qsort_r(&criteria[0 * graph_size], graph_size, sizeof(int), comp_crit_1, (void *)metrics);
    qsort_r(&criteria[1 * graph_size], graph_size, sizeof(int), comp_crit_2, (void *)metrics);
    qsort_r(&criteria[2 * graph_size], graph_size, sizeof(int), comp_crit_3, (void *)weights);
#endif

    block_t *adjacent_colors = (block_t *)malloc(TOTAL_BLOCK_NUM(graph_size) * sizeof(block_t));
    int current_vert;
    int i, j, k;
    for (int indiv_id = 0; indiv_id < pop_size; indiv_id++)
    {
        // set population[indiv_id] all 0
        for (i = 0; i < max_color * TOTAL_BLOCK_NUM(graph_size); i++)
            population[indiv_id][i] = 0;
        
        for (i = 0; i < graph_size; i++)
        {
            if (indiv_id < pop_size * 0.4)
                current_vert = criteria[0 * graph_size + i];
            else if (indiv_id < pop_size * 0.8)
                current_vert = criteria[1 * graph_size + i];
            else
                current_vert = criteria[2 * graph_size + i];

            // Initialize the temporary data.
            for (j = 0; j < TOTAL_BLOCK_NUM(graph_size); j++)
                adjacent_colors[j] = 0;
            for (j = 0; j < TOTAL_BLOCK_NUM(graph_size); j++)
            {
                for (k = 0; k < max_color; k++)
                {
                    // SET_COLOR(adjacent_colors, edges[current_vert][j] & (*indiv)[k][j] & 1);
                    if (current_vert * TOTAL_BLOCK_NUM(graph_size) + j >= graph_size * TOTAL_BLOCK_NUM(graph_size))
                    {
                        printf("Error: %d %d %d\n", current_vert, j, graph_size);
                    }
                    if (edges[current_vert * TOTAL_BLOCK_NUM(graph_size) + j] & (population[indiv_id])[k * TOTAL_BLOCK_NUM(graph_size) + j])
                    {
                        SET_COLOR(adjacent_colors, k);
                        break;
                    }
                }
            }

            // Find the first unused color (starting from 0) and assign this vertex to it.
            for (j = 0; j < max_color; j++)
            {
                if (!CHECK_COLOR(adjacent_colors, j))
                {
                    (population[indiv_id])[j * TOTAL_BLOCK_NUM(graph_size) + ((current_vert) / (sizeof(block_t) * 8))] |= ((block_t)1 << ((current_vert) % (sizeof(block_t) * 8)));
                     break;
                }
            }

            if (j == max_color)
                (population[indiv_id])[rand() % max_color * TOTAL_BLOCK_NUM(graph_size) + ((current_vert) / (sizeof(block_t) * 8))] |= ((block_t)1 << ((current_vert) % (sizeof(block_t) * 8)));
        }
    }

    free(criteria);
    free(adjacent_colors);
    free(degrees);
}

bool read_graph(const char *filename, int graph_size, block_t *edges, int offset_i)
{
    FILE *fp = fopen(filename, "r");

    if (fp == nullptr)
        return false;

    char buffer[64];
    char *token = nullptr, *saveptr = nullptr;
    int row = 0, column = 0;
    while (fgets(buffer, 64, fp) != nullptr)
    {
        buffer[strcspn(buffer, "\n")] = 0;

#ifdef _WIN32
        token = strtok_s(buffer, " ", &saveptr);
#endif
#ifdef __linux__
        token = strtok_r(buffer, " ", &saveptr);
#endif
        if (saveptr[0] == 0)
            break;
        row = atoi(token) + offset_i;
#ifdef _WIN32
        token = strtok_s(nullptr, " ", &saveptr);
#endif
#ifdef __linux__
        token = strtok_r(nullptr, " ", &saveptr);
#endif
        column = atoi(token) + offset_i;

        (edges)[row * TOTAL_BLOCK_NUM(graph_size) + ((column) / (sizeof(block_t) * 8))] |= ((block_t)1 << ((column) % (sizeof(block_t) * 8)));
        (edges)[column * TOTAL_BLOCK_NUM(graph_size) + ((row) / (sizeof(block_t) * 8))] |= ((block_t)1 << ((row) % (sizeof(block_t) * 8)));
    }

    fclose(fp);
    return true;
}

bool read_weights(const char *filename, int graph_size, int *weights)
{
    FILE *fp = fopen(filename, "r");

    if (fp == nullptr)
        return false;

    char buffer[64];
    int vertex = 0;
    while (fgets(buffer, 64, fp) != nullptr && vertex < graph_size)
    {
        buffer[strcspn(buffer, "\n")] = 0;
        weights[vertex] = atoi(buffer);
        vertex++;
    }

    fclose(fp);
    return true;
}

bool is_valid(int graph_size, const block_t *edges, int color_num, const block_t *colors)
{
    // Iterate through vertices.
    int i, j, k, i_block;
    block_t i_mask;
    bool vertex_is_colored, error_flag = false, over_colored;
    for (i = 0; i < graph_size; i++)
    {
        vertex_is_colored = false;
        over_colored = false;
        i_block = BLOCK_INDEX(i);
        i_mask = MASK(i);

        // Iterate through colors and look for the vertex.
        for (j = 0; j < color_num; j++)
        {
            if ((colors[j * TOTAL_BLOCK_NUM(graph_size) + i_block] & i_mask))
            {
                if (!vertex_is_colored)
                {
                    vertex_is_colored = true;
                }
                else
                {
                    over_colored = true;
                }

                for (k = i + 1; k < graph_size; k++)
                { // Through every vertex after i in color j.
                    if (((colors)[j * TOTAL_BLOCK_NUM(graph_size) + ((k) / (sizeof(block_t) * 8))] & ((block_t)1 << ((k) % (sizeof(block_t) * 8)))) && ((edges)[k * TOTAL_BLOCK_NUM(graph_size) +i_block] & i_mask))
                    {
                        // The two vertices have the same color.
                        printf("The vertices %d and %d are connected and have the same color %d.\n", i, k, j);
                        error_flag = true;
                    }
                }
            }
        }

        // Check if the vertex had more then one color.
        if (!vertex_is_colored)
        {
            printf("The vertex %d has no color.\n", i);
            error_flag = true;
        }

        // Check if the vertex had more then one color.
        if (over_colored)
        {
            printf("The vertex %d has more than one color.\n", i);
            error_flag = true;
        }
    }

    return !error_flag;
}

int count_edges(int graph_size, const block_t *edges, int *degrees)
{
    int i, j, total = 0;
    for (i = 0; i < graph_size; i++)
    {
        for (j = 0; j < TOTAL_BLOCK_NUM(graph_size); j++)
        {
            degrees[i] += popcountl((edges)[i * TOTAL_BLOCK_NUM(graph_size) + j]);
        }
        total += degrees[i];
    }

    return total;
}

void print_colors(const char *filename, const char *header, int color_num, int graph_size, const block_t *colors)
{
    FILE *fresults;
    fresults = fopen(filename, "w");

    if (!fresults)
    {
        printf("%s\ncould not print results, aborting ...\n", strerror(errno));
        return;
    }

    fprintf(fresults, "%s\n\n", header);

    for (int i = 0; i < color_num; i++)
        for (int j = 0; j < graph_size; j++)
            if (((colors)[i * TOTAL_BLOCK_NUM(graph_size) + ((j) / (sizeof(block_t) * 8))] & ((block_t)1 << ((j) % (sizeof(block_t) * 8)))))
                fprintf(fresults, "%d %d\n", i, j);

    fclose(fresults);
}

bool exists(int *arr, int len, int target)
{
    for (int i = 0; i < len; i++)
        if (target == arr[i])
            return true;

    return false;
}

int graph_color_greedy(int graph_size, const block_t **edges, block_t **colors, int max_color_possible)
{
    // Go through the queue and color each vertex.
    int *prob_queue = (int *)malloc(graph_size * sizeof(int));
    block_t *adjacent_colors = (block_t *)malloc(TOTAL_BLOCK_NUM(max_color_possible) * sizeof(block_t));
    int max_color = 0, current_vert;
    int i, j, k;
    for (i = 0; i < graph_size; i++)
    {
        // Get a new random vertex.
        do
        {
            prob_queue[i] = rand() % graph_size;
        } while (exists(prob_queue, i, prob_queue[i]));
        current_vert = prob_queue[i];

        // Initialize the temporary data.
        memset(adjacent_colors, 0, (TOTAL_BLOCK_NUM(max_color_possible)) * sizeof(block_t));
        for (j = 0; j < TOTAL_BLOCK_NUM(graph_size); j++)
            for (k = 0; k < max_color_possible; k++)
                if ((edges[current_vert][j] & colors[k][j]))
                    SET_COLOR(adjacent_colors, k);

        // Find the first unused color (starting from 0) and assign this vertex to it.
        for (j = 0; j < max_color_possible; j++)
        {
            if (!CHECK_COLOR(adjacent_colors, j))
            {
                SET_COLOR(colors[j], current_vert);
                if (max_color < j)
                    max_color = j;
                break;
            }
        }
    }

    free(prob_queue);
    free(adjacent_colors);

    return max_color + 1;
}

int count_conflicts(int graph_size, const block_t *color, const block_t *edges, int *conflict_count)
{
    int i, j, total_conflicts = 0;
    for (i = 0; i < graph_size; i++)
    {
        if (CHECK_COLOR(color, i))
        {
            conflict_count[i] = 0;
            for (j = 0; j < TOTAL_BLOCK_NUM(graph_size); j++)
                conflict_count[i] += popcountl(color[j] & edges[i * TOTAL_BLOCK_NUM(graph_size) + j]);
            total_conflicts += conflict_count[i];
        }
    }

    return total_conflicts / 2;
}
