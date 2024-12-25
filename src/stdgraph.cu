#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include "stdgraph.h"

void graph_color_random(int graph_size, const block_t edges[][TOTAL_BLOCK_NUM(SIZE)], block_t colors[][TOTAL_BLOCK_NUM(SIZE)], int max_color)
{
    int index;
    for (int i = 0; i < SIZE; i++)
    {
        index = rand() % max_color;
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
int comp_crit_1(const void* a, const void* b, void* metrics) {
    int* weights = ((int**)metrics)[0];
    int* degrees = ((int**)metrics)[1];
    return (weights[*(int*)a] * degrees[*(int*)a]) - (weights[*(int*)b] * degrees[*(int*)b]);
}

int comp_crit_2(const void* a, const void* b, void* metrics) {
    int* weights = ((int**)metrics)[0];
    int* degrees = ((int**)metrics)[1];
    return (weights[*(int*)a] * degrees[*(int*)a] * degrees[*(int*)a]) - (weights[*(int*)b] * degrees[*(int*)b] * degrees[*(int*)b]);
}

int comp_crit_3(const void* a, const void* b, void* weights) {
    return (((int*)weights)[*(int*)a]) - (((int*)weights)[*(int*)b]);
}
#endif

int popcountl(uint64_t n)
{
    int cnt = 0;
    // printf("Popcount %ld\n", n);
    while (n)
    {
        n &= n - 1; // key point
        ++cnt;
    }
    return cnt;
}

void pop_complex_random(int graph_size, const block_t *edges, const int *weights, int pop_size, block_t **population, int max_color)
{
    block_t(*edges_p)[][TOTAL_BLOCK_NUM(SIZE)] = (block_t(*)[][TOTAL_BLOCK_NUM(SIZE)])edges;

    int criteria[3][SIZE];
    for (int i = 0; i < SIZE; i++)
    {
        criteria[0][i] = i;
        criteria[1][i] = i;
        criteria[2][i] = i;
    }

    int degrees[SIZE];
    count_edges(SIZE, edges, degrees);

    const int *metrics[2] = {weights, degrees};

    #ifdef _WIN32
        qsort_s(criteria[0], SIZE, sizeof(int), comp_crit_1, (void *)metrics);
        qsort_s(criteria[1], SIZE, sizeof(int), comp_crit_2, (void *)metrics);
        qsort_s(criteria[2], SIZE, sizeof(int), comp_crit_3, (void *)weights);
    #endif

    #ifdef __linux__
        qsort_r(criteria[0], SIZE, sizeof(int), comp_crit_1, (void *)metrics);
        qsort_r(criteria[1], SIZE, sizeof(int), comp_crit_2, (void *)metrics);
        qsort_r(criteria[2], SIZE, sizeof(int), comp_crit_3, (void *)weights);
    #endif

    // Go through the queue and color each vertex.
    block_t adjacent_colors[TOTAL_BLOCK_NUM(SIZE)];
    block_t(*indiv)[][TOTAL_BLOCK_NUM(SIZE)];
    int current_vert;
    int i, j, k;
    for (int indiv_id = 0; indiv_id < pop_size; indiv_id++)
    {
        indiv = (block_t(*)[][TOTAL_BLOCK_NUM(SIZE)])population[indiv_id];

        memset(indiv, 0, max_color * TOTAL_BLOCK_NUM(SIZE) * sizeof(block_t));

        for (i = 0; i < SIZE; i++)
        {
            if (indiv_id < 40)
                current_vert = criteria[0][i];
            else if (indiv_id < 80)
                current_vert = criteria[1][i];
            else
                current_vert = criteria[2][i];

            // Initialize the temporary data.
            memset(adjacent_colors, 0, (TOTAL_BLOCK_NUM(max_color)) * sizeof(block_t));
            for (j = 0; j < TOTAL_BLOCK_NUM(SIZE); j++)
            {
                for (k = 0; k < max_color; k++)
                {
                    // SET_COLOR(adjacent_colors, edges[current_vert][j] & (*indiv)[k][j] & 1);
                    if ((*edges_p)[current_vert][j] & (*indiv)[k][j])
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
                    SET_COLOR((*indiv)[j], current_vert);
                    break;
                }
            }

            if (j == max_color)
                SET_COLOR((*indiv)[rand() % max_color], current_vert);
        }
    }
}

bool read_graph(const char *filename, int graph_size, block_t *edges, int offset_i)
{
    block_t(*edges_p)[][TOTAL_BLOCK_NUM(SIZE)] = (block_t(*)[][TOTAL_BLOCK_NUM(SIZE)])edges;
    FILE *fp = fopen(filename, "r");

    if (fp == NULL)
        return false;

    memset(edges, 0, SIZE * TOTAL_BLOCK_NUM(SIZE) * sizeof(block_t));

    char buffer[64];
    char *token, *saveptr;
    int row, column;
    while (fgets(buffer, 64, fp) != NULL)
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
            token = strtok_s(NULL, " ", &saveptr);
        #endif
        #ifdef __linux__
            token = strtok_r(NULL, " ", &saveptr);
        #endif
        column = atoi(token) + offset_i;

        SET_EDGE(row, column, (*edges_p));
    }

    fclose(fp);
    return true;
}

bool read_weights(const char *filename, int graph_size, int weights[])
{
    FILE *fp = fopen(filename, "r");

    if (fp == NULL)
        return false;

    memset(weights, 0, SIZE * sizeof(int));

    char buffer[64];
    int vertex = 0;
    while (fgets(buffer, 64, fp) != NULL && vertex < SIZE)
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
    const block_t(*edges_p)[][TOTAL_BLOCK_NUM(SIZE)] = (block_t(*)[][TOTAL_BLOCK_NUM(SIZE)])edges;
    const block_t(*colors_p)[][TOTAL_BLOCK_NUM(SIZE)] = (block_t(*)[][TOTAL_BLOCK_NUM(SIZE)])colors;

    // Iterate through vertices.
    int i, j, k, i_block;
    block_t i_mask;
    bool vertex_is_colored, error_flag = false, over_colored;
    for (i = 0; i < SIZE; i++)
    {
        vertex_is_colored = false;
        over_colored = false;
        i_block = BLOCK_INDEX(i);
        i_mask = MASK(i);

        // Iterate through colors and look for the vertex.
        for (j = 0; j < color_num; j++)
        {
            if (((*colors_p)[j][i_block] & i_mask))
            {
                if (!vertex_is_colored)
                {
                    vertex_is_colored = true;
                }
                else
                {
                    over_colored = true;
                }

                for (k = i + 1; k < SIZE; k++)
                { // Through every vertex after i in color j.
                    if (CHECK_COLOR((*colors_p)[j], k) && ((*edges_p)[k][i_block] & i_mask))
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

int count_edges(int graph_size, const block_t *edges, int degrees[])
{
    const block_t(*edges_p)[][TOTAL_BLOCK_NUM(SIZE)] = (block_t(*)[][TOTAL_BLOCK_NUM(SIZE)])edges;
    memset(degrees, 0, SIZE * sizeof(int));

    int i, j, total = 0;
    for (i = 0; i < SIZE; i++)
    {
        for (j = 0; j < TOTAL_BLOCK_NUM(SIZE); j++)
            degrees[i] += popcountl((*edges_p)[i][j]);
        total += degrees[i];
    }

    return total;
}

void print_colors(const char *filename, const char *header, int color_num, int graph_size, const block_t *colors)
{
    const block_t(*colors_p)[][TOTAL_BLOCK_NUM(SIZE)] = (block_t(*)[][TOTAL_BLOCK_NUM(SIZE)])colors;

    FILE *fresults;
    fresults = fopen(filename, "w");

    if (!fresults)
    {
        printf("%s\ncould not print results, aborting ...\n", strerror(errno));
        return;
    }

    fprintf(fresults, "%s\n\n", header);

    for (int i = 0; i < color_num; i++)
        for (int j = 0; j < SIZE; j++)
            if (CHECK_COLOR((*colors_p)[i], j))
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

int graph_color_greedy(int graph_size, const block_t edges[][TOTAL_BLOCK_NUM(SIZE)], block_t colors[][TOTAL_BLOCK_NUM(SIZE)], int max_color_possible)
{
    // Go through the queue and color each vertex.
    int prob_queue[SIZE];
    block_t adjacent_colors[TOTAL_BLOCK_NUM(MAX_COLOR)];
    int max_color = 0, current_vert;
    int i, j, k;
    for (i = 0; i < SIZE; i++)
    {
        // Get a new random vertex.
        do
        {
            prob_queue[i] = rand() % SIZE;
        } while (exists(prob_queue, i, prob_queue[i]));
        current_vert = prob_queue[i];

        // Initialize the temporary data.
        memset(adjacent_colors, 0, (TOTAL_BLOCK_NUM(max_color_possible)) * sizeof(block_t));
        for (j = 0; j < TOTAL_BLOCK_NUM(SIZE); j++)
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

    return max_color + 1;
}

int count_conflicts(int graph_size, const block_t *color, const block_t *edges, int *conflict_count)
{
    block_t(*edges_p)[][TOTAL_BLOCK_NUM(SIZE)] = (block_t(*)[][TOTAL_BLOCK_NUM(SIZE)])edges;

    int i, j, total_conflicts = 0;
    for (i = 0; i < SIZE; i++)
    {
        if (CHECK_COLOR(color, i))
        {
            conflict_count[i] = 0;
            for (j = 0; j < TOTAL_BLOCK_NUM(SIZE); j++)
                conflict_count[i] += popcountl(color[j] & (*edges_p)[i][j]);
            total_conflicts += conflict_count[i];
        }
    }

    return total_conflicts / 2;
}
