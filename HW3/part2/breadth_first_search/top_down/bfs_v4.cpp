#include "bfs.h"

#include <cstdlib>
#include <omp.h>

#include "../common/graph.h"

// #define VERBOSE

#ifdef VERBOSE
#include "../common/cycle_timer.h"
#include <stdio.h>
#endif // VERBOSE

constexpr int ROOT_NODE_ID = 0;
constexpr int NOT_VISITED_MARKER = -1;

void vertex_set_clear(VertexSet *list)
{
    list->count = 0;
}

void vertex_set_init(VertexSet *list, int count)
{
    list->max_vertices = count;
    list->vertices = new int[list->max_vertices];
    vertex_set_clear(list);
}

void vertex_set_destroy(VertexSet *list)
{
    delete[] list->vertices;
}

// #define LOCAL_QUEUE_SIZE 2048
// #define LOCAL_QUEUE_SIZE 4096
#define LOCAL_QUEUE_SIZE 8192

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(Graph g, VertexSet *frontier, VertexSet *new_frontier, int *distances)
{
    #pragma omp parallel
    {
        int local_queue[LOCAL_QUEUE_SIZE];
        int local_rear = 0;
        
    #pragma omp for schedule(static, 8) nowait
        for (int i = 0; i < frontier->count; i++) {
            const int node = frontier->vertices[i];
            const int start_edge = g->outgoing_starts[node];
            const int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];

            // attempt to add all neighbors to the new frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                const int outgoing = g->outgoing_edges[neighbor];

                if (distances[outgoing] == NOT_VISITED_MARKER) {
                    const int next_dist = distances[node] + 1;
                    if (__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, next_dist)) {
                        local_queue[local_rear++] = outgoing;   // local ++, local fetch, don't need atomic
                        
                        // local queue full
                        if (local_rear == LOCAL_QUEUE_SIZE) {
                            const int index = __sync_fetch_and_add(&new_frontier->count, LOCAL_QUEUE_SIZE);
                            for (int j = 0; j < LOCAL_QUEUE_SIZE; j++) {
                                new_frontier->vertices[index + j] = local_queue[j];
                            }
                            local_rear = 0;
                        }
                    }
                }
            }
        }

        // handle remaining queue entries
        if (local_rear > 0) {
            int index = __sync_fetch_and_add(&new_frontier->count, local_rear);
            for (int j = 0; j < local_rear; j++) {
                new_frontier->vertices[index + j] = local_queue[j];
            }
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
#pragma omp parallel for schedule(static, 8)
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::current_seconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::current_seconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    // free memory
    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}
