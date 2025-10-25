#include "page_rank.h"

#include <cmath>
#include <cstdlib>
#include <omp.h>

#include "../common/graph.h"

// page_rank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//

struct alignas(64) PaddedDouble {
    double value;
    char padding[64 - sizeof(double)];  // 填充到 64 bytes
};

void page_rank(Graph g, double *solution, double damping, double convergence)
{

    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs

    int nnodes = num_nodes(g);
    double equal_prob = 1.0 / nnodes;

    // 使用 aligned allocation
    PaddedDouble *solution_pad = (PaddedDouble*)malloc(nnodes * sizeof(PaddedDouble));
    PaddedDouble *score_new = (PaddedDouble*)malloc(nnodes * sizeof(PaddedDouble));
    // PaddedDouble *solution_pad = (PaddedDouble*)aligned_alloc(64, nnodes * sizeof(PaddedDouble));
    // PaddedDouble *score_new = (PaddedDouble*)aligned_alloc(64, nnodes * sizeof(PaddedDouble));
    
    
#pragma omp parallel for
    for (int i = 0; i < nnodes; ++i) {
        solution_pad[i].value = equal_prob;   // score_old
    }

    /*
       For PP students: Implement the page rank algorithm here.  You
       are expected to parallelize the algorithm using openMP.  Your
       solution may need to allocate (and free) temporary arrays.

       Basic page rank pseudocode is provided below to get you started:

       // initialization: see example code above
       score_old[vi] = 1/nnodes;

       while (!converged) {

         // compute score_new[vi] for all nodes vi:
         score_new[vi] = sum over all nodes vj reachable from incoming edges
                            { score_old[vj] / number of edges leaving vj  }
         score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / nnodes;

         score_new[vi] += sum over all nodes v in graph with no outgoing edges
                            { damping * score_old[v] / nnodes }

         // compute how much per-node scores have changed
         // quit once algorithm has converged

         global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
         converged = (global_diff < convergence)
       }
     */
    bool converged = false;
    double damping_factor = (1.0-damping) / nnodes;
    double residual_factor = damping / nnodes;

    while (!converged) {
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nnodes; i++) {
            double sum = 0.0;
            const Vertex *start = incoming_begin(g, i);
            const Vertex *end = incoming_end(g, i);
            for (const Vertex *v = start; v != end; v++) {
                sum += solution_pad[*v].value / outgoing_size(g, *v);
            }
            score_new[i].value = damping * sum + damping_factor;
        }
        
        double residual_p = 0.0;
        double global_diff = 0.0;
        
#pragma omp parallel reduction(+:residual_p, global_diff)
{
    #pragma omp for
        for (int i = 0; i < nnodes; i++) {
            if (outgoing_size(g, i) == 0) {
                residual_p += solution_pad[i].value * residual_factor;
            }
        }
            
    #pragma omp for
        for (int i = 0; i < nnodes; i++) {
            score_new[i].value += residual_p;
            global_diff += fabs(score_new[i].value - solution_pad[i].value);
            solution_pad[i].value = score_new[i].value;
        }
}

        converged = (global_diff < convergence);
    }
#pragma omp parallel for
    for (int i = 0; i < nnodes; i++) {
        solution[i] = solution_pad[i].value;
    }
    
    free(solution_pad);
    free(score_new);
}
