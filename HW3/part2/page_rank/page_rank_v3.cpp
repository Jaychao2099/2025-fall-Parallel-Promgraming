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
void page_rank(Graph g, double *solution, double damping, double convergence)
{

    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs

    int nnodes = num_nodes(g);
    double equal_prob = 1.0 / nnodes;
#pragma omp parallel for
    for (int i = 0; i < nnodes; ++i)
    {
        solution[i] = equal_prob;   // score_old
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
    double *score_new = new double[nnodes];

    double damping_tmp = (1.0-damping) / nnodes;
    double residual_tmp = damping / nnodes;

    while (!converged) {
#pragma omp parallel for
        for (int i = 0; i < nnodes; i++) {
            double incoming_v_old_score_sum = 0.0;
            const Vertex *start = incoming_begin(g, i);
            const Vertex *end = incoming_end(g, i);
            for (const Vertex *v = start; v != end; v++) {
                incoming_v_old_score_sum += solution[*v] / outgoing_size(g, *v);
            }
            score_new[i] = incoming_v_old_score_sum * damping + damping_tmp;
        }
        
        double residual_p = 0.0;
        double global_diff = 0.0;
#pragma omp parallel
{
// #pragma omp parallel for reduction(+ : residual_p)
    #pragma omp for reduction(+ : residual_p)
        for (int i = 0; i < nnodes; i++) {
            if (outgoing_size(g, i) == 0) {
                residual_p += solution[i] * residual_tmp;
            }
        }
        
        // double global_diff = 0.0;
// #pragma omp parallel for reduction(+ : global_diff)
    #pragma omp for reduction(+ : global_diff)
        for (int i = 0; i < nnodes; i++) {
            score_new[i] += residual_p;
            global_diff += abs(score_new[i] - solution[i]);
            solution[i] = score_new[i];
        }
}

        converged = (global_diff < convergence);
    }
    delete score_new;
}
