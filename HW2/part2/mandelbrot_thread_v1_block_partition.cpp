#include <array>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "cycle_timer.h"

struct WorkerArgs
{
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
};

extern void mandelbrot_serial(float x0,
                              float y0,
                              float x1,
                              float y1,
                              int width,
                              int height,
                              int start_row,
                              int num_rows,
                              int max_iterations,
                              int *output);

//
// worker_thread_start --
//
// Thread entrypoint.
void worker_thread_start(WorkerArgs *const args)
{

    // TODO FOR PP STUDENTS: Implement the body of the worker
    // thread here. Each thread could make a call to mandelbrot_serial()
    // to compute a part of the output image. For example, in a
    // program that uses two threads, thread 0 could compute the top
    // half of the image and thread 1 could compute the bottom half.
    // Of course, you can copy mandelbrot_serial() to this file and
    // modify it to pursue a better performance.
    double start_time = CycleTimer::current_seconds();
    
    int i = args->threadId;
    int height = args->height;
    int num_threads = args->numThreads;

    int height_per_thread = height / num_threads;
    int remain_height = height % num_threads;

    mandelbrot_serial(args->x0,
                      args->y0,
                      args->x1,
                      args->y1,
                      args->width,
                      height,
                      height_per_thread * i + (i < remain_height ? i : remain_height),
                      height_per_thread + (i < remain_height ? 1 : 0),
                      args->maxIterations,
                      args->output);

    // printf("Hello world from thread %d\n", args->threadId);

    double end_time = CycleTimer::current_seconds();
    
    printf("[worker_thread_start(%d)]:\t[%.3f] ms\n", i, (end_time - start_time) * 1000);
}

//
// mandelbrot_thread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrot_thread(int num_threads,
                       float x0,
                       float y0,
                       float x1,
                       float y1,
                       int width,
                       int height,
                       int max_iterations,
                       int *output)
{
    static constexpr int max_threads = 32;

    if (num_threads > max_threads)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", max_threads);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::array<std::thread, max_threads> workers;
    std::array<WorkerArgs, max_threads> args = {};

    for (int i = 0; i < num_threads; i++)
    {
        // TODO FOR PP STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread

        // args[i].x0 = x0;
        // args[i].y0 = y0;
        // args[i].x1 = x1;
        // args[i].y1 = y1;
        // args[i].width = width;
        // args[i].height = height;
        // args[i].maxIterations = max_iterations;
        // args[i].output = output;
        // args[i].threadId = i;
        // args[i].numThreads = num_threads;
        args[i] = (WorkerArgs){x0, x1, y0, y1, (uint32_t)width, (uint32_t)height, max_iterations, output, i, num_threads};
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i = 1; i < num_threads; i++)
    {
        workers[i] = std::thread(worker_thread_start, &args[i]);
    }

    worker_thread_start(&args[0]);

    // join worker threads
    for (int i = 1; i < num_threads; i++)
    {
        workers[i].join();
    }
}
