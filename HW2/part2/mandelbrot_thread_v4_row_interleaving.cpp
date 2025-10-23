#include <array>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "cycle_timer.h"

struct WorkerArgs
{
    float x0, x1;
    float y0, y1;
    int width;
    int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
};

static __inline __attribute__((__gnu_inline__, __always_inline__))
int mandel(const float c_re, const float c_im, const int count) {
    float z_re = c_re;
    float z_im = c_im;
    int i;

    for (i = 0; i < count; ++i) {
        const float z_re2 = z_re * z_re;
        const float z_im2 = z_im * z_im;

        // if (z_re2 + z_im2 > 4.f) break;
        if (__builtin_expect((z_re2 + z_im2) > 4.0f, 0)) return i;

        const float new_re = z_re2 - z_im2;
        const float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

static __inline __attribute__((__gnu_inline__, __always_inline__))
void mandelbrot_serial(WorkerArgs *const __restrict__ args)
{
    const float x0 = args->x0;
    const float y0 = args->y0;
    const float x1 = args->x1;
    const float y1 = args->y1;
    const int width = args->width;
    const int height = args->height;
    const int max_iterations = args->maxIterations;
    int *__restrict__ output = args->output;
    const int threadId = args->threadId;
    const int num_threads = args->numThreads;
    
    const float dx = (x1 - x0) / (float)width;
    const float dy = (y1 - y0) / (float)height;

    // -------------------- 每 row 交錯執行 ---------------------------------
    int *out_row = output + threadId * width;
    int interval = num_threads * width;
    for (int j = threadId; j < height; j += num_threads) {
        const float y = y0 + ((float)j * dy);
        for (int i = 0; i < width; i++) {
            const float x = x0 + ((float)i * dx);
            
            out_row[i] = mandel(x, y, max_iterations);
        }
        out_row += interval;
    }
}

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

    mandelbrot_serial(args);

    // printf("Hello world from thread %d\n", args->threadId);
    
    double end_time = CycleTimer::current_seconds();
    
    printf("[worker_thread_start(%d)]:\t\t[%.3f] ms\n", args->threadId, (end_time - start_time) * 1000);
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

        args[i] = (WorkerArgs){x0, x1, y0, y1, width, height, max_iterations, output, i, num_threads};
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
