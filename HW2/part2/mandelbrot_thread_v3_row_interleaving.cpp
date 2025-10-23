#include <array>
#include <cstdio>
#include <cstdlib>
#include <thread>

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

static __inline __attribute__((__gnu_inline__, __always_inline__))
int mandel(float c_re, float c_im, int count) {
    float z_re = c_re;
    float z_im = c_im;
    int i;
    for (i = 0; i < count; ++i) {
        float z_re2 = z_re * z_re;
        float z_im2 = z_im * z_im;

        if (z_re2 + z_im2 > 4.f) break;

        float new_re = z_re2 - z_im2;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

static __inline __attribute__((__gnu_inline__, __always_inline__))
void mandelbrot_serial(float x0,
                        float y0,
                        float x1,
                        float y1,
                        int width,
                        int height,
                        int max_iterations,
                        int *__restrict__ output,
                        int threadId,
                        int num_threads)
{
    float dx = (x1 - x0) / (float)width;
    float dy = (y1 - y0) / (float)height;

    // -------------------- 每 row 交錯執行 ---------------------------------
    for (int j = threadId; j < height; j += num_threads) {
        float y = y0 + ((float)j * dy);
        for (int i = 0; i < width; i++) {
            float x = x0 + ((float)i * dx);

            int index = ((j * width) + i);
            output[index] = mandel(x, y, max_iterations);
        }
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

    mandelbrot_serial(args->x0,
                      args->y0,
                      args->x1,
                      args->y1,
                      args->width,
                      args->height,
                      args->maxIterations,
                      args->output,
                      args->threadId,
                      args->numThreads);

    // printf("Hello world from thread %d\n", args->threadId);
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
