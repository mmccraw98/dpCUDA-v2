#include "../../include/utils/thread_pool.h"
ThreadPool::ThreadPool(size_t num_threads)
    : stop(false)
{
    for (size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back([this] {
            worker_thread();
        });
    }
}

void ThreadPool::enqueue(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        tasks.emplace(std::move(task));
    }
    condition.notify_one();
}

void ThreadPool::worker_thread() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);

            // Wait until we have a task or we're stopping
            condition.wait(lock, [this] {
                return stop || !tasks.empty();
            });

            if (stop && tasks.empty()) {
                return; // No more tasks to run
            }

            // Pop next task
            task = std::move(tasks.front());
            tasks.pop();
        }

        // Execute outside of the lock
        task();
    }
}

size_t ThreadPool::getQueueSize() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    return tasks.size();
}

void ThreadPool::shutdown() {
    // Let user know if we still had tasks
    size_t queue_size = getQueueSize();
    if (queue_size > 0) {
        std::cout << "ThreadPool::shutdown: queue size was " << queue_size << std::endl;
    }

    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }

    // Wake up all threads
    condition.notify_all();

    // Join them
    for (std::thread& worker : workers) {
        worker.join();
    }
}

ThreadPool::~ThreadPool() {
    // Ensure a proper shutdown if user didn't call it explicitly
    if (!stop) {
        shutdown();
    }
}