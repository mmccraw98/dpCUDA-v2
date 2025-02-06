#pragma once

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <vector>
#include <iostream>

/**
 * @brief A simple thread pool that runs tasks asynchronously
 */
class ThreadPool {
public:
    /**
     * @brief Constructor
     * @param num_threads Number of threads to spawn
     */
    ThreadPool(size_t num_threads);

    /**
     * @brief Destructor
     */
    ~ThreadPool();

    /**
     * @brief Enqueue a function task for execution
     */
    void enqueue(std::function<void()> task);

    /**
     * @brief Gracefully shutdown the pool, finishing queued tasks first
     */
    void shutdown();

    /**
     * @brief Get the size of the current task queue
     */
    size_t getQueueSize();

private:
    /**
     * @brief Worker thread function
     */
    void worker_thread();

    // Vector of worker threads
    std::vector<std::thread> workers;
    // Task queue
    std::queue<std::function<void()>> tasks;

    // Synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};