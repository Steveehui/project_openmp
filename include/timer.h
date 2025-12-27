#ifndef TIMER_H
#define TIMER_H

#include <chrono>

// High precision timer class using C++11 chrono
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    bool is_running;

public:
    Timer() : is_running(false) {}

    // Start timing
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        is_running = true;
    }

    // Stop timing
    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
        is_running = false;
    }

    // Get elapsed time (seconds)
    double elapsed() const {
        auto end = is_running ? std::chrono::high_resolution_clock::now() : end_time;
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_time);
        return duration.count() / 1000000.0;
    }

    // Get elapsed time (milliseconds)
    double elapsed_ms() const {
        return elapsed() * 1000.0;
    }
};

#endif // TIMER_H
