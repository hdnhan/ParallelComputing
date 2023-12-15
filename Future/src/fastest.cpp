// https://stackoverflow.com/questions/12086622/is-there-a-way-to-cancel-detach-a-future-in-c11
#include <algorithm>
#include <chrono>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

using namespace std;

int delay(int task_id) {
    int time = rand() % 50 + 1;
    std::cout << time << std::endl;
    this_thread::sleep_for(chrono::seconds(time));
    std::cout << "End task " << task_id << " in " << time << "s" << std::endl;
    return time;
}

void get_fastest_and_wait() {
    std::srand(std::time(NULL));

    int ntasks = 10;
    std::vector<std::future<int>> futures;
    for (int i = 0; i < ntasks; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        futures.emplace_back(std::async(std::launch::async, delay, 1));
    }

    // Wait for the first 6 tasks to finish
    int count = 0;
    std::vector<int> results;
    while (count < 6) {
        auto it = std::find_if(futures.begin(), futures.end(), [](std::future<int> const& future) {
            return future.valid() && future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        });
        if (it != futures.end()) {
            results.emplace_back(it->get());
            ++count;
        }
    }

    // Print the results
    std::cout << "Finished tasks: ";
    for (int result : results) std::cout << result << " ";
    std::cout << std::endl;

    int running = 0;
    for (auto& future : futures) {
        running += future.valid() && future.wait_for(std::chrono::seconds(0)) == std::future_status::timeout;
    }
    std::cout << running << " tasks..." << std::endl;

    // Cancel the remaining tasks
    /*
    for (auto& future : futures) {
        if (future.valid()) {
            future.wait();
        }
    }
    */

    /*
    Explain: Before exitting this function, all async functions need to be done first. So
    eventually, we can't cancel the remaining tasks, just wait for them finished.
    */
}

template <typename RESULT_TYPE, typename FUNCTION_TYPE>
std::future<RESULT_TYPE> startDetachedFuture(FUNCTION_TYPE const& func, int id) {
    std::promise<RESULT_TYPE> promise;
    std::future<RESULT_TYPE> future = promise.get_future();

    std::thread([&func, id](std::promise<RESULT_TYPE> p) { p.set_value(func(id)); }, std::move(promise))
        .detach();

    return future;
}

// Convenient macro to save boilerplate template code
#define START_DETACHED_FUTURE(func, id) startDetachedFuture<decltype(func(id)), decltype(func)>(func, id)
void get_fastest_and_ignore() {
    std::srand(std::time(NULL));

    int ntasks = 10;
    std::vector<std::future<int>> futures;
    for (int i = 0; i < ntasks; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        futures.emplace_back(START_DETACHED_FUTURE(delay, 2));
    }

    // Wait for the first 6 tasks to finish
    int count = 0;
    std::vector<int> results;
    while (count < 6) {
        auto it = std::find_if(futures.begin(), futures.end(), [](std::future<int> const& future) {
            return future.valid() && future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        });
        if (it != futures.end()) {
            results.emplace_back(it->get());
            ++count;
        }
    }

    // Print the results
    std::cout << "Finished tasks: ";
    for (int result : results) std::cout << result << " ";
    std::cout << std::endl;

    int running = 0;
    for (auto& future : futures) {
        running += future.valid() && future.wait_for(std::chrono::seconds(0)) == std::future_status::timeout;
    }
    std::cout << running << " tasks..." << std::endl;

    /*
    Explain: The remaining tasks will be ignored, even they are running in the background.
    */
}

int main() {
    std::cout << "\nGet six fastest tasks, four other tasks will be ignored"
              << "\n";
    get_fastest_and_ignore();
    get_fastest_and_wait();

    std::cout << "Exit main!" << std::endl;
    return 0;
}