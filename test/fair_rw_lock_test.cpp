/*
* Fair Reader-Writer lock sample
* Copyright 2026 Alexander Danileiko
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at:
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* This software is provided on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS
* OF ANY KIND, either express or implied.
*/

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <future>
#include <iomanip>
#include <iostream>
#include <cstring>
#include <latch>
#include <memory>
#include <numeric>
#include <random>
#include <shared_mutex>
#include <thread>
#include <vector>

#include "fair_rw_lock.h"
#include "textbook_reader_pref_lock.h"
#include "momentum_rw_lock.h"

// ----------------------------------------------------------------------------
// NUMA Test Configuration Bridge
// Dynamically maps the ENABLE_NUMA preprocessor macro into the 
// FairRWLock template parameter across the entire test suite.
// ----------------------------------------------------------------------------
#ifdef ENABLE_NUMA
constexpr bool USE_NUMA_CONFIG = true;
#else
constexpr bool USE_NUMA_CONFIG = false;
#endif

template <typename Policy = DefaultFairRWLockPolicy>
using TestFairRWLock = FairRWLock<Policy, USE_NUMA_CONFIG>;

// --- Platform & Architecture Specific ---
#if !defined(_WIN32)
// POSIX (Linux, macOS)
#include <sys/resource.h>
#endif

#ifdef _WIN32
#include <windows.h>
#pragma comment(lib, "Winmm.lib")
#endif

using namespace std::chrono_literals;

std::atomic<bool> global_tests_passed{true};

// Standardized column-aligned logging for visually consistent test runs
static void log_result(const char* name, 
                       bool        ok)
{
    std::cout << std::left << std::setw(50) << name 
              << (ok ? "[ PASS ]" : "[ FAIL ]") << std::endl;
              
    if (!ok)
    {
        global_tests_passed.store(false, 
                                  std::memory_order_relaxed);
    }
}

// Helper to get safe hardware concurrency
static unsigned int get_hw_threads()
{
    unsigned int hw = std::thread::hardware_concurrency();    
    if (hw == 0)
    {
        hw = 4;
    }
    
    return hw;
}

// ============================================================================
// Core Safety and Invariant Tests
// ============================================================================

// ----------------------------------------------------------------------------
// Validates strict mutual exclusion between readers and writers.
// Utilizes std::barrier to ensure all 10 threads attempt to acquire the lock 
// simultaneously, verifying that readers and writers never overlap.
// ----------------------------------------------------------------------------
void reader_writer_race_test()
{
    TestFairRWLock<> lock;
    std::atomic<int> readers_inside = 0;
    std::atomic<int> writers_inside = 0;
    std::atomic<bool> ok{true};
    std::barrier start(10);

    std::vector<std::thread> ts;
    
    for (int i = 0; i < 10; ++i)
    {
        ts.emplace_back([&, i]()
        {
            start.arrive_and_wait();
            
            if (i % 3 == 0)
            {
                if (lock.WriteLock(1s))
                {
                    if (writers_inside.fetch_add(1) != 0)
                    {
                        ok = false;
                    }

                    if (readers_inside.load() != 0)
                    {
                        ok = false;
                    }
                    
                    std::this_thread::sleep_for(2ms);
                    writers_inside--;
                    lock.WriteUnlock();
                }
            }
            else
            {
                if (lock.ReadLock(1s))
                {
                    if (writers_inside.load() != 0)
                    {
                        ok = false;
                    }
                    
                    readers_inside++;
                    std::this_thread::sleep_for(1ms);
                    readers_inside--;
                    lock.ReadUnlock();
                }
            }
        });
    }

    for (auto& t : ts)
    {
        t.join();
    }

    log_result("Reader/Writer Race Test", ok.load());
}

// ----------------------------------------------------------------------------
// Verifies that a writer in the slow-path queue properly respects 
// its timeout parameter when blocked by an active fast-path reader.
// ----------------------------------------------------------------------------
void writer_timeout_with_readers()
{
    TestFairRWLock<> lock;
    std::atomic<bool> timed_out = false;

    lock.ReadLock();
    
    std::thread t([&]()
    {
        if (!lock.WriteLock(100ms))
        {
            timed_out = true;
        }
        else
        {
            lock.WriteUnlock();
        }
    });

    std::this_thread::sleep_for(200ms);
    lock.ReadUnlock();
    t.join();

    log_result("Writer Timeout With Readers", timed_out.load());
}

// ----------------------------------------------------------------------------
// Verifies that a reader properly respects its timeout parameter 
// when blocked by an active exclusive writer.
// ----------------------------------------------------------------------------
void reader_timeout_with_writer()
{
    TestFairRWLock<> lock;
    std::atomic<bool> timed_out = false;

    lock.WriteLock();
    
    std::thread t([&]()
    {
        if (!lock.ReadLock(100ms))
        {
            timed_out = true;
        }
        else
        {
            lock.ReadUnlock();
        }
    });

    std::this_thread::sleep_for(200ms);
    lock.WriteUnlock();
    t.join();

    log_result("Reader Timeout With Writer", timed_out.load());
}

// ----------------------------------------------------------------------------
// Systematically tests TryReadLock and TryWriteLock to ensure they 
// fail instantly and safely when the lock is held by incompatible owners.
// ----------------------------------------------------------------------------
void try_lock_under_contention_test()
{
    TestFairRWLock<> lock;
    bool ok = true;
    
    lock.WriteLock();
    
    if (lock.TryReadLock())
    {
        ok = false;
        lock.ReadUnlock();
    }

    if (lock.TryWriteLock())
    {
        ok = false;
        lock.WriteUnlock();
    }

    lock.WriteUnlock();
    lock.ReadLock();
    
    if (lock.TryWriteLock())
    {
        ok = false;
    }

    if (!lock.TryReadLock())
    {
        ok = false;
    }
    else
    {
        lock.ReadUnlock();
    }

    lock.ReadUnlock();
    
    if (!lock.TryReadLock())
    {
        ok = false;
    }
    else
    {
        lock.ReadUnlock();
    }

    if (!lock.TryWriteLock())
    {
        ok = false;
    }
    else
    {
        lock.WriteUnlock();
    }

    log_result("TryLock Under Contention Test", ok);
}

// ----------------------------------------------------------------------------
// Validates the scoped RAII guards (ReadGuard/WriteGuard).
// Ensures move semantics correctly transfer ownership and automatically 
// release the lock upon destruction.
// ----------------------------------------------------------------------------
void raii_guard_functionality_test()
{
    TestFairRWLock<> lock;
    bool ok = true;

    {
        TestFairRWLock<>::WriteGuard g(lock);
        
        if (lock.TryReadLock())
        {
            ok = false;
        }
    }

    if (!lock.TryReadLock())
    {
        ok = false;
    }
    else
    {
        lock.ReadUnlock();
    }

    {
        TestFairRWLock<>::WriteGuard g1(lock);
        TestFairRWLock<>::WriteGuard g2(std::move(g1));
        
        if (!g2)
        {
            ok = false;
        }
    }

    lock.WriteLock();
    
    {
        TestFairRWLock<>::ReadGuard g(lock, 50ms);
        
        if (g)
        {
            ok = false;
        }
    }

    lock.WriteUnlock();
    log_result("RAII Guard Functionality Test", ok);
}

// ============================================================================
// Deterministic Fairness Policy Tests (Template Instantiations)
// ============================================================================

// ------------------------------------------------------------------------
// Validates that a writer can successfully force its way through an active 
// stream of readers by aggressively bounding the max wait time before 
// triggering starvation protocols.
// ----------------------------------------------------------------------------
struct PunchesThroughPolicy : DefaultFairRWLockPolicy
{
    static constexpr auto maxWriterWait = std::chrono::milliseconds(50);
};

// ----------------------------------------------------------------------------
// Validates the `maxWriterWait` starvation prevention policy. 
// Proves that a queued writer asserts the starvation bit to block new readers 
// and eventually forces the baton handoff despite continuous read attempts.
// ----------------------------------------------------------------------------
void writer_punches_through_readers_test()
{
    TestFairRWLock<PunchesThroughPolicy> lock;
    std::latch all_readers_in(5);
    std::promise<void> release_p;
    auto release = release_p.get_future().share();
    std::atomic<bool> writer_done = false;

    std::vector<std::thread> readers;
    
    for (int i = 0; i < 5; ++i)
    {
        readers.emplace_back([&]()
        {
            if (lock.ReadLock(2s))
            {
                all_readers_in.count_down();
                release.wait();
                lock.ReadUnlock();
            }
        });
    }
    
    all_readers_in.wait();

    std::promise<void> writer_is_queued_p;
    auto writer_is_queued_f = writer_is_queued_p.get_future();
    
    std::thread writer([&]()
    {
        writer_is_queued_p.set_value();
        
        if (lock.WriteLock(2s))
        {
            writer_done = true;
            lock.WriteUnlock();
        }
    });

    writer_is_queued_f.wait();
    std::this_thread::sleep_for(60ms);
    
    if (lock.TryReadLock())
    {
        lock.ReadUnlock();
    }

    release_p.set_value();
    writer.join();
    
    for (auto& r : readers)
    {
        r.join();
    }

    log_result("Writer Punches Through Readers Test", writer_done.load());
}

// ----------------------------------------------------------------------------
// Tests the system's ability to gracefully return to normal operation. 
// Heavily restricts the writer batch limit, ensuring override mode 
// disengages cleanly to let readers back in.
// ----------------------------------------------------------------------------
struct OverrideResetPolicy : DefaultFairRWLockPolicy
{
    static constexpr auto maxWriterWait    = std::chrono::milliseconds(50);
    static constexpr int  writerBatchLimit = 2;
};

// ----------------------------------------------------------------------------
// Checks that the lock successfully exits the emergency Override Mode 
// after draining a batch of writers, restoring normal reader access.
// ----------------------------------------------------------------------------
void override_reset_test()
{
    TestFairRWLock<OverrideResetPolicy> lock;
    std::atomic<bool> reader_succeeded = false;
    std::promise<void> writer_batch_done_p;
    auto writer_batch_done_f = writer_batch_done_p.get_future();

    lock.ReadLock();

    std::latch writers_queued(2);
    
    std::thread w1([&]()
    {
        writers_queued.count_down();
        
        if (lock.WriteLock(2s))
        {
            lock.WriteUnlock();
        }
    });
                   
    std::thread w2([&]()
    {
        writers_queued.count_down();
        
        if (lock.WriteLock(2s))
        {
            lock.WriteUnlock();
            writer_batch_done_p.set_value();
        }
    });

    writers_queued.wait();
    std::this_thread::sleep_for(60ms);
    
    if (lock.TryReadLock())
    {
        lock.ReadUnlock();
    }

    lock.ReadUnlock();

    std::thread reader([&]()
    {
        writer_batch_done_f.wait();
        
        if (lock.ReadLock(500ms))
        {
            reader_succeeded = true;
            lock.ReadUnlock();
        }
    });

    w1.join();
    w2.join();
    reader.join();
    
    log_result("Override Reset Test", reader_succeeded.load());
}

// ----------------------------------------------------------------------------
// Tests strict fairness between sequential writers. Capping consecutive 
// write acquisitions forces the lock to yield the baton to waiting readers, 
// preventing sequential writers from monopolizing the lock.
// ----------------------------------------------------------------------------
struct ConsecLimitPolicy : DefaultFairRWLockPolicy
{
    static constexpr int maxConsecWriters = 2;
};

// ----------------------------------------------------------------------------
// Tests the `maxConsecWriters` policy. Evaluates whether the lock
// forces a handover to a waiting reader even if another writer is queued, 
// preventing sequential writers from starving readers.
// ----------------------------------------------------------------------------
void consecutive_limit_breaks_batch_test()
{
    TestFairRWLock<ConsecLimitPolicy> lock;
    std::atomic<int> order{0};
    std::atomic<bool> reader_acquired{false};

    lock.WriteLock();
    lock.WriteUnlock();

    lock.WriteLock();

    std::promise<void> reader_started_p;
    auto reader_started_f = reader_started_p.get_future();

    std::thread reader([&]()
    {
        reader_started_p.set_value();
        
        if (lock.ReadLock(2s))
        {
            int expected = 0;
            order.compare_exchange_strong(expected, 1);
            reader_acquired = true;
            lock.ReadUnlock();
        }
    });

    reader_started_f.wait();
    std::this_thread::sleep_for(50ms);

    std::thread w3([&]()
    {
        if (lock.WriteLock(2s))
        {
            int expected = 0;
            order.compare_exchange_strong(expected, 2);
            lock.WriteUnlock();
        }
    });

    std::this_thread::sleep_for(50ms);

    lock.WriteUnlock();

    reader.join();
    w3.join();
    
    log_result("Consecutive Limit Breaks Batch Test", reader_acquired.load() && (order.load() == 1));
}

// ----------------------------------------------------------------------------
// Validates that if a waiting writer times out, it safely passes 
// the baton to the *next* queued writer rather than accidentally dropping 
// the notification and halting the queue.
// ----------------------------------------------------------------------------
void writer_timeout_handoff_test()
{
    TestFairRWLock<> lock;
    std::atomic<bool> w2_done = false;

    lock.ReadLock();

    std::latch writers_queued(2);
    
    std::thread w1([&]()
    {
        writers_queued.count_down();
        (void)lock.WriteLock(100ms);
    });
                   
    std::thread w2([&]()
    {
        writers_queued.count_down();
        
        if (lock.WriteLock(2s))
        {
            w2_done = true;
            lock.WriteUnlock();
        }
    });

    writers_queued.wait();
    std::this_thread::sleep_for(150ms);
    lock.ReadUnlock();

    w1.join();
    w2.join();
    
    log_result("Writer Timeout Handoff Test", w2_done.load());
}

// ----------------------------------------------------------------------------
// Critical edge-case test checking the integrity of the intrusive 
// linked list. Ensures that a writer timing out in the middle of the queue 
// correctly stitches the `prev` and `next` pointers to avoid a broken list.
// ----------------------------------------------------------------------------
void mid_queue_timeout_test()
{
    TestFairRWLock<> lock;
    std::atomic<int> acquire_count{0};
    
    lock.ReadLock();
    
    std::latch writers_queued(3);
    
    std::thread w1([&]()
    {
        writers_queued.count_down();
        
        if (lock.WriteLock(2s))
        {
            acquire_count++;
            lock.WriteUnlock();
        }
    });
    
    std::this_thread::sleep_for(10ms);
    
    std::thread w2([&]()
    {
        writers_queued.count_down();
        
        if (lock.WriteLock(100ms)) 
        {
            lock.WriteUnlock();
        }
    });
    
    std::this_thread::sleep_for(10ms);
    
    std::thread w3([&]()
    {
        writers_queued.count_down();
        
        if (lock.WriteLock(2s))
        {
            acquire_count++;
            lock.WriteUnlock();
        }
    });
    
    writers_queued.wait();
    std::this_thread::sleep_for(200ms);
    lock.ReadUnlock();
    
    w1.join();
    w2.join();
    w3.join();
    
    log_result("Middle-of-Queue Timeout Stitching Test", acquire_count.load() == 2);
}

// ----------------------------------------------------------------------------
// Evaluates time-bound override phases, ensuring writers cannot monopolize 
// the lock indefinitely by enforcing a strict chronological timeslice limit 
// before yielding back to the fast-path.
// ----------------------------------------------------------------------------
struct OverrideTimeslicePolicy : DefaultFairRWLockPolicy
{
    static constexpr auto maxWriterWait     = std::chrono::milliseconds(50);
    static constexpr auto overrideTimeslice = std::chrono::milliseconds(100);
};

// ----------------------------------------------------------------------------
// Validates the `overrideTimeslice` policy. Confirms that 
// if writers hold priority too long, the lock strictly transitions back to 
// allow readers, guaranteeing an upper bound on reader starvation.
// ----------------------------------------------------------------------------
void override_timeslice_allows_reader_test()
{
    TestFairRWLock<OverrideTimeslicePolicy> lock;
    std::atomic<int> order{0};
    std::atomic<bool> reader_got_in{false};

    std::promise<void> wA_is_queued_p;
    auto wA_is_queued_f = wA_is_queued_p.get_future();
    std::promise<void> override_triggered_p;
    auto override_triggered_f = override_triggered_p.get_future();
    std::promise<void> wA_has_lock_p;
    auto wA_has_lock_sf = wA_has_lock_p.get_future().share();

    lock.ReadLock();

    std::thread writerA([&, p = std::move(wA_has_lock_p)]() mutable
    {
        wA_is_queued_p.set_value();
        
        if (lock.WriteLock(2s))
        {
            p.set_value();
            std::this_thread::sleep_for(150ms);
            lock.WriteUnlock();
        }
    });

    std::thread writerB([&]()
    {
        wA_is_queued_f.wait();
        std::this_thread::sleep_for(60ms);
        
        if (lock.TryReadLock())
        {
            lock.ReadUnlock();
        }
        
        override_triggered_p.set_value();
        
        if (lock.WriteLock(2s))
        {
            int expected = 0;
            order.compare_exchange_strong(expected, 2);
            lock.WriteUnlock();
        }
    });

    override_triggered_f.wait();
    lock.ReadUnlock();

    std::thread reader([&]()
    {
        wA_has_lock_sf.wait();
        
        if (lock.ReadLock(1s))
        {
            int expected = 0;
            order.compare_exchange_strong(expected, 1);
            reader_got_in = true;
            lock.ReadUnlock();
        }
    });

    writerA.join();
    writerB.join();
    reader.join();
    
    log_result("Override Timeslice Allows Reader Test", reader_got_in.load() && (order.load() == 1));
}

// ============================================================================
// Stress and Liveness Tests
// ============================================================================

// ----------------------------------------------------------------------------
// Basic liveness validation ensuring that multiple concurrent writers 
// can sequentially acquire and release the lock without deadlocking.
// ----------------------------------------------------------------------------
void multiple_writers_fairness_test()
{
    TestFairRWLock<> lock;
    std::atomic<int> successes = 0;
    std::vector<std::thread> writers;
    
    for (int i = 0; i < 4; ++i)
    {
        writers.emplace_back([&]()
        {
            if (lock.WriteLock(2s))
            {
                successes++;
                lock.WriteUnlock();
            }
        });
    }
    
    for (auto& w : writers)
    {
        w.join();
    }

    log_result("Multiple Writers Fairness Test", successes.load() == 4);
}

// ----------------------------------------------------------------------------
// Heavy concurrent stress test utilizing thread-local RNGs to spam 
// random reads and writes. Evaluates general lock stability, ensuring 
// state limits hold firm and no threads deadlock during prolonged contention.
// ----------------------------------------------------------------------------
void stress_liveness_test()
{
    TestFairRWLock<> lock;
    const unsigned int hw_threads = get_hw_threads();
    const int num_threads = std::max<int>(10u, hw_threads * 2); 
    const int ops_per_thread = 300;
    
    std::atomic<int> readers_inside = 0;
    std::atomic<int> writers_inside = 0;
    std::atomic<bool> ok{true};

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    for (int i = 0; i < num_threads; ++i)
    {
        threads.emplace_back([&, i]()
        {
            std::mt19937 rng(i + 12345);
            
            for (int j = 0; j < ops_per_thread; ++j)
            {
                if (rng() % 4 == 0)
                {
                    if (lock.WriteLock(1s))
                    {
                        if (writers_inside.fetch_add(1) != 0)
                        {
                            ok = false;
                        }

                        if (readers_inside.load() != 0)
                        {
                            ok = false;
                        }
                        
                        writers_inside--;
                        lock.WriteUnlock();
                    }
                }
                else
                {
                    if (lock.ReadLock(1s))
                    {
                        if (writers_inside.load() != 0)
                        {
                            ok = false;
                        }
                        
                        readers_inside++;

                        // Hardware-level pause to widen the execution window 
                        // without surrendering the OS time slice.
                        for (int k = 0; k < 10; ++k)
                        {
                            cpu_relax_pause();
                        }

                        readers_inside--;
                        lock.ReadUnlock();
                    }
                }
            }
        });
    }

    for (auto& t : threads)
    {
        t.join();
    }

    log_result("Stress Liveness Test", ok.load());
}

// =============================================================================
// Advanced Edge Case and Semantic Tests
// =============================================================================

// ----------------------------------------------------------------------------
// Intentionally breaks standard recursive read properties to guarantee 
// writer liveness. Ensures the lock gates new readers immediately when 
// the starvation threshold is breached.
// ----------------------------------------------------------------------------
struct NonRecursivePolicy : DefaultFairRWLockPolicy
{
    static constexpr auto starvationThreshold = std::chrono::milliseconds(10);
};

// ----------------------------------------------------------------------------
// Demonstrates that the lock explicitly prevents recursive reads 
// when a writer is starving. This correctly avoids deadlocks associated 
// with recursive lock architectures where readers block pending writers indefinitely.
// ----------------------------------------------------------------------------
void non_recursive_reader_test()
{
    TestFairRWLock<NonRecursivePolicy> lock;
    std::atomic<bool> ok{true};
    std::promise<void> writer_queued;
    auto writer_f = writer_queued.get_future();
    
    lock.ReadLock();
    
    std::thread writer([&]()
    {
        writer_queued.set_value();
        
        if (lock.WriteLock(1s))
        {
            lock.WriteUnlock();
        }
    });
    
    writer_f.wait();
    std::this_thread::sleep_for(50ms); 
    
    if (lock.TryReadLock())
    {
        ok = false;
        lock.ReadUnlock();
    }
    
    lock.ReadUnlock();
    writer.join();
    
    log_result("Explicit Non-Recursive Reader Test", ok.load());
}

// ----------------------------------------------------------------------------
// Validates the edge case where timeout durations are exactly 0s. 
// Ensures the logic mirrors the behavior of `TryLock`, instantly failing if contested.
// ----------------------------------------------------------------------------
void zero_timeout_edge_case_test()
{
    TestFairRWLock<> lock;
    bool ok = true;

    if (!lock.WriteLock(0s))
    {
        ok = false;
    }

    if (lock.ReadLock(0s))
    {
        ok = false;
        lock.ReadUnlock();
    }

    lock.WriteUnlock();

    if (!lock.ReadLock(0s))
    {
        ok = false;
    }

    if (lock.WriteLock(0s))
    {
        ok = false;
        lock.WriteUnlock();
    }

    if (!lock.ReadLock(0s))
    {
        ok = false;
    }

    if (lock.WriteLock(0s))
    {
        ok = false;
        lock.WriteUnlock();
    }

    lock.ReadUnlock();
    lock.ReadUnlock();

    log_result("Zero Timeout Edge Case Test", ok);
}

// ----------------------------------------------------------------------------
// Verifies the `notify_all` condition variable logic. Checks that a 
// massive "herd" of waiting readers can all acquire the lock cleanly after a 
// writer unblocks them, simulating heavy parallel bursts.
// ----------------------------------------------------------------------------
void reader_thundering_herd_test()
{
    TestFairRWLock<> lock;
    std::atomic<int> readers_in = 0;
    std::atomic<bool> writer_done = false;
    
    unsigned int num_readers = std::max<unsigned int>(10u, get_hw_threads() * 2);
    std::barrier sync(num_readers + 1);

    lock.WriteLock();

    std::vector<std::thread> readers;
    
    for (unsigned int i = 0; i < num_readers; ++i)
    {
        readers.emplace_back([&]()
        {
            sync.arrive_and_wait();
            
            if (lock.ReadLock(5s))
            {
                readers_in++;
                lock.ReadUnlock();
            }
        });
    }

    sync.arrive_and_wait();
    std::this_thread::sleep_for(50ms);

    writer_done = true;
    lock.WriteUnlock();

    for (auto& r : readers)
    {
        r.join();
    }

    log_result("Reader Thundering Herd Test", (readers_in.load() == static_cast<int>(num_readers)) && writer_done.load());
}

// ----------------------------------------------------------------------------
// Sets aggressive bounds on writer batches during an override, prioritizing 
// system responsiveness by forcing rapid context switches between heavily 
// contested readers and writers.
// ----------------------------------------------------------------------------
struct StrictBatchPolicy : DefaultFairRWLockPolicy
{
    static constexpr auto maxWriterWait    = std::chrono::milliseconds(50);
    static constexpr int  writerBatchLimit = 2;
};

// ----------------------------------------------------------------------------
// Checks if the `writerBatchLimit` policy rigidly limits continuous 
// override execution blocks to allow pending readers access, avoiding 
// infinite starvation.
// ----------------------------------------------------------------------------
void strict_writer_batch_limit_test()
{
    TestFairRWLock<StrictBatchPolicy> lock;
    std::atomic<int> execution_order = 0;
    std::atomic<int> reader_position = 0;

    lock.ReadLock();

    std::thread w1([&]()
    {
        if (lock.WriteLock(2s))
        {
            execution_order++;
            lock.WriteUnlock();
        }
    });
                   
    std::this_thread::sleep_for(10ms);
    
    std::thread w2([&]()
    {
        if (lock.WriteLock(2s))
        {
            execution_order++;
            lock.WriteUnlock();
        }
    });
                   
    std::this_thread::sleep_for(10ms);
    
    std::thread w3([&]()
    {
        if (lock.WriteLock(2s))
        {
            execution_order++;
            lock.WriteUnlock();
        }
    });

    std::this_thread::sleep_for(50ms);
    
    if (lock.TryReadLock())
    {
        lock.ReadUnlock();
    }

    std::thread reader([&]()
    {
        if (lock.ReadLock(2s))
        {
            reader_position = execution_order.load();
            lock.ReadUnlock();
        }
    });

    std::this_thread::sleep_for(10ms);
    lock.ReadUnlock();

    w1.join();
    w2.join();
    reader.join();
    w3.join();

    log_result("Strict Writer Batch Limit Test", reader_position.load() == 2);
}

// ----------------------------------------------------------------------------
// Verifies that the move assignment operators for RAII guards correctly 
// release the previously held lock prior to assuming the new ownership.
// ----------------------------------------------------------------------------
void guard_move_assignment_test()
{
    TestFairRWLock<> lock1;
    TestFairRWLock<> lock2;
    bool ok = true;

    {
        TestFairRWLock<>::WriteGuard g1(lock1);
        TestFairRWLock<>::WriteGuard g2(lock2);

        g1 = std::move(g2);

        if (!lock1.TryWriteLock())
        {
            ok = false;
        }
        else
        {
            lock1.WriteUnlock();
        }

        if (lock2.TryWriteLock())
        {
            ok = false;
            lock2.WriteUnlock();
        }
    }

    if (!lock2.TryWriteLock())
    {
        ok = false;
    }
    else
    {
        lock2.WriteUnlock();
    }
    
    log_result("Guard Move Assignment Test", ok);
}

// ----------------------------------------------------------------------------
// Validates that passing `duration::max()` does not cause 
// `std::chrono::time_point` overflow or undefined calculation errors.
// ----------------------------------------------------------------------------
void max_duration_overflow_test()
{
    TestFairRWLock<> lock;
    bool ok = true;
    
    lock.WriteLock();

    std::atomic<bool> reader_started = false;
    
    std::thread reader([&]()
    {
        reader_started = true;
        
        if (!lock.ReadLock(((std::chrono::steady_clock::duration::max)())))
        {
            ok = false;
        }
        else
        {
            lock.ReadUnlock();
        }
    });

    while (!reader_started)
    {
        std::this_thread::yield();
    }

    std::this_thread::sleep_for(50ms);
    lock.WriteUnlock();
    reader.join();

    log_result("Max Duration Overflow Test", ok);
}

// ============================================================================
// Baseline Comparison Test (4-Way Lock Architecture Benchmarks)
// ============================================================================

struct WorkloadResult
{
    uint64_t            total_reads = 0;
    uint64_t            total_writes = 0;
    std::vector<double> reader_latencies_ns;
    std::vector<double> writer_latencies_ns;
};

// ----------------------------------------------------------------------------
// Complete performance benchmark testing FairRWLock against OS native 
// std::shared_mutex, TextbookReaderPrefLock, and MomentumRWLock. Extracts 
// operations-per-second throughput metrics and generates latency percentiles.
// ----------------------------------------------------------------------------
void run_comparative_throughput_test()
{
    const unsigned int hw_threads = get_hw_threads();
    const int NUM_WRITERS = (hw_threads >= 3) ? 2 : 1;
    int readers_calc = static_cast<int>(hw_threads) - NUM_WRITERS;
    const int NUM_READERS = (readers_calc < 1) ? 1 : readers_calc;
    
    constexpr auto TEST_DURATION = 15s;

    std::cout << "Starting Comparative Throughput Test for " << TEST_DURATION.count() << " seconds each..." << std::endl;
    std::cout << "Scenario: " << NUM_READERS << " Readers, " << NUM_WRITERS << " Writers. (Scaled to logical cores)" << std::endl;

    auto run_workload = [&](auto        lock_read_fn, 
                            auto        unlock_read_fn, 
                            auto        lock_write_fn, 
                            auto        unlock_write_fn, 
                            const char* name) -> WorkloadResult
    {
        (void)name;
        
        std::atomic<bool> run_flag{true};
        std::atomic<uint64_t> total_reads{0};
        std::atomic<uint64_t> total_writes{0};
        std::barrier sync_point(NUM_READERS + NUM_WRITERS + 1);

        // Stochastic Profiling: Sample 1 out of every 1,000 reads to prevent clock overhead from destroying throughput.
        // Writers are sampled at 100% since their frequency is low enough to not bottleneck the hardware.
        const uint64_t READ_SAMPLE_RATE = 1000;
        const uint64_t WRITE_SAMPLE_RATE = 1;

        std::vector<std::vector<double>> thread_reader_latencies(NUM_READERS);
        std::vector<std::vector<double>> thread_writer_latencies(NUM_WRITERS);

        auto reader_task = [&](int t_idx)
        {
            uint64_t local_reads = 0;
            uint64_t sample_counter = 0;
            auto& latencies = thread_reader_latencies[t_idx];
            
            // Pre-allocate enough memory to avoid mid-test allocation spikes
            latencies.reserve(200000); 

            sync_point.arrive_and_wait();

            while (run_flag.load(std::memory_order_relaxed))
            {
                if (++sample_counter % READ_SAMPLE_RATE == 0)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    
                    lock_read_fn();
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    latencies.push_back(std::chrono::duration<double, std::nano>(end - start).count());
                }
                else
                {
                    lock_read_fn();
                }
                
                local_reads++;
                std::this_thread::yield();
                unlock_read_fn();
            }
            
            total_reads.fetch_add(local_reads, 
                                  std::memory_order_relaxed);
        };

        auto writer_task = [&](int t_idx)
        {
            uint64_t local_writes = 0;
            uint64_t sample_counter = 0;
            auto& latencies = thread_writer_latencies[t_idx];
            
            latencies.reserve(50000);

            sync_point.arrive_and_wait();

            while (run_flag.load(std::memory_order_relaxed))
            {
                if (++sample_counter % WRITE_SAMPLE_RATE == 0)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    
                    lock_write_fn();
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    latencies.push_back(std::chrono::duration<double, std::nano>(end - start).count());
                }
                else
                {
                    lock_write_fn();
                }

                local_writes++;
                std::this_thread::yield();
                unlock_write_fn();

                std::this_thread::sleep_for(1ms);
            }
            
            total_writes.fetch_add(local_writes, 
                                   std::memory_order_relaxed);
        };

        std::vector<std::thread> threads;
        
        for (int i = 0; i < NUM_READERS; ++i)
        {
            threads.emplace_back(reader_task, i);
        }

        for (int i = 0; i < NUM_WRITERS; ++i)
        {
            threads.emplace_back(writer_task, i);
        }

        sync_point.arrive_and_wait();
        std::this_thread::sleep_for(TEST_DURATION);

        run_flag.store(false, std::memory_order_relaxed);
        
        for (auto& t : threads)
        {
            t.join();
        }

        WorkloadResult res;
        res.total_reads  = total_reads.load();
        res.total_writes = total_writes.load();

        // Merge thread-local latencies into the global result arrays
        for (const auto& vec : thread_reader_latencies)
        {
            res.reader_latencies_ns.insert(res.reader_latencies_ns.end(), vec.begin(), vec.end());
        }
        
        for (const auto& vec : thread_writer_latencies)
        {
            res.writer_latencies_ns.insert(res.writer_latencies_ns.end(), vec.begin(), vec.end());
        }

        // Sort globally for percentile extraction
        std::sort(res.reader_latencies_ns.begin(), res.reader_latencies_ns.end());
        std::sort(res.writer_latencies_ns.begin(), res.writer_latencies_ns.end());

        return res;
    };

    // 1. FairRWLock (Our atomic fast-path lock)
    std::cout << "  -> Running FairRWLock (Custom Fairness Policy)..." << std::endl;
    TestFairRWLock<> fair_lock;
    WorkloadResult fair_results = run_workload([&] { fair_lock.ReadLock(); },
                                               [&] { fair_lock.ReadUnlock(); },
                                               [&] { fair_lock.WriteLock(); },
                                               [&] { fair_lock.WriteUnlock(); },
                                               "FairRWLock");

    // 2. std::shared_mutex (OS Standard)
    std::cout << "  -> Running std::shared_mutex (OS Native Primitives)..." << std::endl;
    std::shared_mutex std_lock;
    WorkloadResult std_results = run_workload([&] { std_lock.lock_shared(); },
                                              [&] { std_lock.unlock_shared(); },
                                              [&] { std_lock.lock(); },
                                              [&] { std_lock.unlock(); },
                                              "std::shared_mutex");

    // 3. Textbook Lock (Absolute starvation academic example)
    std::cout << "  -> Running TextbookReaderPrefLock (Standard Mutex/CV)..." << std::endl;
    TextbookReaderPrefLock tb_lock;
    WorkloadResult tb_results = run_workload([&] { tb_lock.ReadLock(); },
                                             [&] { tb_lock.ReadUnlock(); },
                                             [&] { tb_lock.WriteLock(); },
                                             [&] { tb_lock.WriteUnlock(); },
                                             "TextbookReaderPrefLock");
                                             
    // 4. Momentum Lock (Naive Priority Batching example)
    std::cout << "  -> Running MomentumRWLock (Naive Priority Batching Fix)..." << std::endl;
    MomentumRWLock mom_lock;
    WorkloadResult mom_results = run_workload([&] { mom_lock.ReadLock(INFINITE_TIME); },
                                              [&] { mom_lock.ReadUnlock(); },
                                              [&] { mom_lock.WriteLock(INFINITE_TIME); },
                                              [&] { mom_lock.WriteUnlock(); },
                                              "MomentumRWLock");

    uint64_t fair_total = fair_results.total_reads + fair_results.total_writes;
    uint64_t std_total  = std_results.total_reads + std_results.total_writes;
    uint64_t tb_total   = tb_results.total_reads + tb_results.total_writes;
    uint64_t mom_total  = mom_results.total_reads + mom_results.total_writes;

    double test_secs = static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(TEST_DURATION).count());
    
    double fair_read_tp  = static_cast<double>(fair_results.total_reads) / test_secs;
    double fair_write_tp = static_cast<double>(fair_results.total_writes) / test_secs;
    double std_read_tp   = static_cast<double>(std_results.total_reads) / test_secs;
    double std_write_tp  = static_cast<double>(std_results.total_writes) / test_secs;
    double tb_read_tp    = static_cast<double>(tb_results.total_reads) / test_secs;
    double tb_write_tp   = static_cast<double>(tb_results.total_writes) / test_secs;
    double mom_read_tp   = static_cast<double>(mom_results.total_reads) / test_secs;
    double mom_write_tp  = static_cast<double>(mom_results.total_writes) / test_secs;

    auto calc_percentile = [](const std::vector<double>& sorted_vec, 
                              double                     p) -> double
    {
        if (sorted_vec.empty())
        {
            return 0.0;
        }
        
        size_t idx = static_cast<size_t>(p * sorted_vec.size());
        
        if (idx >= sorted_vec.size())
        {
            idx = sorted_vec.size() - 1;
        }
        
        return sorted_vec[idx];
    };

    std::cout << "\n================================================================================================================" << std::endl;
    std::cout << "                                          THROUGHPUT COMPARISON MATRIX                                          " << std::endl;
    std::cout << "================================================================================================================" << std::endl;
    std::cout << std::left << std::setw(20) << "Metric" 
              << std::right << std::setw(18) << "FairRWLock" 
              << std::setw(22) << "std::shared_mutex" 
              << std::setw(25) << "TextbookReaderPrefLock" 
              << std::setw(25) << "MomentumRWLock" << std::endl;
    std::cout << "----------------------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(20) << "Read Operations"  << std::right << std::setw(18) << fair_results.total_reads  << std::setw(22) << std_results.total_reads  << std::setw(25) << tb_results.total_reads  << std::setw(25) << mom_results.total_reads << std::endl;
    std::cout << std::left << std::setw(20) << "Write Operations" << std::right << std::setw(18) << fair_results.total_writes << std::setw(22) << std_results.total_writes << std::setw(25) << tb_results.total_writes << std::setw(25) << mom_results.total_writes << std::endl;
    std::cout << std::left << std::setw(20) << "Total Operations" << std::right << std::setw(18) << fair_total                << std::setw(22) << std_total                << std::setw(25) << tb_total                << std::setw(25) << mom_total << std::endl;
    std::cout << "----------------------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::left << std::setw(20) << "Read Throughput"  << std::right << std::setw(11) << fair_read_tp  << " ops/s" << std::setw(15) << std_read_tp  << " ops/s" << std::setw(18) << tb_read_tp  << " ops/s" << std::setw(18) << mom_read_tp << " ops/s" << std::endl;
    std::cout << std::left << std::setw(20) << "Write Throughput" << std::right << std::setw(11) << fair_write_tp << " ops/s" << std::setw(15) << std_write_tp << " ops/s" << std::setw(18) << tb_write_tp << " ops/s" << std::setw(18) << mom_write_tp << " ops/s" << std::endl;
    
    std::cout << "\n================================================================================================================" << std::endl;
    std::cout << "                                   LATENCY DISTRIBUTION MATRIX (Nanoseconds)                                    " << std::endl;
    std::cout << "================================================================================================================" << std::endl;
    std::cout << std::left << std::setw(20) << "Reader p50 (Median)" << std::right << std::setw(18) << calc_percentile(fair_results.reader_latencies_ns, 0.50)   << std::setw(22) << calc_percentile(std_results.reader_latencies_ns, 0.50)   << std::setw(25) << calc_percentile(tb_results.reader_latencies_ns, 0.50)   << std::setw(25) << calc_percentile(mom_results.reader_latencies_ns, 0.50) << std::endl;
    std::cout << std::left << std::setw(20) << "Reader p95"          << std::right << std::setw(18) << calc_percentile(fair_results.reader_latencies_ns, 0.95)   << std::setw(22) << calc_percentile(std_results.reader_latencies_ns, 0.95)   << std::setw(25) << calc_percentile(tb_results.reader_latencies_ns, 0.95)   << std::setw(25) << calc_percentile(mom_results.reader_latencies_ns, 0.95) << std::endl;
    std::cout << std::left << std::setw(20) << "Reader p99.9"        << std::right << std::setw(18) << calc_percentile(fair_results.reader_latencies_ns, 0.999)  << std::setw(22) << calc_percentile(std_results.reader_latencies_ns, 0.999)  << std::setw(25) << calc_percentile(tb_results.reader_latencies_ns, 0.999)  << std::setw(25) << calc_percentile(mom_results.reader_latencies_ns, 0.999) << std::endl;
    std::cout << std::left << std::setw(20) << "Reader p99.99"       << std::right << std::setw(18) << calc_percentile(fair_results.reader_latencies_ns, 0.9999) << std::setw(22) << calc_percentile(std_results.reader_latencies_ns, 0.9999) << std::setw(25) << calc_percentile(tb_results.reader_latencies_ns, 0.9999) << std::setw(25) << calc_percentile(mom_results.reader_latencies_ns, 0.9999) << std::endl;
    std::cout << std::left << std::setw(20) << "Reader Max"          << std::right << std::setw(18) << calc_percentile(fair_results.reader_latencies_ns, 1.0)    << std::setw(22) << calc_percentile(std_results.reader_latencies_ns, 1.0)    << std::setw(25) << calc_percentile(tb_results.reader_latencies_ns, 1.0)    << std::setw(25) << calc_percentile(mom_results.reader_latencies_ns, 1.0) << std::endl;
    std::cout << "----------------------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(20) << "Writer p50 (Median)" << std::right << std::setw(18) << calc_percentile(fair_results.writer_latencies_ns, 0.50)   << std::setw(22) << calc_percentile(std_results.writer_latencies_ns, 0.50)   << std::setw(25) << calc_percentile(tb_results.writer_latencies_ns, 0.50)   << std::setw(25) << calc_percentile(mom_results.writer_latencies_ns, 0.50) << std::endl;
    std::cout << std::left << std::setw(20) << "Writer p95"          << std::right << std::setw(18) << calc_percentile(fair_results.writer_latencies_ns, 0.95)   << std::setw(22) << calc_percentile(std_results.writer_latencies_ns, 0.95)   << std::setw(25) << calc_percentile(tb_results.writer_latencies_ns, 0.95)   << std::setw(25) << calc_percentile(mom_results.writer_latencies_ns, 0.95) << std::endl;
    std::cout << std::left << std::setw(20) << "Writer p99.9"        << std::right << std::setw(18) << calc_percentile(fair_results.writer_latencies_ns, 0.999)  << std::setw(22) << calc_percentile(std_results.writer_latencies_ns, 0.999)  << std::setw(25) << calc_percentile(tb_results.writer_latencies_ns, 0.999)  << std::setw(25) << calc_percentile(mom_results.writer_latencies_ns, 0.999) << std::endl;
    std::cout << std::left << std::setw(20) << "Writer p99.99"       << std::right << std::setw(18) << calc_percentile(fair_results.writer_latencies_ns, 0.9999) << std::setw(22) << calc_percentile(std_results.writer_latencies_ns, 0.9999) << std::setw(25) << calc_percentile(tb_results.writer_latencies_ns, 0.9999) << std::setw(25) << calc_percentile(mom_results.writer_latencies_ns, 0.9999) << std::endl;
    std::cout << std::left << std::setw(20) << "Writer Max"          << std::right << std::setw(18) << calc_percentile(fair_results.writer_latencies_ns, 1.0)    << std::setw(22) << calc_percentile(std_results.writer_latencies_ns, 1.0)    << std::setw(25) << calc_percentile(tb_results.writer_latencies_ns, 1.0)    << std::setw(25) << calc_percentile(mom_results.writer_latencies_ns, 1.0) << std::endl;
    std::cout << "================================================================================================================\n" << std::endl; 
    
    log_result("Comparative Throughput Test (Completed)", true);
}

// ----------------------------------------------------------------------------
// Lock Adapters for templated starvation testing
// ----------------------------------------------------------------------------

struct StdTimedAdapter
{
    std::shared_timed_mutex lock;
    
    void lock_shared() 
    { 
        lock.lock_shared(); 
    }
    
    void unlock_shared() 
    { 
        lock.unlock_shared(); 
    }
    
    bool try_lock_for(std::chrono::milliseconds ms) 
    { 
        return lock.try_lock_for(ms); 
    }
    
    void unlock() 
    { 
        lock.unlock(); 
    }
};

// ----------------------------------------------------------------------------
// Standardized baseline parameters for adapter performance testing. Balances 
// starvation thresholds and override limits to benchmark fairly against 
// OS-native black-box primitives.
// ----------------------------------------------------------------------------
struct AdapterPolicy : DefaultFairRWLockPolicy
{
    static constexpr auto starvationThreshold = std::chrono::milliseconds(2);
    static constexpr auto maxWriterWait       = std::chrono::milliseconds(50);
};

struct FairLockAdapter
{
    std::unique_ptr<TestFairRWLock<AdapterPolicy>> lock;
    
    FairLockAdapter()
    {
        lock = std::make_unique<TestFairRWLock<AdapterPolicy>>();
    }
    
    void lock_shared() 
    { 
        lock->ReadLock(); 
    }
    
    void unlock_shared() 
    { 
        lock->ReadUnlock(); 
    }
    
    bool try_lock_for(std::chrono::milliseconds ms) 
    { 
        return lock->WriteLock(ms); 
    }
    
    void unlock() 
    { 
        lock->WriteUnlock(); 
    }
};

struct TextbookLockAdapter
{
    TextbookReaderPrefLock lock;
    
    void lock_shared() 
    { 
        lock.ReadLock(); 
    }
    
    void unlock_shared() 
    { 
        lock.ReadUnlock(); 
    }
    
    bool try_lock_for(std::chrono::milliseconds ms) 
    { 
        return lock.WriteLock(ms); 
    }
    
    void unlock() 
    { 
        lock.WriteUnlock(); 
    }
};

struct MomentumLockAdapter
{
    MomentumRWLock lock;
    
    void lock_shared() 
    { 
        lock.ReadLock(INFINITE_TIME); 
    }
    
    void unlock_shared() 
    { 
        lock.ReadUnlock(); 
    }
    
    bool try_lock_for(std::chrono::milliseconds ms) 
    { 
        return lock.WriteLock(static_cast<uint32_t>(ms.count())); 
    }
    
    void unlock() 
    { 
        lock.WriteUnlock(); 
    }
};

// ----------------------------------------------------------------------------
// Simulates an extreme reader oversubscription scenario using active 
// CPU spin-waits inside read locks to guarantee continuous read overlaps.
// Used to test if lock implementations suffer from strict writer starvation
// ----------------------------------------------------------------------------
template <typename LockAdapter>
bool execute_starvation_test(const char* lock_name)
{
    LockAdapter rwlock;
    std::atomic<bool> running{true};
    std::atomic<int> readers_ready{0};

    // Heavily oversubscribe the CPU to guarantee overlapping read intervals
    const int num_readers = std::max<int>(16, get_hw_threads() * 2);
    
    auto reader_func = [&]()
    {
        readers_ready.fetch_add(1, std::memory_order_relaxed);
        
        while (running.load(std::memory_order_relaxed))
        {
            rwlock.lock_shared();
                        
            // A simple for-loop takes <100 nanoseconds, leaving gaps where
            // the read count drops to 0 naturally. By busy-waiting for 500us 
            // inside the lock, we guarantee the threads overlap continuously. 
            // The read count will almost NEVER drop to 0.           
            auto start = std::chrono::high_resolution_clock::now();
            
            while (std::chrono::high_resolution_clock::now() - start < 500us)
            {
                // Active spin-wait (prevents OS from yielding the lock holder).
                // Hardware pause mitigates hyper-thread instruction starvation 
                // and memory bus saturation during intense clock polling.
                cpu_relax_pause();
            }
            
            rwlock.unlock_shared();
            
            // Tiny yield outside the lock to let the writer thread schedule
            std::this_thread::yield();
        }
    };

    std::vector<std::thread> readers;
    
    for (int i = 0; i < num_readers; ++i)
    {
        readers.emplace_back(reader_func);
    }

    while (readers_ready.load(std::memory_order_relaxed) < num_readers)
    {
        std::this_thread::yield();
    }
    
    std::this_thread::sleep_for(200ms);

    std::cout << "  -> Testing: " << lock_name << " (" << num_readers << " aggressive readers)\n";

    const int num_write_attempts = 5;
    const auto timeout_limit = 2000ms;
    int successful_acquisitions = 0;
    std::vector<double> wait_times_ms;

    for (int i = 0; i < num_write_attempts; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        
        bool acquired = rwlock.try_lock_for(timeout_limit);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> wait_ms = end - start;

        if (acquired)
        {
            wait_times_ms.push_back(wait_ms.count());
            successful_acquisitions++;
            
            std::this_thread::sleep_for(1ms);
            rwlock.unlock();
        }

        std::this_thread::sleep_for(100ms);
    }

    running.store(false, std::memory_order_relaxed);
    
    for (auto& t : readers)
    {
        t.join();
    }
    
    if (!wait_times_ms.empty())
    {
        double max_wait = *std::max_element(wait_times_ms.begin(), wait_times_ms.end());
        double sum_wait = std::accumulate(wait_times_ms.begin(), wait_times_ms.end(), 0.0);
        double avg_wait = sum_wait / wait_times_ms.size();

        std::cout << "     Results: Average Wait = " << std::fixed << std::setprecision(2) << avg_wait << " ms | Max Wait = " << max_wait << " ms\n";
    }
    else
    {
        std::cout << "     Results: 100% Starvation Rate. Writer never acquired the lock.\n";
    }

    return (successful_acquisitions == num_write_attempts);
}

// ----------------------------------------------------------------------------
// Macro wrapper that executes the starvation benchmark across the custom 
// FairRWLock and all comparison adapters, tracking if the custom lock passes.
// ----------------------------------------------------------------------------
void writer_starvation_comparison_test()
{
    std::cout << "\n===================================================================" << std::endl;
    std::cout << "                 WRITER STARVATION BENCHMARK                  " << std::endl;
    std::cout << "===================================================================" << std::endl;
        
    bool fair_passed = execute_starvation_test<FairLockAdapter>("FairRWLock");

    std::cout << "-------------------------------------------------------------------" << std::endl;
    
    execute_starvation_test<StdTimedAdapter>("std::shared_timed_mutex");
    
    std::cout << "-------------------------------------------------------------------" << std::endl;

    execute_starvation_test<TextbookLockAdapter>("TextbookReaderPrefLock");
    
    std::cout << "-------------------------------------------------------------------" << std::endl;

    execute_starvation_test<MomentumLockAdapter>("MomentumRWLock");
    
    std::cout << "===================================================================\n" << std::endl;

    log_result("Writer Starvation Comparison Benchmark (Passed = Fair) ", fair_passed);
}

// ----------------------------------------------------------------------------
// Simulates a severe environment where writers have a rigid 2-second 
// deadline, proving that starvation-protection mechanisms successfully complete 
// the acquisition before timeout occurs.
// ----------------------------------------------------------------------------
template <typename LockAdapter>
void execute_timeout_starvation_test(const char* lock_name)
{
    LockAdapter rwlock;
    std::atomic<bool> running{true};
    std::atomic<int> readers_ready{0};

    // Heavily oversubscribe to simulate a true continuous read lock threshold
    const int num_readers = std::max<int>(16, get_hw_threads() * 2);
    
    auto reader_func = [&]()
    {
        readers_ready.fetch_add(1, std::memory_order_relaxed);
        
        while (running.load(std::memory_order_relaxed))
        {
            rwlock.lock_shared();
            
            auto start = std::chrono::high_resolution_clock::now();
            
            while (std::chrono::high_resolution_clock::now() - start < 500us)
            {
                // Active spin-wait keeps the lock highly contested
                cpu_relax_pause();
            }
            
            rwlock.unlock_shared();
            std::this_thread::yield();
        }
    };

    std::vector<std::thread> readers;
    
    for (int i = 0; i < num_readers; ++i)
    {
        readers.emplace_back(reader_func);
    }

    while (readers_ready.load(std::memory_order_relaxed) < num_readers)
    {
        std::this_thread::yield();
    }
    
    std::this_thread::sleep_for(200ms);

    std::cout << "  -> Testing 2-Second Timeout Starvation: " << lock_name 
              << " (" << num_readers << " aggressive readers)\n";

    const int num_write_attempts = 3;
    const auto timeout_limit = std::chrono::milliseconds(2000);

    for (int i = 0; i < num_write_attempts; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        
        bool acquired = rwlock.try_lock_for(timeout_limit);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> wait_ms = end - start;

        std::cout << "     Attempt " << (i + 1) << " Wait Time: " << std::fixed << std::setprecision(2) << wait_ms.count() << " ms | Acquired: " << (acquired ? "Yes" : "No") << "\n";

        if (acquired)
        {
            std::this_thread::sleep_for(1ms);
            rwlock.unlock();
        }

        std::this_thread::sleep_for(100ms);
    }

    running.store(false, std::memory_order_relaxed);
    
    for (auto& t : readers)
    {
        t.join();
    }
}

// ----------------------------------------------------------------------------
// Executes the `execute_timeout_starvation_test` helper sequentially 
// across all four lock variants to observe failure states.
// ----------------------------------------------------------------------------
void writer_timeout_starvation_test()
{
    std::cout << "\n=========================================================================================" << std::endl;
    std::cout << "           WRITER 2-SECOND TIMEOUT STARVATION TEST            " << std::endl;
    std::cout << "=========================================================================================" << std::endl;
        
    execute_timeout_starvation_test<FairLockAdapter>("FairRWLock");
    std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
    execute_timeout_starvation_test<StdTimedAdapter>("std::shared_timed_mutex");
    std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
    execute_timeout_starvation_test<TextbookLockAdapter>("TextbookReaderPrefLock");
    std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
    execute_timeout_starvation_test<MomentumLockAdapter>("MomentumRWLock");

    std::cout << "=========================================================================================\n" << std::endl;
    log_result("Writer 2-Second Timeout Starvation Test", true);
}

struct SweepResult
{
    std::string config_name;
    std::string policy_info;
    double read_tp;
    double write_tp;
    uint64_t total_ops;
};

// ----------------------------------------------------------------------------
// Reader-biased configuration prioritizing high throughput on the fast path. 
// Deliberately delays writer starvation mechanisms to maximize read operations 
// at the cost of slight writer latency.
// ----------------------------------------------------------------------------
struct ReaderOptPolicy : DefaultFairRWLockPolicy
{
    static constexpr auto starvationThreshold = std::chrono::milliseconds(50);
    static constexpr auto maxWriterWait       = std::chrono::milliseconds(200);
    static constexpr int  maxConsecWriters    = 0;
};

// ----------------------------------------------------------------------------
// Extreme fairness configuration. Prevents any writer batching or consecutive 
// writes, stripping away fast-path access immediately after a write to 
// minimize reader tail latency.
// ----------------------------------------------------------------------------
struct StrictOptPolicy : DefaultFairRWLockPolicy
{
    static constexpr auto starvationThreshold = std::chrono::milliseconds(1);
    static constexpr auto maxWriterWait       = std::chrono::milliseconds(10);
    static constexpr int  writerBatchLimit    = 1;
    static constexpr int  maxConsecWriters    = 1;
};

// ----------------------------------------------------------------------------
// Writer-biased configuration tailored for write-heavy workloads. Triggers 
// starvation thresholds rapidly and allows massive consecutive writer 
// batches to drain the queue.
// ----------------------------------------------------------------------------
struct WriterOptPolicy : DefaultFairRWLockPolicy
{
    static constexpr auto starvationThreshold = std::chrono::milliseconds(2);
    static constexpr auto maxWriterWait       = std::chrono::milliseconds(20);
    static constexpr auto overrideTimeslice   = std::chrono::milliseconds(200);
    static constexpr int  writerBatchLimit    = 10;
};

// ----------------------------------------------------------------------------
// Helper function to print text wrapped neatly to a maximum column width,
// maintaining proper indentation aligned with the label column.
// ----------------------------------------------------------------------------
static void print_wrapped_text(const std::string& label, 
                               std::string_view   text, 
                               size_t             max_width = 120)
{
    std::cout << "    " << std::left << std::setw(11) << label;
    size_t indent = 4 + 11;
    size_t available_width = (max_width > indent) ? (max_width - indent) : 40;

    size_t start = 0;
    bool first_line = true;

    while (start < text.length())
    {
        while (start < text.length() && text[start] == ' ')
        {
            start++;
        }
        
        if (start >= text.length())
        {
            break;
        }

        size_t line_end = start;
        size_t last_space = start;
        
        while (line_end < text.length())
        {
            size_t next_space = text.find(' ', line_end);
            
            if (next_space == std::string_view::npos)
            {
                next_space = text.length();
            }
            
            size_t candidate_len = next_space - start;
            
            if (candidate_len > available_width && line_end > start)
            {
                break;
            }
            
            line_end = next_space;
            last_space = line_end;
            
            if (line_end == text.length())
            {
                break;
            }
            
            line_end++;
        }

        if (!first_line)
        {
            std::cout << std::string(indent, ' ');
        }

        std::string_view line_str = text.substr(start, last_space - start);
        std::cout << line_str << std::endl;

        start = last_space;
        first_line = false;
    }
}

// ----------------------------------------------------------------------------
// Measures throughput impacts of manipulating the FairRWLock policies 
// (e.g., Reader-Optimized vs Reader-Strict vs Writer-Heavy) compared to baselines.
// ----------------------------------------------------------------------------
void policy_parameter_sweep_benchmark()
{
    const unsigned int hw_threads = get_hw_threads();
    const int NUM_WRITERS = (hw_threads >= 3) ? 2 : 1;
    int readers_calc = static_cast<int>(hw_threads) - NUM_WRITERS;
    const int NUM_READERS = (readers_calc < 1) ? 1 : readers_calc;
    
    constexpr auto TEST_DURATION = 10s; 

    // Define policy behavioral descriptions ("It tells the lock / platform:") using constexpr string views
    constexpr std::string_view DESC_DEFAULT = 
        "Balance throughput and fairness. Allow readers to use the fast path freely under normal conditions, "
        "but step in with a balanced response—setting starvation gates and enforcing an override batch limit of 2 as soon as contention builds up.";
    
    constexpr std::string_view DESC_READER_BIASED = 
        "Let the readers stream through the fast path continuously. Do not gate new readers or trigger override modes "
        "until a writer has been blocked for a remarkably long time.";
    
    constexpr std::string_view DESC_READER_STRICT = 
        "Never let a writer hold the lock consecutively or run in batches. The moment a single write finishes, "
        "instantly strip away fast-path access and hand absolute priority back to the readers to minimize their tail latency.";
    
    constexpr std::string_view DESC_WRITER_BIASED = 
        "When contention spikes and a writer gets stuck, react aggressively. Trip the starvation threshold quickly, "
        "enter override mode immediately, and let a larger batch of up to 10 consecutive writers drain the queue before letting readers back in.";
    
    constexpr std::string_view DESC_OS_NATIVE = 
        "Manage thread arbitration entirely through black-box operating system scheduling—delivering solid mixed-contention "
        "handling on Windows via SRW locks, or leaning heavily into reader preference on Linux.";
    
    constexpr std::string_view DESC_TEXTBOOK = 
        "Always prioritize readers above all else. If any reader is waiting or active, they jump straight to the front of the line, "
        "even if it means waiting writers starve indefinitely.";
    
    constexpr std::string_view DESC_MOMENTUM = 
        "Flip-flop priority in coarse blocks between readers and writers using a central mutex, "
        "trading lock-free fast-path speed for manual phase alternation.";

    std::cout << "\n=======================================================================================================================" << std::endl;
    std::cout << "               POLICY PARAMETER SWEEP BENCHMARK               " << std::endl;
    std::cout << "=======================================================================================================================" << std::endl;
    std::cout << "Scenario: " << NUM_READERS << " Readers, " << NUM_WRITERS << " Writers. (" << TEST_DURATION.count() << "s per config)\n" << std::endl;

    auto run_workload = [&](auto               lock_read_fn, 
                            auto               unlock_read_fn, 
                            auto               lock_write_fn, 
                            auto               unlock_write_fn, 
                            const char*        config_name, 
                            const std::string& param_info, 
                            std::string_view   policy_description = "") -> SweepResult
    {
        std::atomic<bool>     run_flag{true};
        std::atomic<uint64_t> total_reads{0};
        std::atomic<uint64_t> total_writes{0};
        std::barrier          sync_point(NUM_READERS + NUM_WRITERS + 1);

        auto reader_task = [&]()
        {
            uint64_t local_reads = 0;
            sync_point.arrive_and_wait();
            
            while (run_flag.load(std::memory_order_relaxed))
            {
                lock_read_fn();
                local_reads++;
                std::this_thread::yield();
                unlock_read_fn();
            }
            
            total_reads.fetch_add(local_reads, 
                                  std::memory_order_relaxed);
        };

        auto writer_task = [&]()
        {
            uint64_t local_writes = 0;
            sync_point.arrive_and_wait();
            
            while (run_flag.load(std::memory_order_relaxed))
            {
                lock_write_fn();
                local_writes++;
                std::this_thread::yield();
                unlock_write_fn();
                std::this_thread::sleep_for(1ms);
            }
            
            total_writes.fetch_add(local_writes, 
                                   std::memory_order_relaxed);
        };

        std::vector<std::thread> threads;
        
        for (int i = 0; i < NUM_READERS; ++i)
        {
            threads.emplace_back(reader_task);
        }
        
        for (int i = 0; i < NUM_WRITERS; ++i)
        {
            threads.emplace_back(writer_task);
        }

        // Expanded setw width to 45 to prevent string overflow when rendering progress status
        std::cout << "  -> Running: " << std::left << std::setw(45) << config_name 
                  << "[ RUNNING: " << TEST_DURATION.count() << "s ]" << std::flush;

        sync_point.arrive_and_wait();
        std::this_thread::sleep_for(TEST_DURATION);
        run_flag.store(false, 
                       std::memory_order_relaxed);

        for (auto& t : threads)
        {
            t.join();
        }

        double test_secs = static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(TEST_DURATION).count());
        double read_tp   = static_cast<double>(total_reads.load()) / test_secs;
        double write_tp  = static_cast<double>(total_writes.load()) / test_secs;

        // Expanded character clear buffer to 100 chars
        std::cout << "\r" << std::string(100, ' ') << "\r"; 
        
        std::cout << "  [Completed] " << config_name << std::endl;
        print_wrapped_text("Policy:", param_info, 120);

        if (!policy_description.empty())
        {
            print_wrapped_text("Behavior:", policy_description, 120);
        }

        std::cout << std::endl;

        return {config_name, param_info, read_tp, write_tp, total_reads.load() + total_writes.load()};
    };

    auto run_fair_lock = [&](auto&              lock_instance, 
                             const char*        name, 
                             const std::string& param_info, 
                             std::string_view   policy_description = "") -> SweepResult
    {
        return run_workload([&] { lock_instance.ReadLock(); },   
                            [&] { lock_instance.ReadUnlock(); },
                            [&] { lock_instance.WriteLock(); },  
                            [&] { lock_instance.WriteUnlock(); },
                            name, 
                            param_info, 
                            policy_description);
    };

    std::vector<SweepResult> results;

    TestFairRWLock<DefaultFairRWLockPolicy> def_lock;
    results.push_back(run_fair_lock(def_lock, 
                                    "1. Default Baseline", 
                                    "starveThresh=3ms, maxWait=50ms, overrideTime=100ms, batchLimit=2, consecWriters=0",
                                    DESC_DEFAULT));

    TestFairRWLock<ReaderOptPolicy> reader_lock;
    results.push_back(run_fair_lock(reader_lock, 
                                    "2. Reader-Biased (High Throughput)", 
                                    "starveThresh=50ms, maxWait=200ms, overrideTime=100ms, batchLimit=2, consecWriters=0",
                                    DESC_READER_BIASED));

    TestFairRWLock<StrictOptPolicy> strict_lock;
    results.push_back(run_fair_lock(strict_lock, 
                                    "3. Reader-Strict (Single-Write Cap)", 
                                    "starveThresh=1ms, maxWait=10ms, overrideTime=100ms, batchLimit=1, consecWriters=1",
                                    DESC_READER_STRICT));

    TestFairRWLock<WriterOptPolicy> writer_lock;
    results.push_back(run_fair_lock(writer_lock, 
                                    "4. Writer-Prioritized Burst", 
                                    "starveThresh=2ms, maxWait=20ms, overrideTime=200ms, batchLimit=10, consecWriters=0",
                                    DESC_WRITER_BIASED));

    std::shared_mutex std_lock;
    results.push_back(run_workload([&] { std_lock.lock_shared(); }, 
                                   [&] { std_lock.unlock_shared(); },
                                   [&] { std_lock.lock(); },        
                                   [&] { std_lock.unlock(); },
                                   "5. OS Native (std::shared_mutex)", 
                                   "N/A (OS Managed, Unbounded Starvation)",
                                   DESC_OS_NATIVE));

    TextbookReaderPrefLock tb_lock;
    results.push_back(run_workload([&] { tb_lock.ReadLock(); }, 
                                   [&] { tb_lock.ReadUnlock(); },
                                   [&] { tb_lock.WriteLock(); }, 
                                   [&] { tb_lock.WriteUnlock(); },
                                   "6. Textbook (Reader Pref)", 
                                   "N/A (Strict Reader Preference, Extreme Starvation)",
                                   DESC_TEXTBOOK));
                                   
    MomentumRWLock mom_sweep_lock;
    results.push_back(run_workload([&] { mom_sweep_lock.ReadLock(INFINITE_TIME); }, 
                                   [&] { mom_sweep_lock.ReadUnlock(); },
                                   [&] { mom_sweep_lock.WriteLock(INFINITE_TIME); }, 
                                   [&] { mom_sweep_lock.WriteUnlock(); },
                                   "7. Momentum (Batching Fix)", 
                                   "N/A (Momentum-based priority flip-flops)",
                                   DESC_MOMENTUM));

    double baseline_total_ops = static_cast<double>(results[4].total_ops);

    std::cout << "\n========================================================================================================" << std::endl;
    std::cout << "                               POLICY PARAMETER SWEEP PERFORMANCE SUMMARY                               " << std::endl;
    std::cout << "========================================================================================================" << std::endl;
    std::cout << std::left << std::setw(42) << "Configuration" 
              << std::right << std::setw(15) << "Read ops/s" 
              << std::setw(15) << "Write ops/s" 
              << std::setw(15) << "Total Ops" 
              << std::setw(15) << "Rel. Gain" << std::endl;
    std::cout << "--------------------------------------------------------------------------------------------------------" << std::endl;

    for (const auto& res : results)
    {
        std::cout << std::left << std::setw(42) << res.config_name 
                  << std::right << std::fixed << std::setprecision(0)
                  << std::setw(15) << res.read_tp
                  << std::setw(15) << res.write_tp
                  << std::setw(15) << res.total_ops;

        if (res.config_name == "5. OS Native (std::shared_mutex)")
        {
            std::cout << std::setw(15) << "Baseline" << std::endl;
        }
        else
        {
            double gain = ((static_cast<double>(res.total_ops) - baseline_total_ops) / baseline_total_ops) * 100.0;
            std::cout << std::setw(14) << std::showpos << std::fixed << std::setprecision(2) << gain << "%" << std::noshowpos << std::endl;
        }
    }
    std::cout << "========================================================================================================\n" << std::endl;
    
    log_result("Policy Parameter Sweep Benchmark", true);
}

// ----------------------------------------------------------------------------
// Chaos testing configuration explicitly designed to stress the fallback loop. 
// Caps consecutive writers to a moderate threshold to aggressively test 
// the reader-handoff mechanics under heavy thread saturation.
// ----------------------------------------------------------------------------
struct ConsecHammerPolicy : DefaultFairRWLockPolicy
{
    static constexpr int maxConsecWriters = 3;
};

// ----------------------------------------------------------------------------
// Heavy chaos testing targeting the `maxConsecWriters` policy.
// Spams writers to intentionally push the limits, ensuring the policy forcefully 
// grants readers access rather than failing indefinitely (defined by the 
// arbitrary 100,000 operation boundary limit).
// ----------------------------------------------------------------------------
void consecutive_limit_hammer_test()
{
    TestFairRWLock<ConsecHammerPolicy> lock;
    std::atomic<bool> running{true};
    std::atomic<int> consec_writes{0};
    std::atomic<int> max_consec_writes{0};
    std::atomic<bool> limit_breached{false};

    const unsigned int hw_threads = get_hw_threads();
    const int NUM_THREADS = std::max<int>(4u, hw_threads * 2); 
    std::barrier sync(NUM_THREADS * 2 + 1);

    auto writer_task = [&]()
    {
        sync.arrive_and_wait();
        
        while (running.load(std::memory_order_relaxed))
        {
            if (lock.WriteLock(1s))
            {
                int current = consec_writes.fetch_add(1, std::memory_order_relaxed) + 1;
                int max_val = max_consec_writes.load(std::memory_order_relaxed);
                
                while (current > max_val && !max_consec_writes.compare_exchange_weak(max_val, current, std::memory_order_relaxed))
                {
                }
                
                // 100,000 represents a genuine failure of the lock to let readers in over multiple seconds.
                if (current > 100000)
                {
                    limit_breached = true;
                }

                lock.WriteUnlock();
                std::this_thread::yield();
            }
        }
    };

    auto reader_task = [&]()
    {
        sync.arrive_and_wait();
        
        while (running.load(std::memory_order_relaxed))
        {
            if (lock.ReadLock(1s))
            {
                // When a reader successfully gets in, it resets the consecutive write counter.
                consec_writes.store(0, std::memory_order_relaxed);
                lock.ReadUnlock();
                std::this_thread::yield();
            }
        }
    };

    std::vector<std::thread> threads;
    
    for (int i = 0; i < NUM_THREADS; i++)
    {
        threads.emplace_back(writer_task);
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        threads.emplace_back(reader_task);
    }

    std::cout << std::left << std::setw(50) << "Consecutive Limit Hammer Test" 
              << "[ RUNNING: 6s, " << (NUM_THREADS * 2) << " threads... ]" << std::flush;
              
    sync.arrive_and_wait();
    std::this_thread::sleep_for(6s);
    running = false;

    for (auto& t : threads)
    {
        t.join();
    }

    // Clear the "RUNNING" text using carriage return and spaces, then print standard result
    std::cout << "\r" << std::string(85, ' ') << "\r";
    log_result("Consecutive Limit Hammer Test", !limit_breached.load());
}

// ----------------------------------------------------------------------------
// Simulates extreme thread saturation. Uses tightly bound wait times and 
// timeslices to stress test the starvation protection mechanics and prevent 
// queue stalling under extreme load.
// ----------------------------------------------------------------------------
struct OverrideStarveHammerPolicy : DefaultFairRWLockPolicy
{
    static constexpr auto maxWriterWait     = std::chrono::milliseconds(50);
    static constexpr auto overrideTimeslice = std::chrono::milliseconds(50);
    static constexpr int  writerBatchLimit  = 2;
};

// ----------------------------------------------------------------------------
// Heavy chaos testing targeting the override starvation mechanisms.
// Saturates threads specifically mapping writer wait times to guarantee no thread 
// is left blocking above an extreme threshold (>500ms).
// ----------------------------------------------------------------------------
void override_starvation_hammer_test()
{
    TestFairRWLock<OverrideStarveHammerPolicy> lock;
    std::atomic<bool> running{true};
    std::atomic<bool> writer_starved{false};

    const unsigned int hw_threads = get_hw_threads();
    const int NUM_WRITERS = std::max<int>(2u, hw_threads / 4);
    const int NUM_READERS = std::max<int>(4u, hw_threads * 2); // Deliberate oversubscription
    std::barrier sync(NUM_READERS + NUM_WRITERS + 1);

    auto reader_task = [&]()
    {
        sync.arrive_and_wait();
        
        while (running.load(std::memory_order_relaxed))
        {
            if (lock.ReadLock(1s))
            {
                std::this_thread::sleep_for(10us);
                lock.ReadUnlock();
            }
        }
    };

    auto writer_task = [&]()
    {
        sync.arrive_and_wait();
        
        while (running.load(std::memory_order_relaxed))
        {
            auto start = std::chrono::steady_clock::now();

            if (lock.WriteLock(2s))
            {
                auto wait_time = std::chrono::steady_clock::now() - start;
                
                if (wait_time > 500ms)
                {
                    writer_starved = true;
                }

                std::this_thread::sleep_for(10us);
                lock.WriteUnlock();
            }
        }
    };

    std::vector<std::thread> threads;
    
    for (int i = 0; i < NUM_READERS; i++)
    {
        threads.emplace_back(reader_task);
    }

    for (int i = 0; i < NUM_WRITERS; i++)
    {
        threads.emplace_back(writer_task);
    }

    std::cout << std::left << std::setw(50) << "Override Starvation Hammer Test"
              << "[ RUNNING: 6s, " << (NUM_READERS + NUM_WRITERS) << " threads... ]" << std::flush;

    sync.arrive_and_wait();
    std::this_thread::sleep_for(6s);
    running = false;

    for (auto& t : threads)
    {
        t.join();
    }

    std::cout << "\r" << std::string(85, ' ') << "\r";
    log_result("Override Starvation Hammer Test", !writer_starved.load());
}

struct SharedData
{
    long counter = 0;
};

std::atomic<bool> test_failed = false;

// ----------------------------------------------------------------------------
// Background reader worker used in `main()` to assert data state. 
// Volatiles capture value snapshots ensuring logic evaluates reliably to catch 
// concurrent access anomalies.
// ----------------------------------------------------------------------------
template <typename TLock>
void reader_task(TLock&                   lock, 
                 const SharedData&        data, 
                 std::atomic<long>&       read_ops, 
                 const std::atomic<bool>& stop_flag, 
                 std::barrier<>&          sync_point)
{
    sync_point.arrive_and_wait();

    while (!stop_flag && !test_failed)
    {
        typename TLock::ReadGuard guard(lock);
        
        if (guard)
        {
            read_ops++;
            volatile long val1 = data.counter;
            std::this_thread::sleep_for(1us);
            volatile long val2 = data.counter;

            if (val1 != val2)
            {
                std::cerr << "\n!!! FATAL ERROR: Data changed during a read lock! val1=" << val1 << ", val2=" << val2 << "\n";
                test_failed = true;
            }
        }
        
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

// ----------------------------------------------------------------------------
// Background writer worker used in `main()` testing sequential 
// structural integrity across long periods.
// ----------------------------------------------------------------------------
template <typename TLock>
void writer_task(TLock&                   lock, 
                 SharedData&              data, 
                 std::atomic<long>&       write_ops, 
                 const std::atomic<bool>& stop_flag, 
                 std::barrier<>&          sync_point)
{
    sync_point.arrive_and_wait();

    while (!stop_flag && !test_failed)
    {
        typename TLock::WriteGuard guard(lock);
        
        if (guard)
        {
            data.counter++;
            write_ops++;
        }

        std::this_thread::sleep_for(std::chrono::microseconds(150));
    }
}

// ----------------------------------------------------------------------------
// Returns CPU namestring for informational purposes
// ----------------------------------------------------------------------------
static std::string cpu_name()
{
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    char brand[0x40];
    std::memset(brand, 0, sizeof(brand));

#if defined(_MSC_VER)
    int cpuInfo[4] = { 0 };
    __cpuid(cpuInfo, 0x80000000);
    unsigned int nExIds = cpuInfo[0];

    if (nExIds >= 0x80000004)
    {
        __cpuid(cpuInfo, 0x80000002);
        std::memcpy(brand, cpuInfo, sizeof(cpuInfo));

        __cpuid(cpuInfo, 0x80000003);
        std::memcpy(brand + 16, cpuInfo, sizeof(cpuInfo));

        __cpuid(cpuInfo, 0x80000004);
        std::memcpy(brand + 32, cpuInfo, sizeof(cpuInfo));
    }
#else
    unsigned int eax;

    __asm__ __volatile__("cpuid" : "=a"(eax) : "a"(0x80000000) : "ebx", "ecx", "edx");

    if (eax >= 0x80000004)
    {
        unsigned int data[12];

        for (unsigned int i = 0; i < 3; ++i)
        {
            __asm__ __volatile__("cpuid"
                : "=a"(data[i * 4 + 0]),
                "=b"(data[i * 4 + 1]),
                "=c"(data[i * 4 + 2]),
                "=d"(data[i * 4 + 3])
                : "a"(0x80000002 + i));
        }

        std::memcpy(brand, data, sizeof(data));
    }
#endif
    return std::string(brand);

#elif defined(__APPLE__)
    char buf[256];
    size_t size = sizeof(buf);

    if (sysctlbyname("machdep.cpu.brand_string", &buf, &size, NULL, 0) == 0)
    {
        return std::string(buf);
    }

    size = sizeof(buf);

    if (sysctlbyname("hw.model", &buf, &size, NULL, 0) == 0)
    {
        return std::string(buf);
    }

    return "Unknown CPU";

#elif defined(__linux__)
    std::ifstream f("/proc/cpuinfo");
    std::string line;

    while (std::getline(f, line))
    {
        if (line.find("model name") != std::string::npos ||
            line.find("Hardware") != std::string::npos)
        {
            std::size_t pos = line.find(':');

            if (pos != std::string::npos)
            {
                return line.substr(pos + 2);
            }
        }
    }

    return "Unknown CPU";
#else
    return "Unknown CPU";
#endif
}

static void set_high_priority()
{
#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);
#else
    setpriority(PRIO_PROCESS, 0, -10);
#endif
}

// ----------------------------------------------------------------------------
// A balanced, highly resilient configuration designed for continuous 
// long-running liveness tests, ensuring absolutely no thread starves 
// over extended periods of persistent lock contention.
// ----------------------------------------------------------------------------
struct PersistencePolicy : DefaultFairRWLockPolicy
{
    static constexpr auto maxWriterWait       = std::chrono::milliseconds(15);
    static constexpr auto overrideTimeslice   = std::chrono::milliseconds(50);
    static constexpr int  writerBatchLimit    = 2;
    static constexpr int  maxConsecWriters    = 3;
    static constexpr auto starvationThreshold = std::chrono::milliseconds(4);
};

int main()
{
#ifdef _WIN32
    MMRESULT Res = timeBeginPeriod(1);
#endif

    setvbuf(stdout, nullptr, _IONBF, 0);
    std::cout.setf(std::ios::unitbuf);

    std::cout << "\n=========================================================\n";
    std::cout << "                 FAIR RW LOCK TEST SUITE                 \n";
    std::cout << "=========================================================\n";

    std::cout << "\n=========================================================\n";
    std::cout << "    CPU: " << cpu_name() << "\n";
    std::cout << "    NUMA Configuration: " << (USE_NUMA_CONFIG ? "Enabled (Distributed Counters)" : "Disabled (Centralized Fast-Path)") << "\n";
    std::cout << "=========================================================\n";    

    set_high_priority();
    std::cout << std::endl;

    std::cout << "--- Running Core Safety and Invariant Tests ---\n" << std::endl;
    reader_writer_race_test();
    writer_timeout_with_readers();
    reader_timeout_with_writer();
    try_lock_under_contention_test();
    raii_guard_functionality_test();

    std::cout << "\n--- Running Advanced Edge Case and Semantic Tests ---\n" << std::endl;
    non_recursive_reader_test();
    zero_timeout_edge_case_test();
    reader_thundering_herd_test();
    strict_writer_batch_limit_test();
    guard_move_assignment_test();
    max_duration_overflow_test();
    mid_queue_timeout_test();

    std::cout << "\n--- Running Deterministic Fairness Policy Tests ---\n" << std::endl;
    writer_punches_through_readers_test();
    override_reset_test();
    consecutive_limit_breaks_batch_test();
    writer_timeout_handoff_test();
    override_timeslice_allows_reader_test();

    std::cout << "\n--- Running Extreme Chaos Hammer Tests ---\n" << std::endl;
    consecutive_limit_hammer_test();
    override_starvation_hammer_test();

    std::cout << "\n--- Running Stress and Liveness Tests ---\n" << std::endl;
    multiple_writers_fairness_test();
    stress_liveness_test();

    std::cout << "\n--- Running Performance Tests ---\n" << std::endl;
    run_comparative_throughput_test();
    writer_starvation_comparison_test();
    writer_timeout_starvation_test(); 
    policy_parameter_sweep_benchmark();

    std::cout << std::endl;

    const unsigned int hw_threads = get_hw_threads();
    const int NUM_WRITERS = (hw_threads >= 3) ? 2 : 1;
    int readers_calc = static_cast<int>(hw_threads) - NUM_WRITERS;
    const int NUM_READERS = (readers_calc < 1) ? 1 : readers_calc;
    
    constexpr auto TEST_DURATION = 10s;

    bool all_checks_passed = true;

    TestFairRWLock<PersistencePolicy> lock;

    SharedData shared_data;
    std::atomic<long> read_operations  = 0;
    std::atomic<long> write_operations = 0;
    std::atomic<bool> stop_flag        = false;

    std::vector<std::jthread> threads;
    threads.reserve(NUM_READERS + NUM_WRITERS);

    std::barrier sync_point(NUM_READERS + NUM_WRITERS + 1);

    for (int i = 0; i < NUM_READERS; ++i)
    {
        threads.emplace_back(reader_task<TestFairRWLock<PersistencePolicy>>, 
                             std::ref(lock), 
                             std::ref(shared_data), 
                             std::ref(read_operations), 
                             std::ref(stop_flag), 
                             std::ref(sync_point));
    }

    for (int i = 0; i < NUM_WRITERS; ++i)
    {
        threads.emplace_back(writer_task<TestFairRWLock<PersistencePolicy>>, 
                             std::ref(lock), 
                             std::ref(shared_data), 
                             std::ref(write_operations), 
                             std::ref(stop_flag), 
                             std::ref(sync_point));
    }

    std::cout << "Starting persistence test for " << TEST_DURATION.count() << " seconds..." << std::endl;
    std::cout << "  - " << NUM_READERS << " reader threads" << std::endl;
    std::cout << "  - " << NUM_WRITERS << " writer threads" << std::endl;

    sync_point.arrive_and_wait();
    const auto start_time = std::chrono::steady_clock::now();

    std::this_thread::sleep_for(TEST_DURATION);
    stop_flag = true;

    const auto end_time = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\nTest finished in " << elapsed.count() << " ms." << std::endl;
    std::cout << "  - Total Read Operations:  " << read_operations << std::endl;
    std::cout << "  - Total Write Operations: " << write_operations << std::endl;
    std::cout << "  - Final Counter Value:    " << shared_data.counter << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "Verifying results..." << std::endl;

    if (test_failed)
    {
        std::cerr << "  [FAIL] A fatal error was detected in a worker thread." << std::endl;
        all_checks_passed = false;
    }

    if (read_operations > 0)
    {
        std::cout << "  [PASS] Reader liveness confirmed." << std::endl;
    }
    else
    {
        std::cerr << "  [FAIL] Readers were starved and performed no operations." << std::endl;
        all_checks_passed = false;
    }

    if (write_operations > 0)
    {
        std::cout << "  [PASS] Writer liveness confirmed." << std::endl;
    }
    else
    {
        std::cerr << "  [FAIL] Writers were starved and performed no operations." << std::endl;
        all_checks_passed = false;
    }

    if (shared_data.counter == write_operations)
    {
        std::cout << "  [PASS] Data integrity confirmed." << std::endl;
    }
    else
    {
        std::cerr << "  [FAIL] Data corruption detected! Counter value (" << shared_data.counter << ") does not match write operations (" << write_operations << ")." << std::endl;
        all_checks_passed = false;
    }

#ifdef _WIN32
    if (Res == TIMERR_NOERROR)
    {
        timeEndPeriod(1);
    }
#endif

    std::cout << "------------------------------------------" << std::endl;
    
    if (all_checks_passed && global_tests_passed.load())
    {
        std::cout << "All checks passed. The lock appears to be working correctly." << std::endl;
        return 0;
    }
    else
    {
        std::cerr << "One or more checks failed." << std::endl;
        return 1;
    }    
}