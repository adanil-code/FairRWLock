/*
* Fair Reader-Writer lock
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

#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <semaphore>
#include <string_view>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <sched.h>
// ----------------------------------------------------------------------------
// Linux NUMA Forward Declarations
// If EnableNUMA = false, the compiler discards the if constexpr branches 
// invoking these, allowing compilation without installing libnuma-dev or 
// linking -lnuma.
// ----------------------------------------------------------------------------
struct bitmask;
extern "C" 
{
    int numa_available(void);
    int numa_max_node(void);
    int numa_bitmask_isbitset(const struct bitmask*, int);
    int numa_node_of_cpu(int);
    extern struct bitmask* numa_all_nodes_ptr;
}
#endif

#ifndef NDEBUG
#include <sstream>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

// ----------------------------------------------------------------------------
// CPU Relax / Pause Hint
// Emits a hardware-level pause instruction to reduce memory bus contention 
// during brief spin-wait cycles.
// ----------------------------------------------------------------------------
inline void cpu_relax_pause() noexcept
{
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
    __yield();
#elif defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ volatile("yield" ::: "memory");
#else
    // Fallback compiler barrier for unsupported architectures
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

#ifndef NDEBUG
#define LOG_DEBUG(lock_ptr, msg) (lock_ptr)->debug_log(msg)
#else
#define LOG_DEBUG(lock_ptr, msg) ((void)0)
#endif

using namespace std::chrono_literals;

// ----------------------------------------------------------------------------
// Cache Line Constants
//
// We explicitly avoid std::hardware_destructive_interference_size due to ABI 
// instability warnings ([-Winterference-size]) across different compiler flags.
// 
// Apply 128-byte alignment strictly for Apple Silicon to prevent false sharing,
// while preserving optimal 64-byte density for x86_64 and standard Linux ARM64.
// ----------------------------------------------------------------------------
#if defined(__APPLE__) && defined(__aarch64__)
constexpr size_t CACHE_LINE_SIZE = 128;
#else        
constexpr size_t CACHE_LINE_SIZE = 64;
#endif

// -------------------------------------------------------------------------
// NUMA Topology Discovery & Caching
// Encapsulated within a strict templated struct. Ensures topology evaluation 
// metadata resides in global read-only memory, guaranteeing zero false-sharing
// with individual lock instances.
// -------------------------------------------------------------------------
template <bool EnableNUMA>
struct NumaTopology
{
    static const std::vector<uint32_t>& GetValidNodes() noexcept
    {
        static const std::vector<uint32_t> validNodes = []()
        {
            if constexpr (EnableNUMA)
            {
                try
                {
                    std::vector<uint32_t> nodes;

#if defined(_WIN32)
                    ULONG highestNode = 0;

                    if (GetNumaHighestNodeNumber(&highestNode)) [[likely]]
                    {
                        const USHORT maxNode = static_cast<USHORT>(highestNode);

                        for (USHORT i = 0; i <= maxNode; ++i)
                        {
                            ULONGLONG availableMemory = 0;
                            
                            if (GetNumaAvailableMemoryNodeEx(i, &availableMemory))
                            {
                                nodes.push_back(i);
                            }
                        }
                    }
#elif defined(__linux__)
                    if (numa_available() >= 0) [[likely]]
                    {
                        int highestNode = numa_max_node();

                        for (int i = 0; i <= highestNode; ++i)
                        {
                            if (numa_bitmask_isbitset(numa_all_nodes_ptr, i))
                            {
                                nodes.push_back(i);
                            }
                        }
                    }
#endif
                    if (nodes.empty()) [[unlikely]]
                    {
                        nodes.push_back(0);
                    }

                    nodes.shrink_to_fit();
                    
                    return nodes;
                }
                catch (...)
                {
                    // Returns a static fallback to strictly prevent dangling reference UB
                    static const std::vector<uint32_t> fallback{ 0 };                    
                    return fallback;
                }
            }
            else
            {
                static const std::vector<uint32_t> fallback{ 0 };
                return fallback;
            }
        }();

        return validNodes;
    }

    static inline const bool   is_multi_node = EnableNUMA ? (GetValidNodes().size() > 1) : false;
    // Guaranteed to be a power of 2 for zero-cost bitwise modulo operations
    static inline const size_t num_stripes   = EnableNUMA ? std::bit_ceil(GetValidNodes().size()) : 1;
};

// -------------------------------------------------------------------------
// Conditionally Compiled NUMA Storage
// Over-aligned stripes to eliminate reader fast-path cache line bouncing.
// -------------------------------------------------------------------------
template <bool Enable>
struct NumaStorage;

// Zero-overhead state when EnableNUMA = false (0 bytes)
template <>
struct NumaStorage<false> 
{
    inline void init(bool, size_t) noexcept 
    {
    }
};

// Distributed Node Arrays when EnableNUMA = true
template <>
struct NumaStorage<true>
{
    struct alignas(CACHE_LINE_SIZE) ReaderStripe 
    {
        std::atomic<uint32_t> count{ 0 };
        
        ReaderStripe() noexcept : count(0) 
        {
        }
    };
    
    std::unique_ptr<ReaderStripe[]> stripes;

    // Defers allocation dynamically. If only 1 node is detected at startup, 
    // it skips array allocation entirely to conserve L1/L2 cache capacity.
    inline void init(bool is_multi_node, size_t num_stripes) 
    {
        if (is_multi_node) 
        {
            stripes = std::make_unique<ReaderStripe[]>(num_stripes);
        }
    }
    
    NumaStorage() = default;
    NumaStorage(const NumaStorage&) = delete;
    NumaStorage& operator=(const NumaStorage&) = delete;
    
    NumaStorage(NumaStorage&&) noexcept = default;
    NumaStorage& operator=(NumaStorage&&) noexcept = default;
};

// -------------------------------------------------------------------------
// Default Fairness Policy Traits
// -------------------------------------------------------------------------
struct DefaultFairRWLockPolicy
{
    using clock    = std::chrono::steady_clock;
    using duration = clock::duration;

    // The maximum time a writer will wait in the queue before declaring an "override" emergency.
    // If the writer at the head of the queue has been waiting longer than this 
    // duration, the lock enters Override Mode, heavily prioritizing writers.
    static constexpr duration maxWriterWait         = std::chrono::milliseconds(50);
    
    // The duration of the "override" state, giving writers priority to drain their queue.
    // Once Override Mode is triggered, this is how long the lock stays in that state.
    static constexpr duration overrideTimeslice     = std::chrono::milliseconds(100);
    
    // The time a writer waits before setting the WRITER_STARVING flag.
    // This is a "soft" warning. If a writer waits this long, it modifies the atomic state 
    // to block new fast-path readers, stopping the "reader stampede".
    static constexpr duration starvationThreshold   = std::chrono::milliseconds(3);
    
    // During an override, the maximum number of writers that can acquire the lock consecutively.
    // This prevents writers from completely locking out readers during a major writer backlog.
    static constexpr int      writerBatchLimit      = 2;
    
    // Under normal conditions, the hard limit on consecutive writers. 
    // 0 means disabled. If set, this forces writers to yield to readers even without starvation.
    static constexpr int      maxConsecWriters      = 0;
    
    // The number of times a writer will yield to the OS scheduler when it hits a limit.
    // If a writer must wait for fast-path readers to finish, it yields to the OS this many 
    // times before assuming no readers actually exist and bypassing the limit to prevent deadlocks.
    static constexpr int      maxYieldsBeforeBypass = 10;
};

// -------------------------------------------------------------------------
// FairRWLock
// 
// Architecture & Approach:
// The lock implements a hybrid two-tier locking mechanism to maximize 
// throughput while strictly bounding thread starvation.
//
// 1. The Fast Path (Lock-Free):
//    Under uncontended or light loads, threads interact solely with a 
//    aligned atomic bitmask (m_state). Readers simply fetch-add 
//    to a counter. Writers use compare-and-swap. If successful, no semaphores 
//    or mutexes are ever touched and OS transitions are avoided.
//
// 2. The Slow Path (TTAS Spinlock & Semaphores):
//    When contention occurs, threads fall back to an internal queue protected 
//    by a Test-and-Test-and-Set (TTAS) atomic spinlock. Waiting threads sleep 
//    on C++20 semaphores, mirroring the density of native kernel wait gates.
//
// 3. Fairness Policy (The Baton Handoff):
//    Unlike standard RW locks, this lock monitors queue times. If a writer 
//    waits too long, it asserts a WRITER_STARVING bit on the fast-path. This 
//    acts as a gate, forcing all new incoming readers into the slow-path queue.
//    Once the current active readers finish, the lock "hands the baton" to the 
//    starved writer.
//
// 4. NUMA-Aware Distributed Readers (Optional):
//    When EnableNUMA is true and multiple CPU sockets are detected, the lock 
//    distributes reader counts across over-aligned stripe arrays. Fast-path 
//    readers exclusively increment their localized stripe, eliminating 
//    cache-line bouncing on the central state interconnect. Writers assert a 
//    global barricade and dynamically drain the distributed counts.
//
// Recommended Use Cases:
// - High-frequency trading order books.
// - ETW/EDR or telemetry pipelines where millions of read events occur, but 
//   configuration updates (writes) must not be starved.
// - Centralized routing tables or subscriber lists in highly concurrent backends.
// -------------------------------------------------------------------------
template <typename Policy = DefaultFairRWLockPolicy, bool EnableNUMA = false>
class FairRWLock
{
public:
    using clock      = std::chrono::steady_clock;
    using duration   = clock::duration;
    using time_point = clock::time_point;
    using LoggerFn   = std::function<void(std::string_view)>;

    FairRWLock() 
    {
        if constexpr (EnableNUMA)
        {
            m_numaState.init(NumaTopology<EnableNUMA>::is_multi_node, NumaTopology<EnableNUMA>::num_stripes);
        }
    }

#ifndef NDEBUG
    // Assigns an optional diagnostic logger for debugging lock state transitions.
    // Forbids re-attachment to prevent data races and unsafe functor replacement 
    // during active lock contention.
    void SetLogger(LoggerFn lg)
    {
        // First check to avoid locking overhead if already attached.
        if (m_loggerAttached.load(std::memory_order_acquire))
        {
            throw std::logic_error("FairRWLock: Logger already attached.");
        }

        // Serialize assignment to prevent concurrent re-attachments.
        QueueLockGuard lk(m_queueLock);
        
        // Double-check under the lock.
        if (m_loggerAttached.load(std::memory_order_relaxed))
        {
            throw std::logic_error("FairRWLock: Logger already attached.");
        }

        m_logger = std::move(lg);
        
        // Publish the logger availability to all threads. This establishes a 
        // strict happens-before relationship, guaranteeing the functor is fully 
        // visible before any thread evaluates it.
        m_loggerAttached.store(true,
                               std::memory_order_release);
    }
#else
    // Zero-cost abstraction for Release builds
    void SetLogger(LoggerFn) noexcept
    {
    }
#endif

    // -------------------------------------------------------------------------
    // RAII Guards
    // -------------------------------------------------------------------------

    // A scoped RAII guard for safely managing Read locks.
    class ReadGuard
    {
    public:
        // Attempts to acquire a read lock, optionally bound by a timeout.
        explicit ReadGuard(FairRWLock& l, duration timeout = (duration::max)()) : m_lock(&l)
        {
            m_locked = m_lock->ReadLock(timeout);
        }

        // Releases the read lock if it is currently owned.
        ~ReadGuard() noexcept
        {
            if (m_locked)
            {
                m_lock->ReadUnlock();
            }
        }

        ReadGuard(const ReadGuard&) = delete;
        ReadGuard& operator=(const ReadGuard&) = delete;

        // Transfers lock ownership from another ReadGuard.
        ReadGuard(ReadGuard&& o) noexcept : m_lock(o.m_lock), m_locked(o.m_locked)
        {
            o.m_locked = false;
            o.m_lock   = nullptr;
        }

        // Move-assigns a ReadGuard, unlocking the current instance if necessary.
        ReadGuard& operator=(ReadGuard&& o) noexcept
        {
            if (this == &o)
            {
                return *this;
            }
            
            if (m_locked)
            {
                m_lock->ReadUnlock();
            }

            m_lock   = o.m_lock;
            m_locked = o.m_locked;

            o.m_locked = false;
            o.m_lock   = nullptr;
            
            return *this;
        }

        // Evaluates to true if the lock was successfully acquired.
        explicit operator bool() const noexcept
        {
            return m_locked;
        }

    private:
        FairRWLock* m_lock;
        bool        m_locked = false;
    };

    // A scoped RAII guard for safely managing Write locks.
    class WriteGuard
    {
    public:
        // Attempts to acquire an exclusive write lock, optionally bound by a timeout.
        explicit WriteGuard(FairRWLock& l, duration timeout = (duration::max)()) : m_lock(&l)
        {
            m_locked = m_lock->WriteLock(timeout);
        }

        // Releases the write lock if it is currently owned.
        ~WriteGuard() noexcept
        {
            if (m_locked)
            {
                m_lock->WriteUnlock();
            }
        }

        WriteGuard(const WriteGuard&) = delete;
        WriteGuard& operator=(const WriteGuard&) = delete;

        // Transfers lock ownership from another WriteGuard.
        WriteGuard(WriteGuard&& o) noexcept : m_lock(o.m_lock), m_locked(o.m_locked)
        {
            o.m_locked = false;
            o.m_lock   = nullptr;
        }

        // Move-assigns a WriteGuard, unlocking the current instance if necessary.
        WriteGuard& operator=(WriteGuard&& o) noexcept
        {
            if (this == &o)
            {
                return *this;
            }

            if (m_locked)
            {
                m_lock->WriteUnlock();
            }

            m_lock   = o.m_lock;
            m_locked = o.m_locked;

            o.m_locked = false;
            o.m_lock   = nullptr;
            
            return *this;
        }

        // Evaluates to true if the lock was successfully acquired.
        explicit operator bool() const noexcept
        {
            return m_locked;
        }

    private:
        FairRWLock* m_lock;
        bool        m_locked = false;
    };

    // -------------------------------------------------------------------------
    // Manual Locking Interface
    // -------------------------------------------------------------------------

// -------------------------------------------------------------------------
    // Acquires shared read access. 
    // Blocks the thread until the lock is acquired or the optional timeout
    // expires.
    // Returns true on success, false if the timeout was reached.
    // -------------------------------------------------------------------------
    bool ReadLock(duration timeout = (duration::max)())
    {
        // --- TTA (Test-and-Test-and-Add) Fast-Path Filter ---
        // Prevents speculative RMW cache invalidation storms when lock is held.
        uint64_t current_s = m_state.load(std::memory_order_relaxed);
        
        if ((current_s & (WRITE_LOCKED | WRITER_STARVING)) != 0) [[unlikely]]
        {
            return slow_read_lock(timeout);
        }

        if constexpr (EnableNUMA)
        {
            // Dynamically bypasses distributed logic when running strictly on a single node
            if (NumaTopology<EnableNUMA>::is_multi_node) 
            {
                size_t idx = get_stripe_index();
                
                // memory_order_seq_cst enforces strict Store-Load ordering across ARM/PowerPC
                m_numaState.stripes[idx].count.fetch_add(1, 
                                                         std::memory_order_seq_cst);
                                
                uint64_t s = m_state.load(std::memory_order_seq_cst);

                // Fast Path: Only successfully acquired if no writers are active/starving.
                if ((s & (WRITE_LOCKED | WRITER_STARVING)) == 0) [[likely]]
                {
                    if ((s >> 32) > 0) [[unlikely]]
                    {                
                        m_state.fetch_and(STATE_MASK, 
                                          std::memory_order_relaxed);
                    }
                    
                    return true;
                }

                // Rollback the speculative local increment with SeqCst to avoid 
                // the 0-observation distributed counter anomaly.
                uint32_t old_c = m_numaState.stripes[idx].count.fetch_sub(1, 
                                                                          std::memory_order_seq_cst);
                
                if (old_c == 1 && get_total_readers() == 0)
                {                    
                    uint64_t current_s_seq = m_state.load(std::memory_order_seq_cst);
                    
                    if ((current_s_seq & HAS_WAITERS) && !(current_s_seq & WRITE_LOCKED)) [[unlikely]]
                    {
                        slow_read_unlock();
                    }
                }

                return slow_read_lock(timeout);
            }
        }

        // Lock-Free Acquisition: Bypasses CAS loops for O(N) throughput.
        // Hardware unconditionally resolves parallel fetches immediately.        
        uint64_t s = m_state.fetch_add(READ_INC,
                                       std::memory_order_acquire);

        // Guard against bitmask overflow into state flags
        if ((s & OVERFLOW_GUARD) != 0) [[unlikely]]
        {
            m_state.fetch_sub(READ_INC,
                              std::memory_order_release);
                              
            return false;
        }

        // Fast Path: Only successfully acquired if no writers are active/starving.
        if ((s & (WRITE_LOCKED | WRITER_STARVING)) == 0) [[likely]]
        {
            // Safely and idempotently clear the consecutive writer streak 
            // from the upper 32 bits without risking active reader logic.
            if ((s >> 32) > 0) [[unlikely]]
            {                
                m_state.fetch_and(STATE_MASK, 
                                  std::memory_order_relaxed);
            }

            return true;
        }

        // Rollback the speculative increment, the lock is contested.
        uint64_t old_s = m_state.fetch_sub(READ_INC,
                                           std::memory_order_release);
        
        // Critical Baton Pass: If our rollback inadvertently brought the reader count 
        // to zero while writers/readers are waiting, we must wake them up.
        if ((old_s & READ_MASK) == 1 && (old_s & HAS_WAITERS) && !(old_s & WRITE_LOCKED)) [[unlikely]]
        {
            slow_read_unlock();
        }

        return slow_read_lock(timeout);
    }

    // -------------------------------------------------------------------------
    // Attempts to acquire shared read access without blocking.
    // Returns true immediately if successful, false if the lock is heavily 
    // contended or locked.
    // -------------------------------------------------------------------------
    bool TryReadLock() noexcept
    {
        if constexpr (EnableNUMA)
        {
            // Dynamically bypasses distributed logic when running strictly on a single node
            // to maximize L1 cache locality and avoid hashing overhead.
            if (NumaTopology<EnableNUMA>::is_multi_node) 
            {
                uint64_t current_s = m_state.load(std::memory_order_relaxed);
                
                // --- TTA Fast-Path Filter ---
                if ((current_s & (WRITE_LOCKED | WRITER_STARVING)) == 0) [[likely]]
                {
                    size_t idx = get_stripe_index();
                    
                    // memory_order_seq_cst enforces strict Store-Load ordering across ARM/PowerPC
                    m_numaState.stripes[idx].count.fetch_add(1, 
                                                             std::memory_order_seq_cst);
                    
                    // Fetch the central state to check for writer barricades or starvation flags.                
                    uint64_t s = m_state.load(std::memory_order_seq_cst);

                    // Fast Path: Only successfully acquired if no writers are active/starving.
                    if ((s & (WRITE_LOCKED | WRITER_STARVING)) == 0) [[likely]]
                    {
                        if ((s >> 32) > 0) [[unlikely]]
                        {
                            m_state.fetch_and(STATE_MASK, 
                                              std::memory_order_relaxed);
                        }
                        
                        return true;
                    }

                    // Contention detected. We must rollback the speculative local increment.
                    //
                    // Sequential consistency (SeqCst) is mandatory here. If Thread A and Thread B 
                    // rollback simultaneously on different stripes, the hardware memory fence 
                    // guarantees at least one thread observes get_total_readers() == 0 and triggers 
                    // the slow_read_unlock() baton pass, preventing queued writers from starving indefinitely.
                    uint32_t old_c = m_numaState.stripes[idx].count.fetch_sub(1, 
                                                                              std::memory_order_seq_cst);
                    
                    if (old_c == 1 && get_total_readers() == 0)
                    {                    
                        uint64_t current_s_seq = m_state.load(std::memory_order_seq_cst);
                        
                        if ((current_s_seq & HAS_WAITERS) && !(current_s_seq & WRITE_LOCKED)) [[unlikely]]
                        {                        
                            slow_read_unlock(); 
                        }
                    }
                }

                // Fast path failed; attempt to acquire the slow-path queue spinlock without blocking.
                QueueLockGuard lk(m_queueLock, 
                                  std::try_to_lock);
                
                if (!lk.owns_lock())
                {
                    return false;
                }

                check_and_trigger_override_unlocked();
                uint64_t s = m_state.load(std::memory_order_relaxed);
                
                // Attempt to acquire the read lock while holding the queue lock.
                while (can_reader_acquire_unlocked(s))
                {
                    // Apply local stripe increment first
                    size_t idx = get_stripe_index();
                    m_numaState.stripes[idx].count.fetch_add(1, 
                                                             std::memory_order_seq_cst);
                    
                    // Verify global state hasn't shifted to WRITE_LOCKED right as we incremented                    
                    s = m_state.load(std::memory_order_seq_cst);
                    
                    if (can_reader_acquire_unlocked(s))
                    {
                        m_batchWriters.store(0, 
                                             std::memory_order_relaxed);
                                             
                        // Clear the consecutive writer streak from the upper 32 bits to prevent
                        // NUMA streak leakage.
                        if ((s >> 32) > 0) [[unlikely]]
                        {                
                            m_state.fetch_and(STATE_MASK, 
                                              std::memory_order_relaxed);
                        }
                                             
                        return true;
                    }
                    
                    // Global state shifted (e.g., a writer snuck in). Rollback and evaluate handoff.
                    uint32_t old_c = m_numaState.stripes[idx].count.fetch_sub(1, 
                                                                              std::memory_order_seq_cst);
                    
                    if (old_c == 1 && get_total_readers() == 0)
                    {
                        if (m_headWriter.load(std::memory_order_relaxed) != nullptr)
                        {
                            notify_next_writer_unlocked();
                        }
                    }
                    
                    s = m_state.load(std::memory_order_relaxed);
                }
                
                return false;
            }
        }

        uint64_t current_s = m_state.load(std::memory_order_relaxed);
        
        // --- TTA Fast-Path Filter ---
        if ((current_s & (WRITE_LOCKED | WRITER_STARVING)) == 0) [[likely]]
        {
            // Lock-Free Acquisition: Bypasses CAS loops for O(N) throughput.        
            // Falls through here if EnableNUMA=false or multi_node=false.
            uint64_t s = m_state.fetch_add(READ_INC, 
                                           std::memory_order_acquire);
            
            // Guard against bitmask overflow.
            if ((s & OVERFLOW_GUARD) != 0) [[unlikely]]
            {
                // Rollback if maximum reader capacity is reached.
                m_state.fetch_sub(READ_INC, 
                                  std::memory_order_release);
                                  
                return false;
            }

            // Fast Path: Check if no writers are active and no writers are starving.
            if ((s & (WRITE_LOCKED | WRITER_STARVING)) == 0) [[likely]]
            {
                // Clear the consecutive writer streak if present.
                if ((s >> 32) > 0) [[unlikely]]
                {                
                    m_state.fetch_and(STATE_MASK, 
                                      std::memory_order_relaxed);
                }

                return true;
            }

            // Rollback the speculative increment because the lock is contended.
            uint64_t old_s = m_state.fetch_sub(READ_INC, 
                                               std::memory_order_release);
            
            // Critical Baton Pass: Wake waiting threads if we were the last reader.
            if ((old_s & READ_MASK) == 1 && (old_s & HAS_WAITERS) && !(old_s & WRITE_LOCKED)) [[unlikely]]
            {            
                slow_read_unlock();
            }
        }

        // Try to acquire the slow-path queue lock without blocking.
        QueueLockGuard lk(m_queueLock, 
                          std::try_to_lock);
        
        // Return immediately if the queue lock is heavily contended.
        if (!lk.owns_lock())
        {
            return false;
        }

        // Evaluate if Override Mode needs to be activated.
        check_and_trigger_override_unlocked();
        uint64_t s = m_state.load(std::memory_order_relaxed);
        
        // Attempt to acquire the read lock while holding the queue lock.
        while (can_reader_acquire_unlocked(s))
        {
            uint64_t next_s = (s + READ_INC) & STATE_MASK;

            // Use weak CAS since we are looping.
            if (m_state.compare_exchange_weak(s, 
                                              next_s, 
                                              std::memory_order_acquire, 
                                              std::memory_order_relaxed))
            {
                // Reset the writer batch count since a reader successfully acquired.
                m_batchWriters.store(0, 
                                     std::memory_order_relaxed);

                return true;
            }
        }
        
        // The lock cannot be acquired immediately.
        return false;
    }

    // -------------------------------------------------------------------------
    // Releases shared read access.
    // If this is the last reader and a writer is waiting, this function acts as the trigger
    // to wake up the slow-path writer queue.
    // -------------------------------------------------------------------------
    void ReadUnlock() noexcept
    {
        if constexpr (EnableNUMA)
        {
            if (NumaTopology<EnableNUMA>::is_multi_node) 
            {
                size_t idx = get_stripe_index();
                
                // Sequence consistency fence ensures zero-observation is safely published
                uint32_t old_c = m_numaState.stripes[idx].count.fetch_sub(1, 
                                                                          std::memory_order_seq_cst);
                
                assert(old_c > 0 && "ReadUnlock called without corresponding ReadLock");

                if (old_c == 1)
                {
                    if (get_total_readers() == 0)
                    {
                        // Load with seq_cst to prevent ARM64 store-load reordering
                        uint64_t s = m_state.load(std::memory_order_seq_cst);
                        
                        if ((s & HAS_WAITERS) && !(s & WRITE_LOCKED)) [[unlikely]]
                        {
                            slow_read_unlock();
                        }
                    }
                }
                
                return;
            }
        }

        // 'release' optimally ensures all prior memory operations inside the critical section 
        // are globally visible to the next writer.
        uint64_t old_s = m_state.fetch_sub(READ_INC, 
                                           std::memory_order_release);
                                           
        uint32_t readers_before = static_cast<uint32_t>(old_s & READ_MASK);
        
        assert(readers_before > 0 && "ReadUnlock called without corresponding ReadLock");

        if (readers_before == 1 && (old_s & HAS_WAITERS)) [[unlikely]]
        {
            // The inner QueueLockGuard inside slow_read_unlock() establishes the 
            // required 'acquire' barrier for safely interacting with the queue list.
            slow_read_unlock();
        }
    }

    // -------------------------------------------------------------------------
    // Acquires exclusive write access.
    // Blocks the thread until the lock is acquired or the optional timeout expires.
    // Returns true on success, false if the timeout was reached.
    // -------------------------------------------------------------------------
    bool WriteLock(duration timeout = (duration::max)())
    {
        uint64_t s = m_state.load(std::memory_order_acquire);

        while (true)
        {
            uint32_t consec = static_cast<uint32_t>(s >> 32);

            if (Policy::maxConsecWriters == 0 || consec < static_cast<uint32_t>(Policy::maxConsecWriters)) [[likely]]
            {
                bool readers_clear = true;
                
                if constexpr (EnableNUMA) 
                {
                    // For NUMA, we must query the distributed stripe arrays to verify no active readers.
                    if (NumaTopology<EnableNUMA>::is_multi_node) 
                    {
                        readers_clear = (get_total_readers() == 0);
                    }
                }
                
                // Fast Path: Ensures fast-path fails instantly if any readers, writers, or flags exist.
                // NOTE: For NUMA, the lower 32-bits of `s` do not contain the reader count, so `readers_clear`
                // acts as the authoritative gate.
                if ((s & STATE_MASK) == 0 && readers_clear) [[likely]]
                {
                    uint32_t next_consec = (consec < 0xFFFFFFFF) ? consec + 1 : consec;
                    uint64_t next_s = WRITE_LOCKED | (static_cast<uint64_t>(next_consec) << 32);
                    
                    // Optimized to compare_exchange_weak to avoid nested hardware spin-loops on ARM/PowerPC
                    // Utilizing memory_order_seq_cst establishes absolute store-load ordering against remote stripes
                    if (m_state.compare_exchange_weak(s, 
                                                      next_s, 
                                                      std::memory_order_seq_cst, 
                                                      std::memory_order_relaxed)) [[likely]]
                    {                        
                        // By executing the CAS above, this thread globally sets the  WRITE_LOCKED bit. 
                        // This barricade physically prevents any new readers from acquiring their 
                        // local stripes. We then spin-wait efficiently until any residual readers finish draining.
                        drain_numa_readers();
                        return true;
                    }

                    continue;
                }
            }

            break;
        }

        return slow_write_lock(timeout);
    }

    // -------------------------------------------------------------------------
    // Attempts to acquire exclusive write access without blocking.
    // Returns true immediately if successful, false if any readers or writers 
    // are active.
    // -------------------------------------------------------------------------
    bool TryWriteLock() noexcept
    {
        uint64_t s = m_state.load(std::memory_order_acquire);
        uint32_t consec = static_cast<uint32_t>(s >> 32);

        if (Policy::maxConsecWriters == 0 || consec < static_cast<uint32_t>(Policy::maxConsecWriters)) [[likely]]
        {
            bool readers_clear = true;
            
            if constexpr (EnableNUMA) 
            {
                if (NumaTopology<EnableNUMA>::is_multi_node) 
                {
                    readers_clear = (get_total_readers() == 0);
                }
            }
            
            // Evaluate global flags and aggregate NUMA reader clearance simultaneously.
            if ((s & STATE_MASK) == 0 && readers_clear) [[likely]]
            {
                uint32_t next_consec = (consec < 0xFFFFFFFF) ? consec + 1 : consec;
                uint64_t next_s = WRITE_LOCKED | (static_cast<uint64_t>(next_consec) << 32);
                
                if (m_state.compare_exchange_strong(s, 
                                                    next_s, 
                                                    std::memory_order_seq_cst, 
                                                    std::memory_order_relaxed)) [[likely]]
                {
                    if constexpr (EnableNUMA)
                    {
                        // A reader thread could have executed its fetch_add() on a remote NUMA node 
                        // strictly in the nanoseconds between our `readers_clear` evaluation and the CAS 
                        // execution.
                        if (NumaTopology<EnableNUMA>::is_multi_node && get_total_readers() > 0)
                        {
                            // A reader snuck in. We implement a highly bounded spin to avoid phantom aborts
                            // while rigidly maintaining the non-blocking TryLock specification.
                            bool readers_cleared = false;
                            
                            for (int i = 0; i < 32; ++i)
                            {
                                if (get_total_readers() == 0)
                                {
                                    readers_cleared = true;
                                    break;
                                }
                                
                                cpu_relax_pause();
                            }
                            
                            if (!readers_cleared)
                            {
                                // Revert the speculative writer streak increment before aborting
                                // to prevent permanently polluting the fairness tracker.
                                if (next_consec > consec)
                                {
                                    m_state.fetch_sub(1ULL << 32, 
                                                      std::memory_order_relaxed);
                                }

                                WriteUnlock();
                                return false;
                            }
                        }
                    }
                    
                    drain_numa_readers();
                    return true;
                }
            }
        }

        QueueLockGuard lk(m_queueLock, 
                          std::try_to_lock);
        
        if (!lk.owns_lock())
        {
            return false;
        }

        check_and_trigger_override_unlocked();
        s = m_state.load(std::memory_order_relaxed);
        
        while (can_writer_acquire_unlocked(s, false))
        {
            consec = static_cast<uint32_t>(s >> 32);
            uint32_t next_consec = (consec < 0xFFFFFFFF) ? consec + 1 : consec;
            
            uint64_t next_s = (s | WRITE_LOCKED) & STATE_MASK;
            
            // Barging writers must forcibly unblock fast-path readers since they bypass queues
            next_s &= ~WRITER_STARVING; 
            
            next_s |= (static_cast<uint64_t>(next_consec) << 32);

            if (m_state.compare_exchange_weak(s, 
                                              next_s, 
                                              std::memory_order_seq_cst, 
                                              std::memory_order_relaxed))
            {
                update_writer_batch_unlocked();
                
                // Drop the queue lock first to minimize OS-level critical section time.
                lk.unlock();
                
                if constexpr (EnableNUMA)
                {
                    // Exact same TOCTOU rollback applies when acquiring via the slow-path loop.
                    if (NumaTopology<EnableNUMA>::is_multi_node && get_total_readers() > 0)
                    {
                        bool readers_cleared = false;
                        
                        for (int i = 0; i < 32; ++i)
                        {
                            if (get_total_readers() == 0)
                            {
                                readers_cleared = true;
                                break;
                            }
                            
                            cpu_relax_pause();
                        }
                        
                        if (!readers_cleared)
                        {
                            // Revert the speculative writer streak increment safely before aborting
                            if (next_consec > consec)
                            {
                                m_state.fetch_sub(1ULL << 32, 
                                                  std::memory_order_relaxed);
                            }
                            
                            // Revert the batch writer increment if override was active
                            if (m_writerOverride.load(std::memory_order_relaxed))
                            {
                                m_batchWriters.fetch_sub(1, 
                                                         std::memory_order_relaxed);
                            }

                            WriteUnlock();
                            return false;
                        }
                    }
                }
                
                drain_numa_readers();

                return true;
            }
        }
        
        return false;
    }

    // -------------------------------------------------------------------------
    // Releases exclusive write access.
    // Checks policy limits. If consecutive limits are hit, it wakes up waiting readers 
    // instead of the next writer to maintain strict fairness.
    // -------------------------------------------------------------------------
    void WriteUnlock() noexcept
    {
        uint64_t s = m_state.load(std::memory_order_relaxed);
        uint32_t consec = static_cast<uint32_t>(s >> 32);

        // Determine if the maximum consecutive writer limits have been reached.
        bool consecCap   = (Policy::maxConsecWriters > 0 && consec >= static_cast<uint32_t>(Policy::maxConsecWriters));
        
        // Determine if the writer override batch limits have been reached.
        bool overrideCap = (m_writerOverride.load(std::memory_order_relaxed) && writer_limits_reached_unlocked(consec));

        // Fast Path Unlock: If we haven't hit caps, try to cleanly clear the write bit while preserving the streak count.
        if (!consecCap && !overrideCap) [[likely]]
        {
            uint64_t expected = WRITE_LOCKED | (static_cast<uint64_t>(consec) << 32);
            uint64_t next_s = (static_cast<uint64_t>(consec) << 32);
            
            // Ensure globally visible state with memory_order_release.
            if (m_state.compare_exchange_strong(expected, 
                                                next_s, 
                                                std::memory_order_release, 
                                                std::memory_order_relaxed)) [[likely]]
            {
                return;
            }
        }

        // Fall back to the slow path if limits were hit or the CAS failed.
        slow_write_unlock();
    }

#ifndef NDEBUG
    // -------------------------------------------------------------------------
    // Outputs a diagnostic message if a logger is attached.
    // -------------------------------------------------------------------------
    void debug_log(std::string_view s) const noexcept
    {
        // Utilize memory_order_acquire to safely evaluate attachment state 
        // without entering the slow-path mutex.
        if (m_loggerAttached.load(std::memory_order_acquire))
        {
            try
            {
                std::ostringstream oss;
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now().time_since_epoch()).count();
                oss << "[" << ms << " ms] " << s;
                m_logger(oss.str());
            }
            catch (...)
            {
            }
        }
    }
#else
    void debug_log(std::string_view) const noexcept
    {
    }
#endif

private:
    // -------------------------------------------------------------------------
    // 64-Bit State Packing Bitmask Definitions
    // 
    // Upper 32 Bits: Number of consecutive writers (cleared entirely by next reader).
    // Lower 32 Bits: Active reader count and state flags.
    // -------------------------------------------------------------------------
    static constexpr uint64_t READ_INC        = 0x0000000000000001ULL;
    static constexpr uint64_t READ_MASK       = 0x0000000007FFFFFFULL; // Support up to ~134 Million Concurrent Readers
    static constexpr uint64_t OVERFLOW_GUARD  = 0x0000000008000000ULL; 
    static constexpr uint64_t WRITER_STARVING = 0x0000000010000000ULL; 
    static constexpr uint64_t WRITE_LOCKED    = 0x0000000040000000ULL; 
    static constexpr uint64_t HAS_WAITERS     = 0x0000000080000000ULL;
    static constexpr uint64_t STATE_MASK      = 0x00000000FFFFFFFFULL;

    // Intrusive linked-list node for the writer wait queue.
    // This avoids dynamic allocation during lock acquisition.
    struct WriterNode
    {
        WriterNode*           prev = nullptr;
        WriterNode*           next = nullptr;
        std::binary_semaphore sem{ 0 };
        time_point            ts;
    };

// -------------------------------------------------------------------------
    // Test-and-Test-and-Set (TTAS) Spinlock Guard
    // Safely replaces std::unique_lock for low-level queue pointer manipulation.
    // -------------------------------------------------------------------------
    class QueueLockGuard
    {
    public:
        explicit QueueLockGuard(std::atomic_flag& flag) : m_flag(&flag)
        {
            lock();
        }

        QueueLockGuard(std::atomic_flag& flag, std::try_to_lock_t) : m_flag(&flag)
        {
            m_owns = !m_flag->test_and_set(std::memory_order_acquire);
        }

        ~QueueLockGuard()
        {
            if (m_owns)
            {
                unlock();
            }
        }

        void lock()
        {
            int backoff = 1;
            constexpr int MAX_BACKOFF = 64;

            while (m_flag->test_and_set(std::memory_order_acquire))
            {
                while (m_flag->test(std::memory_order_relaxed))
                {
                    // Exponential backoff to prevent interconnect flooding (Thundering Herd)
                    // and mitigate 1-cycle yield instruction saturation on ARM64.
                    for (int i = 0; i < backoff; ++i)
                    {
                        cpu_relax_pause();
                    }
                    
                    // Enclosed std::min in parentheses to bypass Windows min/max macros
                    backoff = (std::min)(backoff << 1, MAX_BACKOFF);
                }
            }
            
            m_owns = true;
        }

        void unlock()
        {
            m_flag->clear(std::memory_order_release);
            m_owns = false;
        }

        bool owns_lock() const noexcept
        {
            return m_owns;
        }

    private:
        std::atomic_flag* m_flag;
        bool              m_owns = false;
    };

    // RAII helper to ensure a waiting writer is always safely unlinked from the queue
    // even if it times out, throws an exception, or is forcefully interrupted.
    struct ScopeUnlinker
    {
        FairRWLock* lock;
        WriterNode* node;
        bool        isActive;

        ScopeUnlinker(FairRWLock* l, WriterNode* n) : lock(l), node(n), isActive(true)
        {
        }

        ~ScopeUnlinker()
        {
            if (isActive)
            {
                bool wasHead = (lock->m_headWriter.load() == node);
                
                lock->unlink_writer_unlocked(node);
                lock->update_has_waiters_unlocked();

                // If the thread abandoning the queue was at the front, we must wake the next in line.
                if (wasHead)
                {
                    if (lock->m_headWriter.load() != nullptr)
                    {
                        lock->notify_next_writer_unlocked();
                    }
                    else
                    {
                        lock->notify_all_readers_unlocked();
                    }
                }
            }
        }
    };

    // Identifies a distributed memory stripe using an immutable thread-local hash.
    //
    // ARCHITECTURAL NOTE: 
    // While OS APIs like sched_getcpu() or GetCurrentProcessorNumber() provide exact 
    // physical topologies, they introduce vDSO/API overhead (~10-20 cycles) that 
    // severely degrades lock-free fast-path throughput. 
    // 
    // This TLS hash executes in ~2 ALU cycles. If the OS migrates the thread to a 
    // different NUMA node, the thread may write to a remote stripe. However, since 
    // the access remains strictly UNCONTENDED, it completely avoids cache-line bouncing 
    // and remains substantially faster than synchronizing on a central state.
    inline size_t get_stripe_index() const noexcept
    {
        if constexpr (EnableNUMA)
        {
            if (!NumaTopology<EnableNUMA>::is_multi_node)
            {
                return 0;
            }

            // Constant-initialized TLS guarantees zero runtime allocation cost
            static std::atomic<size_t> global_thread_counter{ 0 };
            thread_local const size_t thread_id = global_thread_counter.fetch_add(1, 
                                                                                  std::memory_order_relaxed);

            // Exploits zero-cost bitwise modulo since num_stripes is a power of 2
            return thread_id & (NumaTopology<EnableNUMA>::num_stripes - 1);
        }
        else
        {
            return 0;
        }
    }

    // Unifies the total reader count logic across both architectures
    uint32_t get_total_readers() const noexcept
    {
        if constexpr (EnableNUMA)
        {
            if (NumaTopology<EnableNUMA>::is_multi_node)
            {
                uint32_t total = 0;
                
                for (size_t i = 0; i < NumaTopology<EnableNUMA>::num_stripes; ++i) 
                {
                    total += m_numaState.stripes[i].count.load(std::memory_order_seq_cst);
                }
                
                return total;
            }
        }
        
        return static_cast<uint32_t>(m_state.load(std::memory_order_relaxed) & READ_MASK);
    }

    // Drains active distributed readers waiting for their stripe counters to reach zero
    // Actively relies on Ticket-Lock semantics established by setting WRITE_LOCKED prior to call
    inline void drain_numa_readers() const noexcept
    {
        if constexpr (EnableNUMA)
        {
            if (NumaTopology<EnableNUMA>::is_multi_node)
            {
                for (size_t i = 0; i < NumaTopology<EnableNUMA>::num_stripes; ++i)
                {
                    while (m_numaState.stripes[i].count.load(std::memory_order_seq_cst) > 0)
                    {
                        cpu_relax_pause();
                    }
                }
            }
        }
    }

    // Handles locking when the fast-path fails for readers.
    // Registers the reader in the wait counter and waits on the reader condition variable.
    bool slow_read_lock(duration timeout)
    {
        LOG_DEBUG(this, "Reader entered slow path");

        // Increment the slow-path reader counter. This tracks how many readers 
        // are currently spinning or sleeping, which is used for baton-passing.
        m_readersWaiting.fetch_add(1,
                                   std::memory_order_acquire);
                                   
        // Set the HAS_WAITERS bit so current fast-path owners know to invoke 
        // the slow-path unlock routines when they finish.
        m_state.fetch_or(HAS_WAITERS,
                         std::memory_order_relaxed);

        // Acquire the queue spinlock.
        QueueLockGuard lk(m_queueLock);
        
        // Check if any queued writers have exceeded their wait thresholds.
        check_and_trigger_override_unlocked();

        time_point deadline = (timeout == (duration::max)()) ? (time_point::max)() : clock::now() + timeout;
        bool acquired = false;

        while (true)
        {
            uint64_t s = m_state.load(std::memory_order_relaxed);
            
            // Check if current lock policies allow a reader to proceed.
            // This returns false if a writer is active or starving.
            if (can_reader_acquire_unlocked(s))
            {
                bool attempt_centralized = true;

                if constexpr (EnableNUMA)
                {
                    if (NumaTopology<EnableNUMA>::is_multi_node) 
                    {
                        attempt_centralized = false;
                        size_t idx          = get_stripe_index();
                        
                        // Register the read on the thread's local NUMA stripe.
                        m_numaState.stripes[idx].count.fetch_add(1, 
                                                                 std::memory_order_seq_cst);
                        
                        s = m_state.load(std::memory_order_seq_cst);
                        
                        // Verify that the global state is still valid after our local increment.
                        if (can_reader_acquire_unlocked(s))
                        {
                            m_batchWriters.store(0, 
                                                 std::memory_order_relaxed);
                            acquired = true;
                            
                            // Clear the consecutive writer streak from the upper 32 bits to
                            // prevent NUMA streak leakage.
                            if ((s >> 32) > 0) [[unlikely]]
                            {                
                                m_state.fetch_and(STATE_MASK, 
                                                  std::memory_order_relaxed);
                            }
                            
                            break;
                        }
                        
                        // Global state changed (e.g. a writer set WRITE_LOCKED). 
                        // Rollback the local counter using sequential consistency to ensure 
                        // that writers observing `get_total_readers()` see synchronized values.
                        uint32_t old_c = m_numaState.stripes[idx].count.fetch_sub(1, 
                                                                                  std::memory_order_seq_cst);
                        
                        // If this rollback cleared the final active reader, notify the waiting writer.
                        if (old_c == 1 && get_total_readers() == 0)
                        {
                            if (m_headWriter.load(std::memory_order_relaxed) != nullptr)
                            {
                                notify_next_writer_unlocked();
                            }
                        }
                    }
                }

                // Standard UMA logic: Attempt a global compare-and-swap to add a reader.
                if (attempt_centralized)
                {
                    uint64_t next_s = (s + READ_INC) & STATE_MASK;

                    if (m_state.compare_exchange_weak(s,
                                                      next_s,
                                                      std::memory_order_acquire,
                                                      std::memory_order_relaxed))
                    {
                        // Reset the writer batch count since a reader successfully acquired.
                        m_batchWriters.store(0,
                                             std::memory_order_relaxed);
                        acquired = true;

                        break;
                    }
                    
                    continue; // Loop naturally re-evaluates if the CAS fails.
                }
            }

            // Lock is unavailable. Update writer wait times before sleeping.
            check_and_trigger_override_unlocked();
            
            // Release the queue spinlock so other threads can proceed.
            lk.unlock();
            
            bool acquired_sem = false;
            
            // Block the thread until notified by an unlocking writer or timeout.
            if (deadline == (time_point::max)())
            {
                m_semReaders.acquire();
                acquired_sem = true;
            }
            else
            {
                acquired_sem = m_semReaders.try_acquire_until(deadline);
            }
            
            // Re-acquire the queue spinlock upon waking.
            lk.lock();
            
            // If the timeout expired, make a final attempt to acquire the lock.
            // This handles cases where the lock became available exactly at the deadline.
            if (!acquired_sem)
            {
                s = m_state.load(std::memory_order_relaxed);
                
                while (can_reader_acquire_unlocked(s))
                {
                    bool attempt_centralized = true;

                    if constexpr (EnableNUMA)
                    {
                        if (NumaTopology<EnableNUMA>::is_multi_node) 
                        {
                            attempt_centralized = false;
                            size_t idx          = get_stripe_index();
                            
                            m_numaState.stripes[idx].count.fetch_add(1, 
                                                                     std::memory_order_seq_cst);
                            
                            s = m_state.load(std::memory_order_seq_cst);
                            
                            if (can_reader_acquire_unlocked(s))
                            {
                                m_batchWriters.store(0, 
                                                     std::memory_order_relaxed);
                                acquired = true;
                                
                                // Clear the consecutive writer streak from the upper 32 bits to 
                                // prevent NUMA streak leakage.
                                if ((s >> 32) > 0) [[unlikely]]
                                {                
                                    m_state.fetch_and(STATE_MASK, 
                                                      std::memory_order_relaxed);
                                }
                                
                                break;
                            }
                            
                            uint32_t old_c = m_numaState.stripes[idx].count.fetch_sub(1, 
                                                                                      std::memory_order_seq_cst);
                            
                            if (old_c == 1 && get_total_readers() == 0)
                            {
                                if (m_headWriter.load(std::memory_order_relaxed) != nullptr)
                                {
                                    notify_next_writer_unlocked();
                                }
                            }
                                                                     
                            s = m_state.load(std::memory_order_relaxed);
                        }
                    }

                    if (attempt_centralized)
                    {
                        uint64_t next_s = (s + READ_INC) & STATE_MASK;

                        if (m_state.compare_exchange_weak(s,
                                                          next_s,
                                                          std::memory_order_acquire,
                                                          std::memory_order_relaxed))
                        {
                            m_batchWriters.store(0, 
                                                 std::memory_order_relaxed);
                            acquired = true;
                            
                            break;
                        }
                    }
                }

                // Exit the main loop if we didn't acquire it after the timeout.
                break;
            }
        }

        // Cleanup: Thread is exiting the slow path. Decrement the waiting counter.
        m_readersWaiting.fetch_sub(1,
                                   std::memory_order_release);
                                   
        // Update the global flag to reflect if any threads are still waiting.
        update_has_waiters_unlocked();

        // If the thread timed out, and no other readers are waiting, wake up the next writer.
        if (!acquired && m_readersWaiting.load(std::memory_order_relaxed) == 0 && m_headWriter.load() != nullptr)
        {
            notify_next_writer_unlocked();
        }

        if (!acquired) [[unlikely]]
        {
            return false;
        }

        return true;
    }

    // Handles the baton-passing logic when the last active reader leaves.
    // Evaluates policies to decide whether to wake a writer or a batch of queued readers.    
    void slow_read_unlock() noexcept
    {
        // Acquire the queue spinlock for safe queue manipulation.
        QueueLockGuard lk(m_queueLock);
        
        uint64_t s      = m_state.load(std::memory_order_relaxed);
        uint32_t consec = static_cast<uint32_t>(s >> 32);

        // Check if there are writers waiting in the queue.
        if (m_headWriter.load() != nullptr)
        {
            // If writers have exceeded their override caps, give the baton back to readers.
            if (m_writerOverride.load(std::memory_order_relaxed) && writer_limits_reached_unlocked(consec) && 
                m_readersWaiting.load(std::memory_order_relaxed) > 0)
            {
                // Drop the starving bit to allow fast-path readers again.
                m_state.fetch_and(~WRITER_STARVING, 
                                  std::memory_order_relaxed);
                                  
                // Wake all waiting readers.
                notify_all_readers_unlocked();
            }
            else
            {
                // Wake the next writer in line.
                notify_next_writer_unlocked();
            }
        }
        else
        {
            // If no writers are waiting but we were in an override, clear it and wake readers.
            if (m_writerOverride.load(std::memory_order_relaxed))
            {
                m_state.fetch_and(~WRITER_STARVING, 
                                  std::memory_order_relaxed);
                                  
                notify_all_readers_unlocked();
            }
        }
    }

    // Handles locking when the fast-path fails for writers.
    // Safely links the writer into an intrusive linked list, enforcing starvation thresholds.
    bool slow_write_lock(duration timeout)
    {
        LOG_DEBUG(this, "Writer entered slow path");

        // Broadcast to fast-path users that threads are waiting in the queue, 
        // ensuring they trigger the slow-path wake mechanisms upon unlocking.
        m_state.fetch_or(HAS_WAITERS, 
                         std::memory_order_relaxed);
        
        // Acquire the atomic spinlock to safely modify the writer linked list.
        QueueLockGuard lk(m_queueLock);
        
        // Evaluate if the lead writer has been waiting too long.
        check_and_trigger_override_unlocked();

        // Allocate the queue node on the thread's local stack to avoid heap 
        // allocation overhead during contention.
        WriterNode myNode;
        myNode.ts = clock::now();
        link_writer_unlocked(&myNode);
        update_has_waiters_unlocked();

        // RAII guard ensures the node is safely unlinked if an exception occurs 
        // or the function returns early due to a timeout.
        ScopeUnlinker unlinker(this, &myNode);
        time_point deadline = (timeout == (duration::max)()) ? (time_point::max)() : clock::now() + timeout;

        // Establish the exact time this specific writer should trigger starvation mode.
        time_point starve_time = myNode.ts + Policy::starvationThreshold; 
        bool asserted_starvation = false;

        bool acquired = false;
        int yield_count = 0;

        while (true)
        {
            bool is_head = (m_headWriter.load() == &myNode);
            
            if (is_head)
            {
                uint64_t s = m_state.load(std::memory_order_relaxed);
                uint32_t consec = static_cast<uint32_t>(s >> 32);
                
                // Graceful Degradation: If we hit a policy limit, back off briefly. 
                // This gives fast-path readers time to register without hard-sleeping the writer.
                if (writer_limits_reached_unlocked(consec) && 
                    yield_count < Policy::maxYieldsBeforeBypass && timeout > duration::zero())
                {
                    // Release the queue lock so other queue operations can proceed while we back off.
                    lk.unlock();

                    // 1. Hardware Spin Phase: Wait out transient fast-path operations
                    // without invoking an expensive OS context switch.
                    constexpr int SPIN_ITERS = 16;
                    
                    for (int i = 0; i < SPIN_ITERS; ++i)
                    {
                        cpu_relax_pause();
                    }

                    // 2. OS Yield Phase: Surrender the remaining timeslice
                    std::this_thread::yield();

                    lk.lock();

                    // Re-evaluate override policies after the yield
                    check_and_trigger_override_unlocked();
                    yield_count++;

                    continue;
                }

                // If we've yielded enough times, bypass the fairness limits to prevent a deadlock.
                bool bypass_limits = (yield_count >= Policy::maxYieldsBeforeBypass);

                // Check if the writer is logically permitted to acquire the lock (verifies no active readers).
                if (can_writer_acquire_unlocked(s, bypass_limits))
                {
                    uint32_t next_consec = (consec < 0xFFFFFFFF) ? consec + 1 : consec;
                    uint64_t next_s = (s | WRITE_LOCKED) & STATE_MASK;
                    next_s |= (static_cast<uint64_t>(next_consec) << 32);

                    // Attempt to globally set the WRITE_LOCKED flag.
                    if (m_state.compare_exchange_weak(s, 
                                                      next_s, 
                                                      std::memory_order_seq_cst, 
                                                      std::memory_order_relaxed))
                    {
                        acquired = true;
                        
                        break;
                    }

                    continue;
                }
            }

            // Determine whether the thread should sleep until the timeout or the starvation threshold.
            time_point wait_until_time = deadline;
            
            if (is_head && !asserted_starvation && wait_until_time > starve_time)
            {
                wait_until_time = starve_time;
            }

            lk.unlock();
            
            bool acquired_sem = false;
            
            // Sleep on the local stack semaphore until notified by an unlocking thread.
            if (wait_until_time == (time_point::max)())
            {
                myNode.sem.acquire();
                acquired_sem = true;
            }
            else
            {
                acquired_sem = myNode.sem.try_acquire_until(wait_until_time);
            }
            
            lk.lock();
            
            yield_count = 0; 
            
            // If the thread woke up because the starvation threshold was reached, assert the flag.
            if (is_head && !asserted_starvation && clock::now() >= starve_time)
            {
                // Trip the starvation bit to block new fast-path readers.
                m_state.fetch_or(WRITER_STARVING, 
                                 std::memory_order_relaxed);
                                 
                asserted_starvation = true;

                LOG_DEBUG(this, "Writer starvation asserted");
            }

            // If the absolute timeout was reached, make one final acquisition attempt.
            if (!acquired_sem && clock::now() >= deadline)
            {
                is_head = (m_headWriter.load() == &myNode);
                
                if (is_head)
                {
                    uint64_t s = m_state.load(std::memory_order_relaxed);
                    uint32_t consec = static_cast<uint32_t>(s >> 32);
                    bool bypass_limits = (yield_count >= Policy::maxYieldsBeforeBypass);

                    if (can_writer_acquire_unlocked(s, bypass_limits))
                    {
                        uint32_t next_consec = (consec < 0xFFFFFFFF) ? consec + 1 : consec;
                        uint64_t next_s = (s | WRITE_LOCKED) & STATE_MASK;
                        next_s |= (static_cast<uint64_t>(next_consec) << 32);

                        // Use a strong CAS for the final attempt to avoid spurious failures.
                        if (m_state.compare_exchange_strong(s, 
                                                            next_s, 
                                                            std::memory_order_seq_cst, 
                                                            std::memory_order_relaxed))
                        {
                            acquired = true;
                        }
                    }
                }

                break;
            }
        }

        if (!acquired) [[unlikely]]
        {
            return false;
        }

        // Lock acquired, we no longer need the RAII unlinker.
        unlinker.isActive = false;
        unlink_writer_unlocked(&myNode);
        update_has_waiters_unlocked();
        
        update_writer_batch_unlocked();
        
        // Exit the queue critical section before draining remote NUMA reader nodes.
        // Holding the queue lock while spinning on remote stripes would stall the entire 
        // slow-path queue system across all CPU sockets.
        lk.unlock();
        
        // Spin-wait until all distributed NUMA readers have exited their respective stripes.
        // The earlier CAS setting WRITE_LOCKED acts as the barricade to prevent new ones.
        drain_numa_readers();

        return true;
    }

    // Handles the baton passing when a writer finishes.
    void slow_write_unlock() noexcept
    {
        // Acquire the queue spinlock for safe queue manipulation.
        QueueLockGuard lk(m_queueLock);
        
        uint64_t s = m_state.load(std::memory_order_relaxed);
        uint32_t consec = static_cast<uint32_t>(s >> 32);

        // Check if any fairness caps were hit during this write phase.
        bool limits_reached = writer_limits_reached_unlocked(consec);
        
        // Hand off the baton to readers if caps were hit and readers are waiting.
        bool handoff_to_readers = limits_reached && (m_readersWaiting.load(std::memory_order_relaxed) > 0);

        uint64_t clear_mask = WRITE_LOCKED;
        
        if (handoff_to_readers || limits_reached)
        {
            clear_mask |= WRITER_STARVING;
            
            // Ensures the consecutive tracker is zeroed properly when 
            // the writer streak caps out, returning throughput to baseline.
            clear_mask |= (0xFFFFFFFFULL << 32);
        }

        // Safely clear the write lock and any optional starving flags.
        m_state.fetch_and(~clear_mask, 
                          std::memory_order_release);

        // Determine which threads to wake up.
        if (m_headWriter.load() != nullptr)
        {
            if (handoff_to_readers)
            {
                // Wake readers if we've been forced to hand off.
                notify_all_readers_unlocked();
            }
            else
            {
                if (limits_reached)
                {
                    // Safely clear override limits before letting the next writer proceed.
                    clear_override_unlocked();
                }
                
                // Wake the next writer in the queue.
                notify_next_writer_unlocked();
            }
        }
        else
        {
            // If no writers remain, default to waking any waiting readers.
            notify_all_readers_unlocked();
        }
    }

    // Synchronizes the fast-path HAS_WAITERS bit with the actual state of the slow-path queues.
    void update_has_waiters_unlocked() noexcept
    {
        if (m_headWriter.load() != nullptr || 
            m_readersWaiting.load(std::memory_order_relaxed) > 0 || 
            m_writerOverride.load(std::memory_order_relaxed))
        {
            m_state.fetch_or(HAS_WAITERS, 
                             std::memory_order_relaxed);
        }
        else
        {
            m_state.fetch_and(~(HAS_WAITERS | WRITER_STARVING), 
                              std::memory_order_relaxed);
        }
    }

    // Appends a writer node to the back of the queue.
    void link_writer_unlocked(WriterNode* node) noexcept
    {
        node->prev = m_tailWriter;
        node->next = nullptr;
        
        if (m_tailWriter != nullptr)
        {
            m_tailWriter->next = node;
        }

        m_tailWriter = node;

        if (m_headWriter.load() == nullptr)
        {
            m_headWriter.store(node);
        }
    }

    // Removes a writer node from the queue and re-links the surrounding nodes.
    void unlink_writer_unlocked(WriterNode* node) noexcept
    {
        // Track if the node being removed was at the front of the line.
        bool was_head = (m_headWriter.load() == node);

        // Re-link the previous node to bypass the removed node.
        if (node->prev != nullptr)
        {
            node->prev->next = node->next;
        }
        else
        {
            // Update the head pointer if the removed node was first.
            m_headWriter.store(node->next);
        }

        // Re-link the next node to bypass the removed node.
        if (node->next != nullptr)
        {
            node->next->prev = node->prev;
        }
        else
        {
            // Update the tail pointer if the removed node was last.
            m_tailWriter = node->prev;
        }

        // Isolate the removed node.
        node->prev = nullptr;
        node->next = nullptr;

        // If the head was removed, evaluate the starvation policies for the new head.
        if (was_head)
        {
            WriterNode* newHead = m_headWriter.load();
            bool keep_starving = false;

            uint64_t s = m_state.load(std::memory_order_relaxed);
            uint32_t consec = static_cast<uint32_t>(s >> 32);

            // Preserve starvation if we are in override mode and limits haven't been breached.
            if (m_writerOverride.load(std::memory_order_relaxed) && !writer_limits_reached_unlocked(consec))
            {
                keep_starving = true;
            }
            // Preserve starvation if the new head has also been waiting past the threshold.
            else if (newHead != nullptr && clock::now() >= newHead->ts + Policy::starvationThreshold)
            {
                keep_starving = true;
            }

            // Drop the starving bit if the conditions for writer starvation are no longer met.
            if (!keep_starving)
            {
                m_state.fetch_and(~WRITER_STARVING, 
                                  std::memory_order_relaxed);
            }
        }
    }

    // Determines if any fairness policies (batch limit, timeslice, consecutive limits) have been breached.
    bool writer_limits_reached_unlocked(uint32_t current_consec) const noexcept
    {
        // 1. Check if the active override batch has hit the maximum allowed consecutive writers.
        if (Policy::writerBatchLimit > 0 && 
            m_batchWriters.load(std::memory_order_relaxed) >= Policy::writerBatchLimit)
        {
            return true;
        }

        // 2. Check if the global override timeslice window has expired.
        if (m_writerOverride.load(std::memory_order_relaxed) && 
            Policy::overrideTimeslice > duration::zero())
        {
            auto end_rep = m_overrideEndRep.load(std::memory_order_relaxed);
            
            if (clock::now().time_since_epoch().count() >= end_rep)
            {
                return true;
            }
        }

        // 3. Check if the standard (non-override) consecutive writer streak has been capped.
        if (Policy::maxConsecWriters > 0 && 
            current_consec >= static_cast<uint32_t>(Policy::maxConsecWriters))
        {
            return true;
        }

        return false;
    }

    // Determines if it is safe for a reader in the slow-path to acquire the lock.
    bool can_reader_acquire_unlocked(uint64_t s) const noexcept
    {
        // Readers can never acquire if a writer currently owns the exclusive lock.
        if (s & WRITE_LOCKED)
        {
            return false;
        }
        
        uint32_t consec = static_cast<uint32_t>(s >> 32);

        // If a writer is actively starving, readers must wait in the queue unless the 
        // writer has exceeded its allocated fairness limits (forcing a baton handoff).
        if (s & WRITER_STARVING) 
        {
            if (!writer_limits_reached_unlocked(consec))
            {
                return false;
            }
        }

        // Readers can successfully proceed if we are NOT in an override, 
        // OR if the override is active but has capped out on its limits.
        return !m_writerOverride.load(std::memory_order_relaxed) || writer_limits_reached_unlocked(consec);
    }

    // Determines if it is safe for a writer in the slow-path to acquire the lock.
    bool can_writer_acquire_unlocked(uint64_t s, bool ignore_limits) const noexcept
    {
        // Another writer already holds the exclusive lock.
        if ((s & WRITE_LOCKED) != 0)
        {
            return false;
        }
        
        bool has_readers = false;
        
        // Safely check for active readers, accommodating distributed NUMA stripes if enabled.
        if constexpr (EnableNUMA) 
        {
            if (NumaTopology<EnableNUMA>::is_multi_node) 
            {
                has_readers = (get_total_readers() > 0);
            } 
            else 
            {
                has_readers = ((s & READ_MASK) != 0);
            }
        } 
        else 
        {
            has_readers = ((s & READ_MASK) != 0);
        }
        
        // Writers cannot acquire if ANY readers (fast-path or slow-path) are currently active.
        if (has_readers)
        {
            return false;
        }

        uint32_t consec = static_cast<uint32_t>(s >> 32);

        // If the writer has hit its fairness limits, it must yield the lock to waiting readers.
        if (writer_limits_reached_unlocked(consec))
        {
            // Edge case: If limits are reached but absolutely no readers are waiting, 
            // the writer is explicitly permitted to bypass the limit to prevent unnecessary 
            // stalling and potential deadlocks.
            if (ignore_limits && m_readersWaiting.load(std::memory_order_relaxed) == 0)
            {
                return true;
            }
            
            return false;
        }

        return true;
    }

    // Updates internal batch metrics when a writer successfully acquires the lock.
    void update_writer_batch_unlocked() noexcept
    {
        if (m_writerOverride.load(std::memory_order_relaxed))
        {
            m_batchWriters.fetch_add(1, 
                                     std::memory_order_relaxed);
        }
    }

    // Checks if the lead writer has been waiting past the maximum allowed time, triggering Override Mode.
    void check_and_trigger_override_unlocked() noexcept
    {
        if (!m_writerOverride.load(std::memory_order_relaxed) && m_headWriter.load() != nullptr)
        {
            // std::max wrapped in parenthesis to bypass colliding with <windows.h> max macro
            auto wait_start = (std::max)(m_headWriter.load()->ts, m_lastOverrideEnd);

            if (clock::now() - wait_start >= Policy::maxWriterWait)
            {
                auto end_val = (clock::now() + Policy::overrideTimeslice).time_since_epoch().count();

                m_overrideEndRep.store(end_val, 
                                       std::memory_order_relaxed);
                                       
                m_writerOverride.store(true, 
                                       std::memory_order_relaxed);
                                       
                m_batchWriters.store(0, 
                                     std::memory_order_relaxed);

                m_state.fetch_or(WRITER_STARVING, 
                                 std::memory_order_relaxed);

                LOG_DEBUG(this, "Override enabled");

                m_headWriter.load()->sem.release();
            }
        }
    }

    void notify_next_writer_unlocked() noexcept
    {
        if (m_headWriter.load() != nullptr)
        {
            m_headWriter.load()->sem.release();
        }
    }

    void notify_all_readers_unlocked() noexcept
    {
        clear_override_unlocked();
        
        int readersToWake = m_readersWaiting.load(std::memory_order_relaxed);
        
        if (readersToWake > 0)
        {
            m_semReaders.release(readersToWake);
        }
    }

    void clear_override_unlocked() noexcept
    {
        if (m_writerOverride.load(std::memory_order_relaxed))
        {
            m_lastOverrideEnd = clock::now();

            LOG_DEBUG(this, "Override cleared");
        }

        m_writerOverride.store(false, 
                               std::memory_order_relaxed);
                               
        m_batchWriters.store(0, 
                             std::memory_order_relaxed);

        m_state.fetch_and(~WRITER_STARVING, 
                          std::memory_order_relaxed);

        update_has_waiters_unlocked();
    }

    // -------------------------------------------------------------------------
    // Class Variables (2-Cache-Line Segregation)
    // 
    // In-Object Footprint (x86_64 / standard ARM64): 
    // - Non-NUMA (EnableNUMA = false): 128 Bytes (exactly 2 cache lines)
    // - NUMA-Aware (EnableNUMA = true):  128 Bytes (2 cache lines) + heap array
    // 
    // In-Object Footprint (Apple Silicon):
    // - Non-NUMA / NUMA: 256 Bytes (exactly two 128-byte cache lines)
    // 
    // Line 1: Fast-Path Domain. Uncontended readers/writers only touch this line.
    // Line 2: Slow-Path Domain. Isolated to prevent TTAS spinlock false-sharing 
    //         from invalidating the fast-path state across CPU cores.
    // -------------------------------------------------------------------------

    // --- CACHE LINE 1 (Offset 0x00) ---
    // Forces the object itself to start on a cache line to prevent external false sharing.
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> m_state{ 0 };
    
    // Non-NUMA: 0 bytes (empty struct optimization).
    // NUMA: 8 bytes (std::unique_ptr to distributed stripe array).    
    [[no_unique_address]] NumaStorage<EnableNUMA>  m_numaState;

    // --- CACHE LINE 2 (Offset 0x40 on x86_64, 0x80 on Apple Silicon) ---
    // The alignas directive pads out the remainder of Line 1 and pushes the slow-path 
    // queue state exactly to the start of the next hardware cache line.
    //
    // MEMORY FOOTPRINT VS. PERFORMANCE TRADE-OFF:
    // If object footprint size is a crucial factor (e.g., allocating millions of locks 
    // for highly fine-grained data structures), the `alignas(CACHE_LINE_SIZE)` on 
    // `m_headWriter` below can be removed.
    //
    // Footprint Reduction: Removing it packs the entire lock into a single cache line 
    // (64 bytes on x86_64, 128 bytes on Apple Silicon).
    // 
    // Estimated Performance Cost: 
    // - Uncontended workloads: 0% degradation (remains a single L1 cache hit).
    // - Contended workloads: about 50% degradation in fast-path throughput. Threads 
    //   spinning on `m_queueLock` will trigger false-sharing invalidations on 
    //   `m_state`, causing cache-line bouncing across the CPU interconnect and 
    //   introducing tail latency spikes.
    
    // 8-Byte Aligned Members
    alignas(CACHE_LINE_SIZE) std::atomic<WriterNode*> m_headWriter{ nullptr }; 
    WriterNode*                                       m_tailWriter = nullptr;
    std::atomic<clock::duration::rep>                 m_overrideEndRep{ 0 };   
    time_point                                        m_lastOverrideEnd{};   
    std::counting_semaphore<PTRDIFF_MAX>              m_semReaders{ 0 };

    // 4-Byte Aligned Members
    std::atomic<int>                                  m_readersWaiting{ 0 }; 
    std::atomic<int>                                  m_batchWriters{ 0 };   

    // 1-Byte Aligned Members
    mutable std::atomic_flag                          m_queueLock = ATOMIC_FLAG_INIT;
    std::atomic<bool>                                 m_writerOverride{ false }; 

#ifndef NDEBUG
    // Diagnostic state (Usually pushes size to a 3rd cache line in Debug builds)
    LoggerFn                                          m_logger;
    std::atomic<bool>                                 m_loggerAttached{ false };
#endif
};