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
#include <cassert>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <semaphore>
#include <string_view>
#include <thread>

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
// Ideal Use Cases:
// - High-frequency trading order books.
// - ETW/EDR or telemetry pipelines where millions of read events occur, but 
//   configuration updates (writes) must not be starved.
// - Centralized routing tables or subscriber lists in highly concurrent backends.
// -------------------------------------------------------------------------
template <typename Policy = DefaultFairRWLockPolicy>
class FairRWLock
{
public:
    using clock      = std::chrono::steady_clock;
    using duration   = clock::duration;
    using time_point = clock::time_point;
    using LoggerFn   = std::function<void(std::string_view)>;

    FairRWLock() = default;

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
        explicit ReadGuard(FairRWLock& l, duration timeout = duration::max()) : m_lock(&l)
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
        explicit WriteGuard(FairRWLock& l, duration timeout = duration::max()) : m_lock(&l)
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
    bool ReadLock(duration timeout = duration::max())
    {
        // Lock-Free Acquisition: Bypasses CAS loops for O(N) throughput.
        // Hardware unconditionally resolves parallel fetches immediately.
        uint64_t s = m_state.fetch_add(READ_INC,
                                       std::memory_order_acquire);

        // Guard against reader count overflow (28 bits maximum).
        if ((s & READ_MASK) == READ_MASK) [[unlikely]]
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
        // Speculatively increment the read count to attempt lock-free acquisition.
        uint64_t s = m_state.fetch_add(READ_INC, 
                                       std::memory_order_acquire);
        
        // Guard against reader count overflow.
        if ((s & READ_MASK) == READ_MASK) [[unlikely]]
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
        s = m_state.load(std::memory_order_relaxed);
        
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
    bool WriteLock(duration timeout = duration::max())
    {
        uint64_t s = m_state.load(std::memory_order_acquire);

        while (true)
        {
            uint32_t consec = static_cast<uint32_t>(s >> 32);

            if (Policy::maxConsecWriters == 0 || consec < static_cast<uint32_t>(Policy::maxConsecWriters)) [[likely]]
            {
                // Fast Path: Only acquire if the lower 32-bits are absolute zero 
                // (no readers, no writers, no waiters, no starving bits).
                if ((s & STATE_MASK) == 0) [[likely]]
                {
                    uint32_t next_consec = (consec < 0xFFFFFFFF) ? consec + 1 : consec;
                    uint64_t next_s = WRITE_LOCKED | (static_cast<uint64_t>(next_consec) << 32);
                    
                    // Optimized to compare_exchange_weak to avoid nested hardware spin-loops on ARM/PowerPC
                    if (m_state.compare_exchange_weak(s, 
                                                      next_s, 
                                                      std::memory_order_acquire, 
                                                      std::memory_order_relaxed)) [[likely]]
                    {
                        return true;
                    }

                    // Loop re-evaluates automatically with the newly updated `s` from the failed CAS
                    continue;
                }
            }

            break;
        }

        return slow_write_lock(timeout);
    }

    // -------------------------------------------------------------------------
    // Attempts to acquire exclusive write access without blocking.
    // Returns true immediately if successful, false if any readers or writers are active.
    // -------------------------------------------------------------------------
    bool TryWriteLock() noexcept
    {
        // Load the current state to evaluate limits and fast-path acquisition.
        uint64_t s = m_state.load(std::memory_order_acquire);
        uint32_t consec = static_cast<uint32_t>(s >> 32);

        // Ensure we are not breaching the consecutive writer limit.
        if (Policy::maxConsecWriters == 0 || consec < static_cast<uint32_t>(Policy::maxConsecWriters)) [[likely]]
        {
            // Fast Path: Only acquire if the lower 32-bits are absolute zero.
            if ((s & STATE_MASK) == 0) [[likely]]
            {
                uint32_t next_consec = (consec < 0xFFFFFFFF) ? consec + 1 : consec;
                uint64_t next_s = WRITE_LOCKED | (static_cast<uint64_t>(next_consec) << 32);
                
                // Strong CAS is correct here because TryWriteLock does not loop on the fast path.
                if (m_state.compare_exchange_strong(s, 
                                                    next_s, 
                                                    std::memory_order_acquire, 
                                                    std::memory_order_relaxed)) [[likely]]
                {
                    return true;
                }
            }
        }

        // Fast path failed; attempt to acquire the slow-path queue lock without blocking.
        QueueLockGuard lk(m_queueLock, 
                          std::try_to_lock);
        
        // Return immediately if the queue lock is contended.
        if (!lk.owns_lock())
        {
            return false;
        }

        // Evaluate if Override Mode needs to be activated.
        check_and_trigger_override_unlocked();
        s = m_state.load(std::memory_order_relaxed);
        
        // Loop to attempt acquisition if the lock state permits writers.
        while (can_writer_acquire_unlocked(s, false))
        {
            consec = static_cast<uint32_t>(s >> 32);
            uint32_t next_consec = (consec < 0xFFFFFFFF) ? consec + 1 : consec;
            uint64_t next_s = (s | WRITE_LOCKED) & STATE_MASK;
            next_s |= (static_cast<uint64_t>(next_consec) << 32);

            // Use weak CAS since we are looping.
            if (m_state.compare_exchange_weak(s, 
                                              next_s, 
                                              std::memory_order_acquire, 
                                              std::memory_order_relaxed))
            {
                // Track batch limits since we successfully acquired as a writer.
                update_writer_batch_unlocked();

                return true;
            }
        }
        
        // The lock cannot be acquired immediately.
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
    static constexpr uint64_t READ_MASK       = 0x000000000FFFFFFFULL; 
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
        std::binary_semaphore sem{0};
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
            while (m_flag->test_and_set(std::memory_order_acquire))
            {
                while (m_flag->test(std::memory_order_relaxed))
                {
                    cpu_relax_pause();
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

    // Handles locking when the fast-path fails for readers.
    // Registers the reader in the wait counter and waits on the reader condition variable.
    bool slow_read_lock(duration timeout)
    {
        LOG_DEBUG(this, "Reader entered slow path");

        // Register this reader in the slow-path wait queue.
        m_readersWaiting.fetch_add(1,
                                   std::memory_order_acquire);
                                   
        // Ensure the fast-path knows there are threads waiting.
        m_state.fetch_or(HAS_WAITERS,
                         std::memory_order_relaxed);

        // Acquire the queue spinlock.
        QueueLockGuard lk(m_queueLock);
        
        // Evaluate if Override Mode needs to be activated before we wait.
        check_and_trigger_override_unlocked();

        time_point deadline = (timeout == duration::max()) ? time_point::max() : clock::now() + timeout;
        bool acquired = false;

        while (true)
        {
            uint64_t s = m_state.load(std::memory_order_relaxed);
            
            // Attempt to acquire the lock directly if the state permits.
            if (can_reader_acquire_unlocked(s))
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

                // Retry CAS on weak failure.
                continue;
            }

            // Re-evaluate override policies before going to sleep.
            check_and_trigger_override_unlocked();
            
            // Release the spinlock to allow other threads to operate while we sleep.
            lk.unlock();
            
            bool acquired_sem = false;
            
            // Sleep on the semaphore until woken or the timeout expires.
            if (deadline == time_point::max())
            {
                m_semReaders.acquire();
                acquired_sem = true;
            }
            else
            {
                acquired_sem = m_semReaders.try_acquire_until(deadline);
            }
            
            // Reacquire the spinlock after waking up.
            lk.lock();
            
            // If we timed out, attempt one last optimistic acquisition loop.
            if (!acquired_sem)
            {
                s = m_state.load(std::memory_order_relaxed);
                
                // Mitigates spurious CAS failures directly following a timeout wake event. 
                // We utilize a weak exchange loop here to verify whether the lock is 
                // logically acquirable despite the timeout expiration.
                while (can_reader_acquire_unlocked(s))
                {
                    uint64_t next_s = (s + READ_INC) & STATE_MASK;

                    if (m_state.compare_exchange_weak(s,
                                                      next_s,
                                                      std::memory_order_acquire,
                                                      std::memory_order_relaxed))
                    {
                        // Reset the writer batch count on successful acquire.
                        m_batchWriters.store(0,
                                             std::memory_order_relaxed);
                        acquired = true;
                        
                        break;
                    }
                }

                // Exit the main loop if we didn't acquire it after timeout.
                break;
            }
        }

        // Unregister this reader from the slow-path wait queue.
        m_readersWaiting.fetch_sub(1,
                                   std::memory_order_release);
                                   
        // Synchronize the fast-path HAS_WAITERS bit.
        update_has_waiters_unlocked();

        // If we gave up, but there are no more waiting readers, we need to pass the baton back to writers.
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
        
        uint64_t s = m_state.load(std::memory_order_relaxed);
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

        m_state.fetch_or(HAS_WAITERS, 
                         std::memory_order_relaxed);
        
        QueueLockGuard lk(m_queueLock);
        
        check_and_trigger_override_unlocked();

        WriterNode myNode;
        myNode.ts = clock::now();
        link_writer_unlocked(&myNode);
        update_has_waiters_unlocked();

        ScopeUnlinker unlinker(this, &myNode);
        time_point deadline = (timeout == duration::max()) ? time_point::max() : clock::now() + timeout;

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

                    check_and_trigger_override_unlocked();
                    yield_count++;

                    continue;
                }

                bool bypass_limits = (yield_count >= Policy::maxYieldsBeforeBypass);

                if (can_writer_acquire_unlocked(s, bypass_limits))
                {
                    uint32_t next_consec = (consec < 0xFFFFFFFF) ? consec + 1 : consec;
                    uint64_t next_s = (s | WRITE_LOCKED) & STATE_MASK;
                    next_s |= (static_cast<uint64_t>(next_consec) << 32);

                    if (m_state.compare_exchange_weak(s, 
                                                      next_s, 
                                                      std::memory_order_acquire, 
                                                      std::memory_order_relaxed))
                    {
                        acquired = true;
                        
                        break;
                    }

                    continue;
                }
            }

            time_point wait_until_time = deadline;
            
            if (is_head && !asserted_starvation && wait_until_time > starve_time)
            {
                wait_until_time = starve_time;
            }

            lk.unlock();
            
            bool acquired_sem = false;
            
            if (wait_until_time == time_point::max())
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
            
            if (is_head && !asserted_starvation && clock::now() >= starve_time)
            {
                // Trip the starvation bit to block new fast-path readers.
                m_state.fetch_or(WRITER_STARVING, 
                                 std::memory_order_relaxed);
                                 
                asserted_starvation = true;

                LOG_DEBUG(this, "Writer starvation asserted");
            }

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

                        if (m_state.compare_exchange_strong(s, 
                                                            next_s, 
                                                            std::memory_order_acquire, 
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
        
        // If handing off to readers or limits were breached, drop the starving bit.
        if (handoff_to_readers || limits_reached)
        {
            clear_mask |= WRITER_STARVING;
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
        if (Policy::writerBatchLimit > 0 && 
            m_batchWriters.load(std::memory_order_relaxed) >= Policy::writerBatchLimit)
        {
            return true;
        }

        if (m_writerOverride.load(std::memory_order_relaxed) && 
            Policy::overrideTimeslice > duration::zero())
        {
            auto end_rep = m_overrideEndRep.load(std::memory_order_relaxed);
            
            if (clock::now().time_since_epoch().count() >= end_rep)
            {
                return true;
            }
        }

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
        if (s & WRITE_LOCKED)
        {
            return false;
        }
        
        uint32_t consec = static_cast<uint32_t>(s >> 32);

        if (s & WRITER_STARVING) 
        {
            if (!writer_limits_reached_unlocked(consec))
            {
                return false;
            }
        }

        return !m_writerOverride.load(std::memory_order_relaxed) || writer_limits_reached_unlocked(consec);
    }

    // Determines if it is safe for a writer in the slow-path to acquire the lock.
    bool can_writer_acquire_unlocked(uint64_t s, bool ignore_limits) const noexcept
    {
        if (s & (WRITE_LOCKED | READ_MASK))
        {
            return false;
        }

        uint32_t consec = static_cast<uint32_t>(s >> 32);

        if (writer_limits_reached_unlocked(consec))
        {
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
            auto wait_start = std::max(m_headWriter.load()->ts, m_lastOverrideEnd);

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
    // Class Variables (Optimized memory packing layout)
    // -------------------------------------------------------------------------

    // Fast Path (8 bytes)
    // Upper 32 bits: Consecutive writers streak.
    // Lower 32 bits: Active reader count and state flags (e.g., WRITE_LOCKED)  
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> m_state{0};
        
    // Slow Path State Pack
    // Explicitly grouped to minimize internal struct padding.
    
    // 4-byte types (Grouped side-by-side to equal exactly 8 bytes)
    // The number of readers currently registered in the slow-path wait queue
    alignas(CACHE_LINE_SIZE) std::atomic<int> m_readersWaiting{0}; 
    // The number of consecutive writers that have acquired the lock during the current override phase.
    std::atomic<int>                          m_batchWriters{0};   
    
    // 8-byte types
    // Pointer to the front of the intrusive linked list of waiting writers
    std::atomic<WriterNode*>                  m_headWriter{nullptr}; 
    // Pointer to the back of the intrusive linked list of waiting writers.
    // Safe to be non-atomic as it is only accessed/modified under m_queueLock.
    WriterNode*                               m_tailWriter = nullptr;
    // The raw integer representation (ticks) of the deadline for the current Override Mode.
    std::atomic<clock::duration::rep>         m_overrideEndRep{0};   
    // The timestamp of when the previous Override Mode ended.
    time_point                                m_lastOverrideEnd{};   
    // Counting semaphore for waking pooled readers safely without dynamic allocation.
    std::counting_semaphore<PTRDIFF_MAX>      m_semReaders{0};
    
    // 1-byte types
    // Flag indicating the lock is in Emergency Override Mode.
    std::atomic<bool>                         m_writerOverride{false}; 
    // Atomic Spinlock to protect queue manipulation.
    mutable std::atomic_flag                  m_queueLock = ATOMIC_FLAG_INIT;

#ifndef NDEBUG
    // Optional callback function for emitting diagnostic lock transition logs
    LoggerFn                                  m_logger;
    // Flag indicating if the diagnostic logger has been attached.
    std::atomic<bool>                         m_loggerAttached{ false };
#endif
};
