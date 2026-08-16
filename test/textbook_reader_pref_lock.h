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

#pragma once

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string_view>

#ifndef NDEBUG
#define LOG_DEBUG(lock_ptr, msg) (lock_ptr)->debug_log(msg)
#else
#define LOG_DEBUG(lock_ptr, msg) ((void)0)
#endif

// -------------------------------------------------------------------------
// TextbookReaderPrefLock
// 
// Architecture & Approach:
// This is a strict, standard textbook reader-preference lock. It uses a 
// standard std::mutex and std::condition_variable approach without atomic 
// fast-paths. 
//
// Fairness Policy:
// - Absolute Reader Preference: Incoming readers bypass waiting writers as 
//   long as no writer is currently active.
// - Writer Starvation: Highly likely under heavy, continuous read loads.
// -------------------------------------------------------------------------
class TextbookReaderPrefLock
{
public:
    using clock    = std::chrono::steady_clock;
    using duration = clock::duration;
    using LoggerFn = std::function<void(std::string_view)>;

private:
    // -------------------------------------------------------------------------
    // Class Variables (Synchronized State)
    // -------------------------------------------------------------------------
    int                     m_activeReaders{ 0 };      // Number of readers currently holding the lock
    int                     m_waitingReaders{ 0 };     // Number of readers waiting in the queue
    int                     m_waitingWriters{ 0 };     // Number of writers waiting in the queue
    bool                    m_activeWriter{ false };   // Flag indicating if a writer is currently holding the lock

    mutable std::mutex      m_mtx;                     // Mutex for internal state synchronization
    std::condition_variable m_cvReaders;               // Condition variable for blocked readers
    std::condition_variable m_cvWriters;               // Condition variable for blocked writers

#ifndef NDEBUG
    LoggerFn                m_logger;                  // Optional callback function for emitting diagnostic lock transition logs
    std::atomic<bool>       m_loggerAttached{ false }; // Flag indicating if the diagnostic logger has been attached
#endif

public:
    TextbookReaderPrefLock() noexcept = default;
    ~TextbookReaderPrefLock() noexcept = default;
    
    TextbookReaderPrefLock(const TextbookReaderPrefLock&) = delete;
    TextbookReaderPrefLock(TextbookReaderPrefLock&&) = delete;
    TextbookReaderPrefLock& operator=(const TextbookReaderPrefLock&) = delete;
    TextbookReaderPrefLock& operator=(TextbookReaderPrefLock&&) = delete;

#ifndef NDEBUG
    // -------------------------------------------------------------------------
    // Assigns an optional diagnostic logger for debugging lock state transitions.
    // Forbids re-attachment to prevent data races and unsafe functor replacement 
    // during active lock contention.
    // -------------------------------------------------------------------------
    void SetLogger(LoggerFn lg)
    {
        // First check to avoid locking overhead if already attached.
        if (m_loggerAttached.load(std::memory_order_acquire))
        {
            throw std::logic_error("TextbookReaderPrefLock: Logger already attached.");
        }

        // Serialize assignment to prevent concurrent re-attachments.
        std::unique_lock lk(m_mtx);
        
        // Double-check under the lock.
        if (m_loggerAttached.load(std::memory_order_relaxed))
        {
            throw std::logic_error("TextbookReaderPrefLock: Logger already attached.");
        }

        m_logger = std::move(lg);
        
        // Publish the logger availability to all threads.
        m_loggerAttached.store(true,
                               std::memory_order_release);
    }
#else
    // -------------------------------------------------------------------------
    // Zero-cost abstraction for Release builds
    // -------------------------------------------------------------------------
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
        explicit ReadGuard(TextbookReaderPrefLock& l, duration timeout = (duration::max)()) : m_lock(&l)
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
        TextbookReaderPrefLock* m_lock;
        bool m_locked = false;
    };

    // A scoped RAII guard for safely managing Write locks.
    class WriteGuard
    {
    public:
        // Attempts to acquire an exclusive write lock, optionally bound by a timeout.
        explicit WriteGuard(TextbookReaderPrefLock& l, duration timeout = (duration::max)()) : m_lock(&l)
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
        TextbookReaderPrefLock* m_lock;
        bool m_locked = false;
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
        // Acquire the mutex to inspect and safely modify the lock state.
        std::unique_lock lk(m_mtx);
        
        // Register this reader in the waiting queue.
        m_waitingReaders++;

        // Strict Reader Predicate: Block ONLY if a writer is actively writing.
        // Waiting writers are completely ignored.
        auto predicate = [this]() 
            { 
                return !m_activeWriter; 
            };

        bool acquired = false;
        
        // Sleep on the condition variable until the predicate is satisfied or timeout expires.
        if (timeout == (duration::max)())
        {
            m_cvReaders.wait(lk, predicate);
            acquired = true;
        }
        else
        {
            acquired = m_cvReaders.wait_until(lk, clock::now() + timeout, predicate);
        }

        // Decrement the wait count after the waiting attempt concludes.
        m_waitingReaders--;

        // Evaluate if the lock acquisition failed due to a timeout.
        if (!acquired) [[unlikely]]
        {
            return false;
        }

        // Lock acquired successfully
        LOG_DEBUG(this, "ReadLock acquired");
        
        m_activeReaders++;
        
        return true;
    }

    // -------------------------------------------------------------------------
    // Attempts to acquire shared read access without blocking.
    // Returns true immediately if successful, false if a writer is currently active.
    // -------------------------------------------------------------------------
    bool TryReadLock() noexcept
    {
        // Try to acquire the unique lock without blocking.
        std::unique_lock lk(m_mtx, std::try_to_lock);
        
        // Check if lock acquisition failed or if a writer is currently holding the lock.
        if (!lk.owns_lock() || m_activeWriter)
        {
            return false;
        }

        // Lock acquired successfully
        LOG_DEBUG(this, "TryReadLock acquired");
        
        m_activeReaders++;
        
        return true;
    }

    // -------------------------------------------------------------------------
    // Releases shared read access.
    // If this is the last active reader, and writers are waiting, this wakes ONE writer.
    // -------------------------------------------------------------------------
    void ReadUnlock() noexcept
    {
        // Acquire the mutex for safe state manipulation.
        std::unique_lock lk(m_mtx);
        
        m_activeReaders--;
        
        LOG_DEBUG(this, "ReadUnlock released");

        // If this was the last active reader, and writers are waiting, wake ONE writer.
        if (m_activeReaders == 0 && m_waitingWriters > 0) [[unlikely]]
        {
            // Drop lock before notifying to prevent pessimistic wait morphing
            lk.unlock(); 
            
            m_cvWriters.notify_one();
        }
    }

    // -------------------------------------------------------------------------
    // Acquires exclusive write access.
    // Blocks the thread until the lock is acquired or the optional timeout expires.
    // Returns true on success, false if the timeout was reached.
    // -------------------------------------------------------------------------
    bool WriteLock(duration timeout = (duration::max)())
    {
        // Acquire the mutex to inspect and safely modify the lock state.
        std::unique_lock lk(m_mtx);
        
        // Register this writer in the waiting queue.
        m_waitingWriters++;

        // Writer Predicate: Block if ANY writer is writing OR ANY reader is reading.
        auto predicate = [this]() 
            { 
                return !m_activeWriter && m_activeReaders == 0; 
            };

        bool acquired = false;
        
        // Sleep on the condition variable until the predicate is satisfied or timeout expires.
        if (timeout == (duration::max)())
        {
            m_cvWriters.wait(lk, predicate);
            acquired = true;
        }
        else
        {
            acquired = m_cvWriters.wait_until(lk, clock::now() + timeout, predicate);
        }

        // Decrement the wait count after the waiting attempt concludes.
        m_waitingWriters--;

        // Evaluate if the lock acquisition failed due to a timeout.
        if (!acquired) [[unlikely]]
        {
            return false;
        }

        // Lock acquired successfully
        LOG_DEBUG(this, "WriteLock acquired");
        
        m_activeWriter = true;
        
        return true;
    }

    // -------------------------------------------------------------------------
    // Attempts to acquire exclusive write access without blocking.
    // Returns true immediately if successful, false if any readers or writers are active.
    // -------------------------------------------------------------------------
    bool TryWriteLock() noexcept
    {
        // Try to acquire the unique lock without blocking.
        std::unique_lock lk(m_mtx, std::try_to_lock);
        
        // Check if lock acquisition failed or if there are any active readers/writers.
        if (!lk.owns_lock() || m_activeWriter || m_activeReaders > 0)
        {
            return false;
        }

        // Lock acquired successfully
        LOG_DEBUG(this, "TryWriteLock acquired");
        
        m_activeWriter = true;
        
        return true;
    }

    // -------------------------------------------------------------------------
    // Releases exclusive write access.
    // Uses Strict Reader Preference Unlock Logic: Prioritizes waking up ALL queued readers.
    // -------------------------------------------------------------------------
    void WriteUnlock() noexcept
    {
        // Acquire the mutex for safe state manipulation.
        std::unique_lock lk(m_mtx);
        
        m_activeWriter = false;
        
        LOG_DEBUG(this, "WriteUnlock released");

        // Strict Reader Preference Unlock Logic:
        // Prioritize waking up ALL queued readers.
        // Only if zero readers are waiting do we consider waking a single queued writer.
        if (m_waitingReaders > 0)
        {
            // Drop lock before waking the readers
            lk.unlock(); 
            
            m_cvReaders.notify_all();
        }
        else if (m_waitingWriters > 0)
        {
            // Drop lock before waking the writer
            lk.unlock(); 
            
            m_cvWriters.notify_one();
        }
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
                oss << "[" << ms << " ms] TextbookBaseline: " << s;
                m_logger(oss.str());
            }
            catch (...)
            {
            }
        }
    }
#else
    // -------------------------------------------------------------------------
    // Zero-cost abstraction for Release builds
    // -------------------------------------------------------------------------
    void debug_log(std::string_view) const noexcept
    {
    }
#endif
};