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

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string_view>

#ifndef NDEBUG
#define LOG_DEBUG(lock_ptr, msg) (lock_ptr)->debug_log(msg)
#else
#define LOG_DEBUG(lock_ptr, msg) ((void)0)
#endif

// -------------------------------------------------------------------------
// MomentumRWLock
// 
// Architecture & Approach:
// This lock uses a standard std::mutex and std::condition_variable approach 
// to coordinate readers and writers. It manages lock handover 
// using a state flag (m_writerWantsPriority) to pass control between 
// reading and writing phases.
//
// Fairness Policy:
// - Momentum Handoff: When the last active reader unlocks and writers are 
//   waiting, the lock explicitly grants priority to the writers. 
//   Similarly, an unlocking writer will pass this priority token to the next 
//   waiting writer if the writer queue is not empty.
// - Reader Yielding: Incoming readers are blocked if a writer is currently 
//   active OR if a waiting writer has been granted priority.
// - Starvation Prevention: Writer starvation is mitigated because the 
//   priority flag stops new readers from stampeding and acquiring the lock 
//   once the current read phase concludes. Readers are only awakened 
//   en masse once the writer queue is fully drained and the priority flag 
//   is cleared.
// -------------------------------------------------------------------------

constexpr uint32_t INFINITE_TIME = static_cast<uint32_t>(-1);

class MomentumRWLock
{
public:
    using clock    = std::chrono::steady_clock;
    using LoggerFn = std::function<void(std::string_view)>;

private:
    // -------------------------------------------------------------------------
    // Class Variables (Synchronized State)
    // -------------------------------------------------------------------------
    std::mutex              m_mutex;                        // Mutex for internal state synchronization
    std::condition_variable m_readerCond;                   // Condition variable for blocked readers
    std::condition_variable m_writerCond;                   // Condition variable for blocked writers

    int                     m_activeReaders{ 0 };           // Number of readers currently holding the lock
    int                     m_waitingWriters{ 0 };          // Number of writers waiting for the lock
    bool                    m_activeWriter{ false };        // Flag indicating if a writer is currently holding the lock
    bool                    m_writerWantsPriority{ false }; // Flag to grant priority to waiting writers

#ifndef NDEBUG
    LoggerFn                m_logger;                       // Optional callback function for emitting diagnostic lock transition logs
    std::atomic<bool>       m_loggerAttached{ false };      // Flag indicating if the diagnostic logger has been attached
#endif

public:
    MomentumRWLock() noexcept = default;
    ~MomentumRWLock() noexcept = default;
    MomentumRWLock(const MomentumRWLock&) = delete;
    MomentumRWLock(MomentumRWLock&&) = delete;
    MomentumRWLock& operator=(const MomentumRWLock&) = delete;
    MomentumRWLock& operator=(MomentumRWLock&&) = delete;

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
            throw std::logic_error("MomentumRWLock: Logger already attached.");
        }

        // Serialize assignment to prevent concurrent re-attachments.
        std::unique_lock<std::mutex> lock(m_mutex);
        
        // Double-check under the lock.
        if (m_loggerAttached.load(std::memory_order_relaxed))
        {
            throw std::logic_error("MomentumRWLock: Logger already attached.");
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
        explicit ReadGuard(MomentumRWLock& l, uint32_t milliseconds = INFINITE_TIME) : m_lock(&l)
        {
            m_locked = m_lock->ReadLock(milliseconds);
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
        MomentumRWLock* m_lock;
        bool m_locked = false;
    };

    // A scoped RAII guard for safely managing Write locks.
    class WriteGuard
    {
    public:
        // Attempts to acquire an exclusive write lock, optionally bound by a timeout.
        explicit WriteGuard(MomentumRWLock& l, uint32_t milliseconds = INFINITE_TIME) : m_lock(&l)
        {
            m_locked = m_lock->WriteLock(milliseconds);
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
        MomentumRWLock* m_lock;
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
    bool ReadLock(uint32_t Milliseconds)
    {
        // Acquire the mutex to inspect and safely modify the lock state.
        std::unique_lock<std::mutex> lock(m_mutex);

        // Wait predicate: Block readers if a writer is active OR
        // if a writer is waiting AND has been granted priority.
        auto predicate = [this]()
            {
                return !m_activeWriter && !m_writerWantsPriority;
            };

        bool acquired = false;

        // Sleep on the condition variable until the predicate is satisfied or timeout expires.
        if (Milliseconds == INFINITE_TIME)
        {
            m_readerCond.wait(lock, predicate);
            acquired = true; // wait ensures predicate is true upon waking
        }
        else
        {
            acquired = m_readerCond.wait_for(lock, std::chrono::milliseconds(Milliseconds), predicate);
        }

        // Evaluate if the lock acquisition failed due to a timeout.
        if (!acquired) [[unlikely]]
        {
            return false; // Timeout
        }

        // Lock acquired successfully
        m_activeReaders++;
        
        LOG_DEBUG(this, "ReadLock acquired");

        return true;
    }

    // -------------------------------------------------------------------------
    // Releases shared read access.
    // If this is the last reader and a writer is waiting, this function acts as 
    // the trigger to wake up the slow-path writer queue.
    // Returns true on success, false if there is a logic error (unbalanced lock).
    // -------------------------------------------------------------------------
    bool ReadUnlock()
    {
        // Acquire the mutex for safe queue manipulation.
        std::unique_lock<std::mutex> lock(m_mutex);

        // Guard against unbalanced unlock attempts.
        if (m_activeReaders <= 0) [[unlikely]]
        {
            // Trying to unlock when no readers are active - indicates a logic error
            return false;
        }

        m_activeReaders--;
        
        LOG_DEBUG(this, "ReadUnlock released");

        // If this was the last reader AND writers are waiting,
        // grant priority to a writer and notify them.
        if (m_activeReaders == 0 && m_waitingWriters > 0) [[unlikely]]
        {
            m_writerWantsPriority = true; // Grant priority

            // Manually unlock before notifying to prevent Wait Morphing
            lock.unlock();
            
            // Wake one waiting writer
            m_writerCond.notify_one();    
        }

        return true;
    }

    // -------------------------------------------------------------------------
    // Acquires exclusive write access.
    // Blocks the thread until the lock is acquired or the optional timeout expires.
    // Returns true on success, false if the timeout was reached.
    // -------------------------------------------------------------------------
    bool WriteLock(uint32_t Milliseconds)
    {
        // Acquire the mutex to inspect and safely modify the lock state.
        std::unique_lock<std::mutex> lock(m_mutex);

        // Register this writer in the waiting queue.
        m_waitingWriters++;

        // Wait predicate: Block writers if another writer is active OR
        // if any readers are active.
        auto predicate = [this]()
            {
                return !m_activeWriter && m_activeReaders == 0;
            };

        bool acquired = false;

        // Sleep on the condition variable until the predicate is satisfied or timeout expires.
        if (Milliseconds == INFINITE_TIME)
        {
            m_writerCond.wait(lock, predicate);
            acquired = true;
        }
        else
        {
            acquired = m_writerCond.wait_for(lock, std::chrono::milliseconds(Milliseconds), predicate);
        }

        // Decrement the wait count after the waiting attempt concludes.
        m_waitingWriters--; 

        // Evaluate if the lock acquisition failed due to a timeout.
        if (!acquired) [[unlikely]]
        {
            // If priority was granted to a writer, we might have "consumed" a notification.
            if (m_writerWantsPriority)
            {
                // If other writers are still waiting, we must pass the notification
                // to one of them to prevent it from being lost.
                if (m_waitingWriters > 0)
                {
                    // Drop lock before passing the baton to the next writer
                    lock.unlock();
                    
                    m_writerCond.notify_one();
                }
                else
                {
                    // We were the last waiting writer. Clear the priority flag
                    // and wake up readers who might be blocked by it.
                    m_writerWantsPriority = false;

                    // Drop lock before unleashing the reader stampede
                    lock.unlock();
                    
                    m_readerCond.notify_all();
                }
            }

            return false; // Timeout
        }

        // Lock acquired successfully
        m_writerWantsPriority = false; // Priority fulfilled, so reset it
        m_activeWriter = true;
        
        LOG_DEBUG(this, "WriteLock acquired");

        return true;
    }

    // -------------------------------------------------------------------------
    // Releases exclusive write access.
    // Hands off priority depending on wait queues.
    // Returns true on success, false if there is a logic error (unbalanced lock).
    // -------------------------------------------------------------------------
    bool WriteUnlock()
    {
        // Acquire the mutex for safe queue manipulation.
        std::unique_lock<std::mutex> lock(m_mutex);

        // Guard against unbalanced unlock attempts.
        if (!m_activeWriter) [[unlikely]]
        {
            // Trying to unlock when no writer is active - indicates a logic error
            return false;
        }

        m_activeWriter = false;
        
        LOG_DEBUG(this, "WriteUnlock released");

        // Decide who to wake up:
        // If writers are waiting, grant priority and wake one writer.
        // Otherwise, wake all waiting readers.
        if (m_waitingWriters > 0)
        {
            m_writerWantsPriority = true; // Grant priority to the next writer

            // Drop lock before waking the writer
            lock.unlock();
            
            m_writerCond.notify_one();
        }
        else
        {
            // No writers waiting, ensure priority flag is clear and wake readers
            m_writerWantsPriority = false;

            // Drop lock before waking the readers
            lock.unlock();
            
            m_readerCond.notify_all();
        }

        return true;
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
                oss << "[" << ms << " ms] MomentumRWLock: " << s;
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