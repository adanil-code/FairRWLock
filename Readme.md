# FairRWLock: Starvation-Resistant Reader-Writer Lock

![Platform: Windows | Linux | macOS](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-blue)
![Language: C++20](https://img.shields.io/badge/Language-C%2B%2B20-orange)
![Environment: User Mode](https://img.shields.io/badge/Environment-User%20Mode-success)

Designed for high-concurrency C++20 environments where throughput is critical, but thread starvation is unacceptable.

* **Hybrid Two-Tier Locking** combining an atomic uncontended fast path with a managed wait queue.
* **Deterministic Fairness Escalation** preventing starvation through explicit policy thresholds.
* **Logical Baton Handoff** explicitly identifying and notifying the next eligible waiter.
* **Configurable Fairness Policy** allowing starvation thresholds, writer batching, and escalation behavior to be tuned for different workloads.
* **NUMA-Aware Scaling** optionally distributing reader counts across physical sockets to eliminate fast-path cache-line bouncing.

⚠️ **Note:** This lock is designed for high-concurrency scenarios such as high-frequency trading order books, ETW/EDR telemetry pipelines, and centralized routing tables.

## Table of Contents
1. [The Problem: Unpredictable Fairness and Starvation](#1-the-problem-unpredictable-fairness-and-starvation)
2. [Key Architectural Highlights](#2-key-architectural-highlights)
3. [The Fairness State Machine](#3-the-fairness-state-machine)
4. [Algorithm & State Layout](#4-algorithm--state-layout)
5. [Locking Semantics & API](#5-locking-semantics--api)
6. [Comparison to Alternative Approaches](#6-comparison-to-alternative-approaches)
7. [When to use](#7-when-to-use-vs-when-not-to-use)
8. [Quick Start](#8-quick-start) 
9. [Benchmarks & Scaling Performance](#9-benchmarks--scaling-performance)
10. [Project Structure](#10-project-structure)
11. [Building Test Code](#11-building-test-code)
12. [Conclusion](#12-conclusion)
13. [License](#13-license)

---

## 1. The Problem: Unpredictable Fairness and Starvation

In highly concurrent C++20 applications, reader-writer locks are useful for protecting data that is frequently read but less frequently mutated. However, the fairness characteristics of the underlying synchronization primitive can become important when contention is sustained.

Standard reader-writer primitives generally do not expose a portable policy for controlling how long a particular reader or writer may remain waiting. Their exact scheduling and wake-up behavior is implementation- and platform-dependent.

Under sustained contention, this can produce:

* **Unpredictable Tail Latencies:** A waiting thread may remain unscheduled or repeatedly lose acquisition opportunities due to scheduler behavior and contention.
* **Reader/Writer Imbalance:** A continuous stream of readers can delay waiting writers, while aggressive writer preference can conversely delay readers.
* **Limited Fairness Control:** Applications generally cannot configure explicit starvation thresholds, writer batching, or escalation behavior through the standard reader-writer-lock interface.

`FairRWLock` moves the fairness policy into user space while retaining an atomic fast path for uncontended acquisition.

The objective is not to replace the operating system scheduler. Instead, the lock controls lock admission and waiter selection so that prolonged contention changes the acquisition policy before starvation becomes unbounded by the lock's own arbitration logic.

---

## 2. Key Architectural Highlights

`FairRWLock` uses a hybrid architecture consisting of an atomic fast path and a spinlock/semaphore slow path.

### Atomic Fast Path
The uncontended read path avoids the internal queue lock and uses a single atomic increment.
On supported platforms where `std::atomic<uint64_t>` is lock-free, this operation is lock-free at the hardware atomic level. The overall lock is not lock-free because its contention path uses an atomic spinlock and C++20 semaphores.

Readers use a speculative atomic increment:

```text
reader
  |
  v
atomic increment
  |
  +-- no WRITE_LOCKED / WRITER_STARVING --> acquire
  |
  +-- contested --------------------------> rollback + slow path
```

This allows the common uncontended reader case to avoid the internal queue lock entirely.

Writers use a compare-and-swap (CAS) operation on the packed state. If the lock is immediately available and the configured fairness limits permit acquisition, the writer can acquire without entering the slow path.

### NUMA-Aware Distributed Readers (Optional)
When the `EnableNUMA` template parameter is `true` and multiple CPU sockets are detected, the lock distributes reader counts across over-aligned stripe arrays. Fast-path readers exclusively increment their localized stripe, eliminating cache-line bouncing on the central state interconnect. Writers assert a global barricade and dynamically drain the distributed counts. 

If the template parameter `EnableNUMA` is set to `false`, no NUMA-related code is compiled or linked into the binary, ensuring zero overhead on single-socket architectures.

### Slow Path: Managed Wait Queues
When the fast path cannot safely grant access, the operation enters the slow path.
The slow path uses:
* a Test-and-Test-and-Set (TTAS) `std::atomic_flag` spinlock for coordinating queue state;
* a `std::counting_semaphore` for waiting readers;
* an intrusive FIFO linked list for waiting writers;
* per-writer `std::binary_semaphore` for targeted writer notification.

Waiting writer nodes are owned by the waiting writer itself. This avoids dynamic allocation during lock acquisition.
The writer queue therefore provides an explicit ordering mechanism rather than leaving writer selection entirely to OS-level wake-up behavior.

### Logical Baton Handoff
The lock explicitly identifies the next eligible waiter.
For example, when the final active reader leaves while writers are queued, the lock can identify the writer at the head of the writer queue and notify that writer directly.

This is a logical baton handoff, not a scheduler-level transfer of execution. The notification identifies the preferred next owner, but the operating system still determines when that thread actually executes.

The distinction is important:
* **FairRWLock controls:** who should acquire next
* **Operating system controls:** when that thread actually runs

### Cache-Line Alignment
The implementation uses 64-byte alignment by default and 128-byte alignment on Apple ARM64.
This is intended to reduce false sharing around frequently accessed atomic state. The exact benefit is microarchitecture- and workload-dependent, so benchmark results should be used to validate the choice for a particular deployment CPU.

---

## 3. The Fairness State Machine

The core feature of `FairRWLock` is its configurable fairness policy, represented by `FairRWLockPolicy`.
The policy parameters form a progression rather than being completely independent knobs. Under sustained contention, the lock progressively changes admission and priority behavior.

### High-Level Progression

```text
                  writer arrives
                       |
                       v
                +--------------+
                | Normal Mode  |
                +--------------+
                       |
             starvationThreshold
                       |
                       v
              +------------------+
              | Writer Starving  |
              | reader gate set  |
              +------------------+
                       |
                 maxWriterWait
                       |
                       v
              +------------------+
              | Override Mode    |
              | writer priority  |
              +------------------+
                    /       \
             limit reached   timeslice active
                  /             \
                 v               |
        +----------------+       |
        | Reader Phase   | <-----+
        +----------------+
```

The important distinction is between reader admission control and writer handoff.
When a writer has waited beyond `starvationThreshold`, the lock asserts `WRITER_STARVING`. This does not forcibly schedule the writer. Instead, it changes the fast-path admission rules so that new readers can no longer continue bypassing the queued writer.

If the lead writer continues waiting beyond `maxWriterWait`, the lock can enter Override Mode. Override Mode gives queued writers stronger priority while still applying configurable writer limits.

### Policy Parameters

| Policy | Description | Default |
| :--- | :--- | :--- |
| **`starvationThreshold`** | Time a queued writer waits before `WRITER_STARVING` is asserted and new fast-path readers are gated. | 3 ms |
| **`maxWriterWait`** | Time after which prolonged writer waiting can trigger Override Mode. | 50 ms |
| **`overrideTimeslice`** | Maximum duration of the current Override Mode phase. | 100 ms |
| **`writerBatchLimit`** | Maximum number of writers that may acquire consecutively during an override phase before reader admission is restored. | 2 |
| **`maxConsecWriters`** | Optional maximum number of consecutive writers under normal operation. 0 disables the limit. | 0 |
| **`maxYieldsBeforeBypass`** | Number of scheduler-yield attempts made when a writer reaches a fairness limit before allowing the writer to bypass that limit when necessary. | 10 |

 ⚠️ **Note:** `maxWriterWait` is a fairness escalation threshold, not a hard real-time deadline. The lock cannot guarantee a maximum wall-clock acquisition time because thread scheduling remains under the operating system.

### Writer Starvation Threshold
When a queued writer reaches `starvationThreshold`, the lock sets `WRITER_STARVING` in the atomic state.
The purpose is to stop a continuous stream of new readers from repeatedly taking the fast path while an existing writer is waiting.
The threshold is therefore an admission-control threshold, not a scheduling deadline.

Conceptually:
* **Before threshold:** `readers ---> fast path ---> acquire` | `writer ---> queue`
* **After threshold:** `readers ---> fast path ---> rejected` | `writer ---> queue/head ---> priority`

Existing readers are allowed to finish normally. The objective is to prevent new readers from extending the active reader phase indefinitely.

### Override Mode
If the lead writer remains queued beyond `maxWriterWait`, the lock can enter Override Mode.
Override Mode:
* marks the lock as writer-starving;
* records an override expiration time;
* resets the writer batch counter;
* gives queued writers priority;
* limits the number of consecutive writers through `writerBatchLimit`;
* eventually permits a reader phase once the configured writer limits are reached.

Override Mode is therefore an escalation mechanism for severe writer starvation, rather than the normal operating mode of the lock.

### Reader Resurgence
Writer preference is intentionally not allowed to continue indefinitely.
Once the relevant writer limit is reached, the lock can clear the starvation gate and wake waiting readers.
This produces a controlled cycle:
`readers` → `writer contention` → `writer gate` → `writer acquisition` → `writer batch / override limit` → `reader phase` → `normal arbitration`

The exact behavior is controlled by the configured policy rather than by a fixed reader-preference or writer-preference rule.

---

## 4. Algorithm & State Layout

### 4.1 Two-Tier Architecture
The lock separates the common uncontended case from contention management.

```text
                         +----------------+
                         | Lock operation |
                         +-------+--------+
                                 |
                    +------------+------------+
                    |                         |
                 Fast path                Slow path
                    |                         |
             atomic state           spinlock + semaphores
                    |                         |
             +------+-------+        +--------+--------+
             |              |        |                 |
           Reader         Writer   Readers           Writers
             |              |        |                 |
          fetch-add         CAS   counting_sem    FIFO queue
```

The fast path is deliberately small. The slow path contains the fairness machinery.

### 4.2 64-Bit State Packing

The primary lock state is stored in `std::atomic<uint64_t> m_state`.
The state is divided into an upper 32-bit writer streak and a lower 32-bit state field.

```text
63                         32 31    30    29    28    27    26         0
+----------------------------+-----+-----+-----+-----+-----+-----------+
| Consecutive writer count   |wait |write| un- |starv|guard|  readers  |
|                            |ers  |lock | used|ing  |     |           |
+----------------------------+-----+-----+-----+-----+-----+-----------+
```

The lower 32 bits contain:
* `bit 28`: `WRITER_STARVING`
* `bit 30`: `WRITE_LOCKED`
* `bit 31`: `HAS_WAITERS`
* `bits 0-27`: active reader count

The upper 32 bits contain:
* `bits 32-63`: consecutive writer count

The reader count therefore uses 28 bits of the packed state.
The packed representation allows the fast path to inspect reader activity, writer ownership, and fairness state with a single atomic value.

### 4.3 Consecutive Writer Tracking
The upper 32 bits track the number of consecutive writer acquisitions.
A successful writer acquisition increments the streak.
A reader acquisition clears the streak.
This allows the policy to distinguish between `writer -> writer -> writer -> writer` and `writer -> readers -> writer`. The distinction is important for `maxConsecWriters` and writer batching.

### 4.4 Writer Queue
Waiting writers are represented by intrusive nodes:

```cpp
struct WriterNode
{
    WriterNode* prev;
    WriterNode* next;
    std::binary_semaphore sem{0};
    time_point ts;
};
```

The queue is maintained as `m_headWriter` to `m_tailWriter`. The queue is FIFO for writers entering the slow path.
Each waiting writer retains ownership of its own queue node for the duration of its wait. The node is removed when the writer acquires the lock or abandons the wait due to timeout.
This design avoids dynamic allocation during lock acquisition.

### 4.5 Writer Starvation Detection
Each writer records its queue-entry timestamp.
The lead writer is evaluated against `starvationThreshold` and `maxWriterWait`.
The first threshold asserts `WRITER_STARVING`.
The second can activate Override Mode.
The use of `std::chrono::steady_clock` ensures that these durations are measured against a monotonic clock rather than wall-clock adjustments.

### 4.6 Reader Admission
Readers first attempt the atomic fast path. A reader can acquire immediately when the relevant state permits it.
If `WRITE_LOCKED` or `WRITER_STARVING` prevents fast-path acquisition, the speculative reader increment is rolled back and the reader enters the slow path.
This is important because the starvation mechanism does not require every reader to acquire the internal spinlock. Only readers encountering the fairness gate or another contention condition need to enter the slow path.

### 4.7 Writer Acquisition
Writers first inspect the packed state.
If there are no active readers, no active writer, and no blocking fairness limit, the writer attempts a CAS to acquire the write bit.
If the fast path cannot acquire the lock, the writer enters the FIFO queue.
Only the writer at the head of the queue is eligible for normal queued acquisition.
This is the primary mechanism by which the lock avoids repeatedly selecting arbitrary waiting writers.

### 4.8 Policy Limits and Scheduler Yield
When the lead writer encounters a configured writer limit while waiting for the appropriate reader phase, the implementation can temporarily yield to the operating-system scheduler.
The policy parameter `maxYieldsBeforeBypass` limits the number of these yield attempts.
The purpose is to avoid turning a fairness limit into an indefinite software wait when the expected reader transition does not occur.
After the configured number of yield attempts, the writer may bypass the limit when the acquisition conditions permit it.
This is a deliberate graceful-degradation mechanism rather than a spin loop.

---

## 5. Locking Semantics & API

`FairRWLock` provides both manual lock/unlock operations and RAII guards.

### Read Lock
`bool ReadLock(duration timeout = duration::max());`
Attempts to acquire shared/read access.
Returns `true` if acquired, or `false` if the timeout expires before acquisition. The default `duration::max()` means wait indefinitely.

### Write Lock
`bool WriteLock(duration timeout = duration::max());`
Attempts to acquire exclusive/write access.
Returns `true` if acquired, or `false` if the timeout expires before acquisition.

### Try Read Lock
`bool TryReadLock() noexcept;`
Attempts to acquire read access without intentionally blocking.
The operation may inspect the slow-path spinlock using `std::try_to_lock`, but does not wait for the queue lock or for another lock holder.

### Try Write Lock
`bool TryWriteLock() noexcept;`
Attempts to acquire write access without intentionally blocking.
As with `TryReadLock()`, the slow-path queue lock is only attempted with `std::try_to_lock`.

### Read Unlock
`void ReadUnlock() noexcept;`
Releases a read acquisition.
When the final active reader leaves and writers are waiting, the unlock operation can trigger the writer baton handoff.

### Write Unlock
`void WriteUnlock() noexcept;`
Releases a write acquisition.
The unlock path evaluates the active fairness policy and determines whether to hand off to the next writer, transition to a reader phase, clear an override, or return to normal arbitration.

### Diagnostic Logging (Debug Builds)
`void SetLogger(std::function<void(std::string_view)> lg);`
Assigns an optional diagnostic logger for debugging lock state transitions.
This function is active in Debug builds (`#ifndef NDEBUG`) and compiles to a zero-cost abstraction in Release builds. To prevent data races and unsafe functor replacement during active lock contention, re-attachment is forbidden and will throw a `std::logic_error`.

### RAII Guards
The preferred usage model is RAII via `FairRWLock::ReadGuard` and `FairRWLock::WriteGuard`.
Both guards:
* acquire the lock during construction;
* release it during destruction if acquisition succeeded;
* are non-copyable;
* are movable;
* support optional acquisition timeouts;
* provide `operator bool()` to test whether acquisition succeeded.

```cpp
FairRWLock<> lock;

{
    FairRWLock<>::ReadGuard guard(lock);

    if (guard)
    {
        // Protected read access.
    }
}
```

A timeout can be supplied:

```cpp
using namespace std::chrono_literals;

FairRWLock<>::WriteGuard guard(lock, 50ms);

if (guard)
{
    // Protected write access.
}
else
{
    // Acquisition timed out.
}
```

### Locking Semantics
`FairRWLock` is intended for conventional non-recursive reader/writer locking.
The lock does not provide a read-to-write upgrade operation or a write-to-read downgrade operation.
Applications should use the RAII guards where practical to ensure that successful acquisitions are released on every control-flow path.

### Timeout Semantics
Timeouts use `std::chrono::steady_clock`.
This means timeout measurement is monotonic and is not affected by changes to the system wall clock.
A timeout is a maximum waiting interval for acquisition; it is not a guarantee that the thread will be scheduled at any particular instant.

---

## 6. Comparison to Alternative Approaches

The project includes simpler reader-writer lock implementations to provide architectural points of comparison.

### `std::shared_mutex`
The standard C++ reader-writer primitive provides an established, portable interface.
Its primary advantages are a standard C++ API, mature implementations, minimal application-level complexity, and platform-specific optimization beneath the standard interface.
However, the standard interface does not expose a configurable fairness policy equivalent to `FairRWLockPolicy`. `FairRWLock` instead makes fairness escalation an explicit part of the algorithm.

| Property                                   | `std::shared_mutex`      | `FairRWLock`          |
| :---                                       | :---                     | :---                  |
| **Standard C++ API**                       | ✓                        | —                     |
| **Shared/exclusive locking**               | ✓                        | ✓                     |
| **Configurable starvation threshold**      | —                        | ✓                     |
| **Explicit writer queue**                  | Implementation-dependent | ✓                     |
| **Writer starvation escalation**           | Implementation-dependent | ✓                     |
| **Configurable writer batching**           | —                        | ✓                     |
| **Atomic fast-path state**                 | Implementation-dependent | ✓                     |
| **Application-controlled fairness policy** | —                        | ✓                     |
| **Contention Primitives**                  | OS-dependent             | Spinlock + Semaphores |

`FairRWLock` does not attempt to claim a hard real-time guarantee that `std::shared_mutex` lacks. Its distinction is that fairness policy is explicitly implemented and configurable.

### `TextbookReaderPrefLock`
`TextbookReaderPrefLock` represents a conventional textbook reader-preference implementation using one mutex, reader and writer condition variables, and explicit active-reader and active-writer state.
Its policy is intentionally simple: If no writer is active, readers may acquire. Waiting writers do not prevent new readers.
Under sustained read traffic, this can allow writers to remain waiting indefinitely. `FairRWLock` changes that behavior by allowing a waiting writer to eventually assert a reader admission gate.
The architectural trade-off is additional complexity in exchange for explicit starvation management and an atomic reader fast path.

### `MomentumRWLock`
`MomentumRWLock` uses a mutex/CV design that attempts to manually toggle priority between readers and writers.
It uses a priority flag to favor a waiting writer after the active reader/writer phase completes.
The primary architectural difference is that `MomentumRWLock` performs its state coordination through an OS-level mutex, whereas `FairRWLock` separates the common uncontended path (atomic state) from contention/fairness management (atomic spinlock + C++20 semaphores + writer queue).
`FairRWLock` therefore introduces substantially more state and policy machinery in exchange for avoiding the internal mutex on the common atomic fast path.

---

## 7. When to Use vs. When Not to Use

`FairRWLock` occupies a specific architectural niche. It is optimized for systems that require the high throughput of an optimistic lock-free fast path, but cannot afford the unpredictable tail latencies of OS-mediated lock contention.

### ✅ When to Use

* **Predictable Writer Latency (Soft Real-Time):** Ideal for ETW/EDR telemetry pipelines, financial trading routing tables, or configuration management where an endless stream of parallel reads must not permanently starve a critical write/update operation.
* **Cross-Platform Determinism:** When you require identical lock-arbitration and admission semantics across Windows, Linux, and macOS. It normalizes behavior so you are not subjected to the whims of the OS scheduler (such as 100% POSIX writer starvation on Linux, or 50+ ms SRW latency spikes on Windows).
* **Zero-Allocation Contention Paths:** When your hot paths strictly forbid heap allocations. The intrusive queue allocates `WriterNode` structures directly on the waiting thread's stack.
* **Architecture-Specific Tuning:** When deploying to platforms with known false-sharing footprints, allowing you to leverage the explicit 128-byte cache-line padding (e.g., Apple Silicon / ARM64).
* **NUMA / High-Core-Count Read Scaling:** When deploying to 64+ core server topologies where read traffic is continuous and highly parallel. Enabling the `EnableNUMA` parameter distributes reader counts across physical sockets, preventing global L1 cache-line invalidation on the central `m_state` counter.

### ❌ When Not to Use

* **Hard Real-Time Systems:** If your architecture requires strict Worst-Case Execution Time (WCET) bounds, this lock's graceful degradation and yield backoffs are inappropriate. Use a strict Phase-Fair lock (`pflock`) instead.
* **Low-Contention General Purpose Logic:** If thread starvation is not a measured issue in your application profile, the standard `std::shared_mutex` provides a simpler dependency and adequate performance.

---

## 8. Quick Start

Include the header:

```cpp
#include "fair_rw_lock.h"
#include <iostream>

FairRWLock<> rwLock;

void ReadData()
{
    FairRWLock<>::ReadGuard guard(rwLock);

    if (guard)
    {
        std::cout << "Reading data securely...\n";
    }
}

void WriteData()
{
    FairRWLock<>::WriteGuard guard(rwLock);

    if (guard)
    {
        std::cout << "Writing data securely...\n";
    }
}
```

### Enabling NUMA Support
To instantiate a NUMA-aware lock, pass `true` as the second template parameter:

```cpp
// Instantiates the lock with NUMA-aware reader distribution
FairRWLock<DefaultFairRWLockPolicy, true> numaRwLock;
```

### Custom Fairness Policy
The policy can be configured when constructing the lock:

```cpp
using namespace std::chrono_literals;

// Derive from the default policy to inherit base types and unchanged limits
struct PersistencePolicy : DefaultFairRWLockPolicy
{
    static constexpr auto maxWriterWait       = 15ms;
    static constexpr auto overrideTimeslice   = 50ms;
    static constexpr int  writerBatchLimit    = 2;
    static constexpr int  maxConsecWriters    = 3;
    static constexpr auto starvationThreshold = 4ms;
};

FairRWLock<PersistencePolicy> customLock;
```

The policy should be tuned according to the workload rather than treated as a universal optimal configuration.
---

## 9. Benchmarks & Scaling Performance

To evaluate `FairRWLock`, the included benchmark suite compares it against the standard C++ reader-writer implementation (`std::shared_mutex`), a textbook reader-preference implementation (`TextbookReaderPrefLock`), and a momentum-based implementation (`MomentumRWLock`) across different hardware and operating-system environments.

The complete benchmark and test suite is included in the repository for reproducibility.

### Windows 10/11 Performance

#### Test Methodology

* **Target Platforms:** Windows 10/11.
* **Compiler:** The test suite was compiled with MSVC 2026.
* **Workload:** High-frequency mixed-contention scenarios scaled to the logical core count of each CPU, with two writers and the remaining logical cores used as readers.
* **Starvation Testing:** Continuous reader oversubscription is used to intentionally stress writer acquisition, including workloads with up to 40 concurrent readers.
* **Benchmark Duration:** Throughput tests run continuously for 15 seconds per implementation.

#### Total Throughput Speedup (Mixed Contention)

The following results show total operations processed per second during the sustained mixed-contention benchmark:

| Hardware Topology                            | `std::shared_mutex` Ops/Sec | `FairRWLock` Ops/Sec | Total Ops Speedup   |
| :---                                         | ---:                        | ---:                 | ---:                |
| **Intel Core i7-8086K** (Desktop, 12-Thread) | ~8.38 Million               | ~11.95 Million       | **+42.48% Faster**  |
| **Intel Core i7-1165G7** (Mobile, 8-Thread)  | ~7.92 Million               | ~18.81 Million       | **+137.48% Faster** |
| **Intel Core i7-12700H** (Hybrid, 20-Thread) | ~7.85 Million               | ~20.53 Million       | **+161.33% Faster** |

The results show a substantial throughput advantage for `FairRWLock` on the tested Windows systems. The magnitude of the improvement varies with CPU architecture and workload.

#### Writer Starvation Resilience (Reader Oversubscription)

This test evaluates how long a writer must wait to acquire the lock while subjected to a continuous stream of read requests. For this benchmark, the `starvationThreshold` in the `FairRWLock` policy was explicitly set to 3 ms.

| Implementation             | i7-8086K (24 Readers) | i7-1165G7 (16 Readers) | i7-12700H (40 Readers) |
| :---                       | ---:                  | ---:                   | ---:                   |
| **TextbookReaderPrefLock** | 100% Starvation       | 100% Starvation        | 100% Starvation        |
| **MomentumRWLock**         | 100% Starvation       | 100% Starvation        | 100% Starvation        |
| **`std::shared_mutex`**    | 0.50 ms avg wait      | 0.48 ms avg wait       | 0.49 ms avg wait       |
| **FairRWLock**             | 2.88 ms avg wait      | 3.24 ms avg wait       | 2.93 ms avg wait       |

#### Explanation of Windows Results

On Windows, the tested `std::shared_mutex` implementation is backed by Slim Reader/Writer (SRW) locks, which perform well under reader oversubscription and successfully prevent the total writer starvation observed in simpler implementations.

However, since scheduling remains dependent on the operating system, the tested `std::shared_mutex` implementation exhibited significant writer latency spikes under peak contention. In our benchmark latency distributions, maximum observed writer latencies ranged from **39.38 ms to 57.22 ms** across the tested hardware topologies.

`FairRWLock` provides application-controlled fairness rather than relying entirely on platform-specific scheduling behavior. With a 3 ms `starvationThreshold`, average writer wait times remained close to the configured threshold (~2.88 ms to 3.24 ms), while maximum observed writer latencies were **5.01 ms to 17.67 ms**, depending on policy configuration and thread count.

This threshold functions as an admission-control gate rather than a hard real-time scheduling deadline; the operating system still determines when a waiting thread actually executes.

The throughput results represent the primary difference on the tested Windows systems, with `FairRWLock` ranging from **42.48% to 161.33% higher throughput** than `std::shared_mutex` under mixed contention.

---

### Linux Performance

#### Test Methodology

* **Target Platforms:** Fedora and Ubuntu running as VMware guests with 8 logical CPUs on an Intel Core i7-8086K host system.
* **Compilers:** The Fedora test suite was compiled with g++ 16, and the Ubuntu test suite used g++ 15.
* **Workload:** High-frequency mixed-contention scenarios using 6 readers and 2 writers.
* **Starvation Testing:** Continuous reader oversubscription using 16 concurrent readers to intentionally stress writer acquisition.
* **Benchmark Duration:** Throughput tests run continuously for 15 seconds per implementation.

#### Total Throughput Speedup (Mixed Contention)

The following results show total operations processed per second during the sustained mixed-contention benchmark:

| Linux Distribution      | `std::shared_mutex` Ops/Sec | `FairRWLock` Ops/Sec | Total Ops Speedup |
| :---                    | ---:                        | ---:                 | ---:              |
| **Fedora** (VM, 8-Core) | ~6.41 Million               | ~6.75 Million        | **+5.27% Faster** |
| **Ubuntu** (VM, 8-Core) | ~6.04 Million               | ~6.34 Million        | **+5.06% Faster** |

The throughput advantage on the tested Linux systems is modest (+5.06% to +5.27%) compared to Windows, demonstrating that relative lock performance is highly dependent on platform and workload.

#### Writer Starvation Resilience (Reader Oversubscription)

This test evaluates writer acquisition under sustained reader pressure. The `starvationThreshold` was again set to 3 ms.

| Implementation             | Fedora (16 Readers) | Ubuntu (16 Readers) |
| :---                       | ---:                | ---:                |
| **TextbookReaderPrefLock** | 100% Starvation     | 100% Starvation     |
| **MomentumRWLock**         | 100% Starvation     | 100% Starvation     |
| **`std::shared_mutex`**    | 100% Starvation     | 100% Starvation     |
| **FairRWLock**             | 3.18 ms avg wait    | 3.25 ms avg wait    |

#### Explanation of Linux Results

On Linux, `std::shared_mutex` is typically implemented using POSIX `pthread_rwlock_t`, which can exhibit strong reader-preference characteristics under sustained reader oversubscription.

During the starvation benchmark, waiting writers were unable to acquire `std::shared_mutex`, reaching the **15.00-second** test timeout. In contrast, `FairRWLock` transferred lock admission to waiting writers at ~3.18 ms to 3.25 ms in accordance with its 3 ms policy, with maximum observed writer latencies of **4.27 ms to 4.96 ms**.

This difference represents the primary result on Linux: `FairRWLock` provides a consistent application-level fairness policy across platforms, preventing application-level writer starvation from depending entirely on the fairness behavior of the platform's standard-library implementation.

As with Windows, the configured starvation threshold should be understood as an admission-control policy rather than a hard real-time execution deadline.

---

### Performance Summary

The benchmark results highlight several key observations:

* **Windows Throughput:** `FairRWLock` demonstrated substantial throughput gains over `std::shared_mutex`, ranging from **+42.48% to +161.33%** on the tested systems.
* **Linux Fairness:** While Linux throughput gains were modest (~+5%), `FairRWLock` prevented the 100% writer starvation observed in `std::shared_mutex` under the tested workload, reducing observed writer wait times from the **15.00-second test timeout** to **under 5 ms**.
* **Tail Latency:** On Windows, the tested `std::shared_mutex` implementation exhibited maximum writer latency spikes of **39 ms – 57 ms**, while `FairRWLock` recorded maximum observed writer latencies of **~5 ms – 17 ms** depending on workload and policy configuration.
* **Platform Differences:** `std::shared_mutex` behavior varied significantly between Windows (SRW lock-backed) and Linux (POSIX-backed), highlighting that standard primitives do not guarantee identical scheduling or fairness behavior across platforms.
* **Fairness Policy:** The simpler implementations consistently suffered from starvation under continuous reader traffic, whereas `FairRWLock` predictably granted admission around the configured policy thresholds.

These results reflect empirical observations under specific test configurations rather than universal performance guarantees. Different CPU microarchitectures, operating systems, standard library implementations, thread counts, and read/write ratios will yield different results.

The complete benchmark harness is included with the project so that these measurements can be reproduced and evaluated under custom application workloads.

---

## 10. Project Structure

The repository is organized into distinct layers to separate the core lock logic from the cross-platform test suite:

* **`fair_rw_lock/`**: Contains the fair RW lock implementation.
    * `fair_rw_lock.h`: The cross-platform C++20 header for Linux, macOS, and Windows applications.

* **`test/`**: Cross-platform user-mode performance test suite.
    * `fair_rw_lock_test.cpp`: Test suite implementation.
    * `textbook_reader_pref_lock.h`: Provided for reference; a classic implementation of a reader-preference lock.
    * `momentum_rw_lock.h`: Provided for reference; a momentum RW lock (manages lock handover to pass control between reading and writing phases).
---

## 11. Building Test Code

### Linux / macOS
Requires a C++20 compliant compiler (GCC or Clang). 

If you are compiling a NUMA-aware build on Linux, you must install the `libnuma` development packages first:

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install libnuma-dev numactl

# RHEL / Fedora / CentOS
sudo dnf install numactl-devel
```

*Using the Build Script (Recommended)*

```bash
./build.sh [--clean | -c] [--type Release | Debug] [--compiler g++ | clang++] [--numa]
```

```text
Defaults:
 - Build type: Release
 - Compiler: g++
 - Reuses existing build/
 - NUMA support: OFF

Options:
 -c, --clean     → Remove build/ before building
 -t, --type      → Set build type (Release or Debug)
 --compiler      → Choose compiler (g++ or clang++)
 --numa          → Enable NUMA-aware lock capabilities in the build
 -h, --help      → Show help and exit
 ```

```bash
# Standard Release build using default C++ compiler
./build.sh

# Clean Release build with NUMA support enabled
./build.sh --clean --type Release --numa
```

*Using CMake Manually*

The included `CMakeLists.txt` automatically detects your platform, handles threading library links, and configures optimized build flags (including IPO/LTO). To enable NUMA-aware locks, pass `-DENABLE_NUMA=ON` to the configuration step.

```bash
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_NUMA=ON
    cmake --build . -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

*Without CMake or `build.sh`*

If you are integrating `fair_rw_lock.h` directly into your own build system and require NUMA support on Linux, you must explicitly inject the `ENABLE_NUMA=1` preprocessor definition (e.g., `#define ENABLE_NUMA=1` or `-DENABLE_NUMA=1` in your compiler flags) and ensure your binary links against `libnuma` (`-lnuma`).

### Windows
Native Visual Studio 2026 Solution (.slnx) and Project (.vcxproj) files are included in the repository.

* **Visual Studio 2026:** Requires C++20 support (`v145` toolset). The provided project file includes target builds `Debug`/`Release` for non-NUMA aware lock compilation and `Debug-NUMA`/`Release-NUMA` for NUMA-aware lock compilation.

---

## 12. Conclusion

By moving fairness policies into user space and separating the atomic uncontended path from the managed queue, `FairRWLock` resolves the unpredictable tail latencies inherent in OS-mediated reader-writer primitives. The core advantage is predictable admission control: rather than relying on the black-box scheduling of POSIX or SRW locks, the application deterministically caps writer starvation at the lock-arbitration level. 

Based on this architecture's mechanical sympathy, the performance and fairness advantages will compound across specific emerging hardware topologies and deployment environments:

*   **High-Core-Count Microarchitectures (Threadripper / Xeon / EPYC):** On systems with massive core counts, continuous read-streams can permanently starve writers using standard primitives due to localized cache coherence storms. The logical baton handoff mitigates this by preemptively gating new fast-path readers, ensuring configuration updates or writer phases execute predictably regardless of the parallel read volume.
*   **Asymmetric / Hybrid Architectures (Intel Big.LITTLE):** Thread scheduling volatility across P-Cores and E-Cores can cause waiting writers to yield inconsistently. The tiered hardware backoff and bounded scheduler-yield limits (`maxYieldsBeforeBypass`) specifically prevent threads descheduled on slower E-Cores from deadlocking the fairness state machine, safely bypassing limits if reader transitions stall.
*   **ARM64 Cloud Instances (AWS Graviton / Apple Silicon):** The deliberate 128-byte cache-line alignment specifically targets the destructive interference patterns common on Apple Silicon and modern ARM server chips. Combined with optimized `compare_exchange_weak` usage on the fast path, the lock actively avoids nested hardware spin-loops on processors with weak memory models.
*   **Cross-Platform Telemetry & EDR Pipelines:** Since standard primitive behavior diverges wildly between Windows (SRW locks) and Linux (POSIX rwlocks), high-throughput cross-platform pipelines often suffer from inconsistent telemetry drops under peak load. This user-space fairness state machine guarantees an identical lock admission profile across OS environments, bounding worst-case writer latency to the configured starvation thresholds rather than OS scheduler quirks.

## 13. License

This project is licensed under the Apache License, Version 2.0. 

You may not use this file except in compliance with the License. You may obtain a copy of the License at:
[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, 
either express or implied. See the `LICENSE` file for the specific language governing permissions and limitations.