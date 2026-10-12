# vector-db-engine
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![CI](https://github.com/sohamshewani/vector-db-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/sohamshewani/vector-db-engine/actions) [![Security: Hardened](https://img.shields.io/badge/Security-Hardened-blue.svg)](https://github.com/sohamshewani/vector-db-engine) [![Stars](https://img.shields.io/github/stars/sohamshewani/vector-db-engine?style=flat)](https://github.com/sohamshewani/vector-db-engine)

## 1. Executive Summary & Problem Space

`vector-db-engine` is a high-performance, embedded vector database engine written in modern C (C11). Designed from the ground up for low-latency similarity search and dense vector analytics, it targets systems demanding deterministic execution profiles, minimal memory overhead, and predictable cache utilization. 

Traditional vector databases often suffer from severe memory amplification, lock contention during concurrent insertions, and poor cache locality due to pointer-heavy graph structures. `vector-db-engine` resolves these bottlenecks through:
* **Contiguous Columnar Layouts (`src/core/column.c`)**: Maximizing SIMD vectorization and hardware prefetching by packing high-dimensional embeddings into sequential memory blocks.
* **Lock-Free Indexing & Bitmap Filtering (`src/index/bitmap.c`)**: Utilizing atomic bitmap operations to eliminate mutex bottlenecks during high-throughput parallel queries.
* **Zero-Copy Execution Pipelines (`src/execution/vector_engine.c`)**: Processing vectors in vectorized batches to minimize instruction cache thrashing and function call overhead.
* **Custom Arenas & Bounded Allocators (`include/vector_db/allocator.h`)**: Bypassing OS-level heap fragmentation via deterministic region-based memory pools.

---

## 2. Structural ASCII Architecture / Dataflow Diagram

The system is architected in distinct, decoupled layers separating memory management, core data representation, index structures, and execution pipelines.

```
 +-----------------------------------------------------------------------+
 |                     Client / Application Layer                        |
 +-----------------------------------------------------------------------+
                                    |
                                    v
 +-----------------------------------------------------------------------+
 |                  Execution & Vector Engine (src/execution/)           |
 |   +------------------------+  +------------------------------------+  |
 |   | Batch Dispatcher       |->| Vectorized Evaluation Pipeline     |  |
 |   +------------------------+  +------------------------------------+  |
 +-----------------------------------------------------------------------+
            |                                         |
            v                                         v
 +----------------------------------+  +---------------------------------+
 |      Index Layer (src/index/)    |  |     Core Engine (src/core/)     |
 |  +----------------------------+  |  |  +---------------------------+  |
 |  | Atomic Bitmap Filters      |  |  |  | Columnar Storage          |  |
 |  | (`bitmap.c`)               |  |  |  | (`column.c`)              |  |
 |  +----------------------------+  |  |  +---------------------------+  |
 +----------------------------------+  |  | Compression Engine        |  |
            |                          |  | (`compression.c`)         |  |
            |                          |  +---------------------------+  |
            +--------------------+     +---------------------------------+
                                 |                    |
                                 v                    v
 +-----------------------------------------------------------------------+
 |                Allocator & Memory Subsystem (include/)                |
 |  +----------------------------------+  +---------------------------+  |
 |  | Region / Arena Allocators        |  | Type Safety & Limits      |  |
 |  | (`allocator.h`)                  |  | (`types.h`)               |  |
 |  +----------------------------------+  +---------------------------+  |
 +-----------------------------------------------------------------------+
                                    |
                                    v
 +-----------------------------------------------------------------------+
 |                    OS Kernel / Hardware / SIMD                        |
 +-----------------------------------------------------------------------+
```

---

## 3. Concurrency Model & Memory Invariants

`vector-db-engine` employs a hybrid concurrency model optimized for read-heavy analytical workloads and concurrent point insertions:
* **Thread Safety**: Read paths are entirely lock-free, leveraging epoch-based memory reclamation and atomic pointer swaps. Write paths use fine-grained partition locks to prevent global mutex serialization.
* **Memory Ownership**: Memory arenas (`include/vector_db/allocator.h`) enforce strict ownership hierarchies. Sub-allocations inherit the lifetime of their parent arena, completely eliminating double-frees and memory leaks during complex query transformations.
* **Atomic Primitives**: Synchronization across bitmap filters (`src/index/bitmap.c`) relies exclusively on C11 atomic operations (`stdatomic.h`) with sequentially consistent and acquire-release memory orderings, preventing race conditions during concurrent bit-flipping operations.

---

## 4. Performance Characteristics & Algorithmic Complexity Table

| Operation | Target Subsystem | Time Complexity | Space Complexity | Latency Target (p99) |
| :--- | :--- | :--- | :--- | :--- |
| **Point Insert** | `src/core/column.c` | $O(1)$ amortized | $O(D)$ ($D$ = dimensions) | $< 1.2\mu\text{s}$ |
| **Vector Scan** | `src/execution/vector_engine.c`| $O(N \cdot D)$ | $O(1)$ auxiliary | $< 500\mu\text{s}$ ($N=10^6$) |
| **Bitmap Filter** | `src/index/bitmap.c` | $O(N / 64)$ | $O(N / 8)$ bytes | $< 15\mu\text{s}$ |
| **Compression** | `src/core/compression.c` | $O(N \cdot D)$ | $O(N \cdot D)$ | $< 2.0\text{ms}$ / MB |
| **Memory Allocation**| `include/vector_db/allocator.h`| $O(1)$ | $O(1)$ overhead | $< 25\text{ns}$ |

---

## 5. Security Architecture & Threat Model

Security is enforced at the compilation, allocation, and boundary-check layers:
* **Input Boundaries**: All incoming vector payloads pass through strict validation layers defined in `include/vector_db/types.h`, verifying dimension bounds, alignment constraints, and NaN/Infinity numeric filters.
* **Memory Corruption Protections**: The engine makes extensive use of bounded memory operations (`memcpy_s`-style guarantees), canary-guarded arenas, and explicit zeroing of sensitive buffers upon deallocation to prevent information leakage.
* **Sanitization**: Continuous integration pipelines enforce strict checks with AddressSanitizer (ASan), UndefinedBehaviorSanitizer (UBSan), and Valgrind memory leak verifications.
* **Threat Vectors Mitigated**: Stack/heap buffer overflows, integer overflows during batch size multiplications, use-after-free vulnerabilities, and timing attacks on index lookups.

---

## 6. Build, Verification & Benchmarking Recipes

### Prerequisites
* A C11-compliant compiler (`gcc` >= 9.0 or `clang` >= 10.0)
* GNU Make (`make`)
* Standard POSIX development tools

### Compilation & Build
To build the static library, test suites, and benchmarks in release mode with optimizations enabled:
```bash
make clean
make release
```

### Running Test Suites (Invariant Verification)
To execute the invariant verification test suite and validate memory safety invariants:
```bash
make test
```

### Executing Performance Benchmarks
To run the high-performance throughput and latency benchmarks (`benchmarks/bench_engine.c`):
```bash
make bench