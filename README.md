# Lock-Free GPU Slab Allocator

A thread-safe memory allocator designed for massively parallel GPU environments, implementing hierarchical slab allocation with lock-free synchronization primitives.

## Overview

This project implements a custom memory management system for CUDA-enabled GPUs that addresses the unique challenges of allocating memory across thousands of concurrent threads. 
Traditional CPU allocators fail at GPU scales due to unprecedented contention levels, making lock-free atomic operations essential for performance.

### Key Features

- **Lock-Free Concurrency**: All operations use atomic CAS (Compare-And-Swap) without traditional locks
- **Hierarchical Design**: Three-tier architecture (Arena → Slab → Object) for efficient management
- **Variable Object Sizes**: Dynamic capacity calculation supporting 8-512 byte allocations
- **Cross-Platform**: Conditional compilation for both CUDA device code and CPU host code
- **Extensive Testing**: Stress tested with 16,384+ concurrent GPU threads

## Architecture

### Components

**SlabArena**: Top-level manager organizing multiple fixed-size slabs (default 4KB each)
- Maintains free list of available slabs using intrusive linked list
- Implements round-robin fallback for slab acquisition
- Supports up to 256 slabs (1MB total arena size in default configuration)

**SlabProxy**: Per-slab metadata managing allocation state through atomic operations
- `allocState`: 64-bit packed value (upper 32 bits: object size, lower 32 bits: allocation count)
- `allocMask`: Bitmask tracking which objects are allocated (supports up to 64 objects per element)
- `reservationState`: Lifecycle state (FREE, PARTIAL, FULL)

**SimpleAllocator**: User-facing interface providing allocation/deallocation primitives
- Manages object size consistency
- Tracks allocation statistics
- Handles slab lifecycle coordination

### Memory Layout

```
Slab Structure (4KB):
┌─────────────────────────────────────┐
│  Allocation Bitmask (8-64 bytes)    │  <- Tracks allocated objects
├─────────────────────────────────────┤
│  Object 0 (variable size)           │
│  Object 1 (variable size)           │
│  ...                                │
│  Object N (variable size)           │
└─────────────────────────────────────┘
```

## Thread Safety Mechanisms

### Atomic Operations

All critical sections use acquire-release memory ordering:
- `load_acquire`: Ensures subsequent reads see prior writes
- `store_release`: Makes all prior writes visible to other threads
- `CAS_acq_rel`: Atomic compare-and-swap with full ordering guarantees

### Race Condition Mitigations

**Stale Mask Load Race**
- Problem: Reusing loaded mask values across CAS retries
- Solution: Fresh atomic load on each iteration

**Count-Bit Ordering Race**
- Problem: Bit allocation before count increment created inconsistent states
- Solution: Count-first reservation with rollback on bit allocation failure

**Index Calculation Error**
- Problem: Hardcoded stride assumptions violated by variable sizes
- Solution: Dynamic capacity calculation per object size

## Building

### Prerequisites

- CUDA Toolkit 11.0+
- C++14 compatible compiler
- NVIDIA GPU with compute capability 3.5+

### Compilation

```bash
nvcc -std=c++14 -arch=sm_70 test_gpu.cu -o gpu_allocator_test
```

For CPU-only testing (compilation check):
```bash
g++ -std=c++14 -DGPU_ONLY allocator.h -c
```



## Testing

The project includes comprehensive testing infrastructure:

### Test Suites

**allocatorTestKernel**: Standard concurrent allocation test
- 4,096 threads (256 threads × 16 blocks)
- 1,000 iterations per thread
- Mixed allocation/deallocation workload
- Data corruption detection via test patterns

**stressTestKernel**: High-contention stress test
- 16,384 threads (512 threads × 32 blocks)
- 2,000 iterations per thread
- 80% allocation rate for maximum pressure
- 3-second timed execution
  

### Running Tests

```bash
./gpu_allocator_test
```

Expected output includes:
- Throughput (operations/second)
- Success rate percentage
- Leak detection results
- Arena utilization statistics

## Performance

Preliminary results on NVIDIA GPUs:
- Throughput: Millions of operations per second
- Scalability: Tested up to 16,384 concurrent threads
- Memory efficiency: ~95% utilization under typical workloads

Note: Performance varies based on GPU architecture and workload characteristics.

## Known Limitations

- **Remaining Race Conditions**: Data corruption still detected under extreme stress testing
- **Multi-Element Masks**: Additional synchronization may be needed for >64 objects per slab
- **No Defragmentation**: Fragmented slabs remain allocated until completely empty
- **Fixed Arena Size**: Arena capacity set at compile time via template parameters

## Future Work

- Complete elimination of remaining race conditions
- Performance optimization once thread safety is fully validated
- Dynamic arena resizing
- Memory defragmentation support
- Formal verification of concurrent correctness

## Research Context

This allocator was developed as part of undergraduate research into GPU memory management challenges. The project explores fundamental questions about achieving thread safety at GPU scales, 
where traditional concurrent programming approaches must be carefully adapted for massively parallel environments.

See the accompanying [research paper] (https://redhawks-my.sharepoint.com/:w:/r/personal/hderege_seattleu_edu/_layouts/15/Doc.aspx?sourcedoc=%7B2D7765F6-A8D2-4585-949A-FAC93D915430%7D&file=hderege%20Research%20Paper.docx&action=default&mobileredirect=true&DefaultItemOpen=1&wdOrigin=MARKETING.WORD.SIGNIN%2CAPPHOME-WEB.JUMPBACKIN&wdPreviousSession=3d0e52e9-9317-4437-b62e-f55cfddd4d57&wdPreviousSessionSrc=AppHomeWeb&ct=1757695043269) for detailed analysis of race conditions, synchronization strategies, and experimental methodology.

## Dependencies

### Custom Atomic Library

The project uses a custom atomic operations wrapper (`intr/mod.h`) that provides:
- Cross-platform atomic intrinsics (CUDA and GCC)
- Consistent memory ordering semantics
- System-wide atomic operations for GPU global memory

## Configuration

### Template Parameters

Customize allocator behavior via template parameters:

```cpp
template <
    typename ARENA_SIZE,              // Total arena size (e.g., Size<1024*1024>)
    template<typename> typename SLAB_PROXY_TYPE,  // Proxy implementation
    typename SLAB_TYPE,               // Slab structure (e.g., Slab<4096>)
    typename SLAB_ADDR_TYPE           // Address type for slab indices
>
class SlabArena;
```

Example configurations:
```cpp
// 4KB slabs, 1MB arena
typedef SlabArena<Size<1024*1024>, defaultSlabProxy, Slab<4096>> TestSlabArena;

// 64KB slabs, 4MB arena
typedef SlabArena<Size<4*1024*1024>, defaultSlabProxy, Slab<65536>> LargeSlabArena;
```


## Acknowledgments

Research conducted at Seattle University under the supervision of Dr. Cuneo.

## Contact

Hirna Derege - [hirnadereg@gmail.com](mailto:hirnadereg@gmail.com)

---

*This is an active research project. Thread safety under extreme concurrency remains an ongoing challenge.*
