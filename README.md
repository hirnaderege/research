# thread-safe GPU memory allocator 🧠

> a lock-free slab allocator for concurrent GPU workloads — no mutexes, no waiting in line

built as part of a faculty-led research project at Seattle University. the allocator hands out fixed-size memory slabs to thousands of concurrent GPU threads using atomic operations instead of locks, keeping `alloc` and `free` at **O(1)** even under heavy contention.

**[→ see it live in the trace visualizer](https://hirnaderege.github.io/portfolio/#/visualizer)** — every flash is a real event from an actual execution trace.

---

## the problem

GPU programs run thousands of threads simultaneously. when they all need memory at the same time, a naive allocator with a single global lock becomes a bottleneck — threads pile up waiting, and all that parallelism goes to waste.

the solution: **slab allocation** + **lock-free synchronization**. each thread grabs a pre-sized slab from a pool using atomic CAS (compare-and-swap) instead of a mutex. if a slab isn't available, the thread retries rather than blocking — keeping the GPU moving.

---

## API

```cpp
void* slab_alloc();   // O(1) lock-free allocation
void  slab_free(void* ptr);  // O(1) lock-free free
```

no `cudaMalloc` per-thread, no global mutex, no blocking.

---

## how it works

```
GPU kernel launch: N threads, each needs memory

for each thread:
    pick a slab index (thread ID % num_slabs)
    try atomic CAS to claim it
    if claimed  → allocate, use, free when done
    if contended → retry (logged as "retry" event)

result: allocation without a single lock
```

the `retry` event is the key metric — direct evidence of contention under load. the visualizer makes this visible across 128, 512, and 4,096 thread configurations.

---

## the visualizer

since the allocator runs on a GPU, you can't just add print statements. instead, i instrumented it to write a full execution trace — every `alloc`, `free`, and `retry` event, timestamped and tagged with thread ID and slab ID.

the visualizer replays that trace in a browser:

- **the die** — a grid of every memory slab, lit amber when allocated, pulsing red on contention
- **thread activity strip** — which threads are active vs retrying at each moment
- **three traces**: 128, 512, and 4,096 threads — watch contention grow with scale

```
trace256.json: 4,096 threads, 4,000,000+ real events
→ preprocessed to ~25k representative playback events
→ full stats (peak occupancy, total retries, contention rate)
   computed from every single real event before downsampling
```

nothing in the visualizer is simulated. the numbers are real.

---

## the stack

| piece | what it is |
|---|---|
| C++ / CUDA | allocator implementation |
| atomic CAS | lock-free thread synchronization |
| Python | trace preprocessing (4M events → browser-sized JSON) |
| HTML / Canvas API | visualizer rendering |
| `Float32Array`, `Uint8Array` | fast per-frame slab state tracking |
| `requestAnimationFrame` | smooth 60fps playback loop |

---

## files

```
allocator/          C++/CUDA allocator source
traces/             raw execution traces
visualizer/
  preprocess.py     shrinks raw traces for browser playback
  main.html         visualizer entry point
  engine.js         canvas rendering + playback logic
  data/             preprocessed trace JSONs
```

---

## things i learned

- why global locks hurt GPU throughput — the contention rate in trace256 tells that story
- atomic compare-and-swap as a building block for lock-free data structures
- how to instrument low-level C++/CUDA code without wrecking performance
- canvas rendering at 60fps with typed arrays (normal JS arrays are too slow)
- the difference between "downsampled for display" and "accurate statistics"

---

research project, Seattle University, 2025–2026 🎓
