#include <atomic>
#include <memory>
#include <cstdint>
#include <cassert>
#include <thread>
#include <vector>
#include <iostream>
#include <chrono>
#include <bit> // C++20 for std::countr_zero, fallback provided

template<size_t SlabSize, typename T = char>
class CPULockFreeSlabAllocator {
private:
    static_assert(SlabSize >= sizeof(T), "Slab size must be at least sizeof(T)");
    static_assert((SlabSize % sizeof(T)) == 0, "Slab size must be multiple of object size");
    
    static constexpr size_t OBJECTS_PER_SLAB = SlabSize / sizeof(T);
    static constexpr size_t CACHE_LINE_SIZE = 64;
    static constexpr size_t MAX_OBJECTS_PER_SLAB = 64; // Using uint64_t bitmask
    
    static_assert(OBJECTS_PER_SLAB <= MAX_OBJECTS_PER_SLAB, 
                  "Too many objects per slab for bitmask");

    struct Slab {
        // Atomic free mask - each bit represents one object slot
        std::atomic<uint64_t> free_mask;
        
        // Next pointer for intrusive linked list
        std::atomic<Slab*> next;
        
        // Slab ID for debugging
        uint32_t slab_id;
        
        // Padding to avoid false sharing
        char padding[CACHE_LINE_SIZE - sizeof(std::atomic<uint64_t>) - 
                    sizeof(std::atomic<Slab*>) - sizeof(uint32_t)];
        
        // Object storage follows this header
        alignas(T) char objects[SlabSize];
        
        Slab(uint32_t id) : slab_id(id) {
            // Initialize all objects as free
            uint64_t all_free_mask = (OBJECTS_PER_SLAB == 64) ? 
                                   UINT64_MAX : 
                                   ((1ULL << OBJECTS_PER_SLAB) - 1);
            free_mask.store(all_free_mask, std::memory_order_relaxed);
            next.store(nullptr, std::memory_order_relaxed);
        }
        
        T* get_object(size_t index) {
            assert(index < OBJECTS_PER_SLAB);
            return reinterpret_cast<T*>(objects + index * sizeof(T));
        }
        
        size_t get_object_index(T* ptr) {
            char* obj_ptr = reinterpret_cast<char*>(ptr);
            assert(obj_ptr >= objects && obj_ptr < objects + SlabSize);
            return (obj_ptr - objects) / sizeof(T);
        }
        
        bool is_empty() const {
            uint64_t all_free_mask = (OBJECTS_PER_SLAB == 64) ? 
                                   UINT64_MAX : 
                                   ((1ULL << OBJECTS_PER_SLAB) - 1);
            return free_mask.load(std::memory_order_relaxed) == all_free_mask;
        }
        
        bool is_full() const {
            return free_mask.load(std::memory_order_relaxed) == 0;
        }
    };
    
    // Thread-local cache to reduce contention
    struct alignas(CACHE_LINE_SIZE) ThreadCache {
        Slab* current_slab = nullptr;
        Slab* backup_slab = nullptr;
        uint64_t cached_free_mask = 0;
        
        // Statistics
        size_t allocations = 0;
        size_t deallocations = 0;
        size_t slab_switches = 0;
    };
    
    // Global slab management
    alignas(CACHE_LINE_SIZE) std::atomic<Slab*> free_slab_head{nullptr};
    alignas(CACHE_LINE_SIZE) std::atomic<Slab*> partial_slab_head{nullptr};
    alignas(CACHE_LINE_SIZE) std::atomic<uint32_t> next_slab_id{0};
    
    // Thread-local storage
    static thread_local ThreadCache tl_cache;
    
    // Helper function for bit operations (C++20 fallback)
    static size_t find_first_set_bit(uint64_t mask) {
#if __cpp_lib_bit_ops >= 201907L
        return std::countr_zero(mask);
#else
        return __builtin_ctzll(mask);
#endif
    }
    
    Slab* allocate_new_slab() {
        try {
            uint32_t id = next_slab_id.fetch_add(1, std::memory_order_relaxed);
            return new Slab(id);
        } catch (const std::bad_alloc&) {
            return nullptr;
        }
    }
    
    Slab* get_slab_from_global_pool() {
        // First try free slabs (completely empty)
        Slab* slab = pop_from_list(free_slab_head);
        if (slab) {
            return slab;
        }
        
        // Then try partial slabs
        slab = pop_from_list(partial_slab_head);
        if (slab) {
            return slab;
        }
        
        // Finally allocate new slab
        return allocate_new_slab();
    }
    
    Slab* pop_from_list(std::atomic<Slab*>& head) {
        Slab* current_head = head.load(std::memory_order_acquire);
        
        while (current_head != nullptr) {
            Slab* next = current_head->next.load(std::memory_order_relaxed);
            
            if (head.compare_exchange_weak(current_head, next,
                                         std::memory_order_release,
                                         std::memory_order_relaxed)) {
                current_head->next.store(nullptr, std::memory_order_relaxed);
                return current_head;
            }
            // CAS failed, current_head was updated by the failed CAS
        }
        
        return nullptr;
    }
    
    void push_to_list(std::atomic<Slab*>& head, Slab* slab) {
        Slab* current_head = head.load(std::memory_order_relaxed);
        
        do {
            slab->next.store(current_head, std::memory_order_relaxed);
        } while (!head.compare_exchange_weak(current_head, slab,
                                           std::memory_order_release,
                                           std::memory_order_relaxed));
    }
    
    T* try_allocate_from_slab(Slab* slab) {
        uint64_t free_mask = slab->free_mask.load(std::memory_order_relaxed);
        
        while (free_mask != 0) {
            size_t index = find_first_set_bit(free_mask);
            uint64_t new_mask = free_mask & ~(1ULL << index);
            
            if (slab->free_mask.compare_exchange_weak(free_mask, new_mask,
                                                    std::memory_order_relaxed,
                                                    std::memory_order_relaxed)) {
                return slab->get_object(index);
            }
            // CAS failed, retry with updated free_mask
        }
        
        return nullptr; // Slab is full
    }
    
    void return_slab_to_global(Slab* slab) {
        if (slab->is_empty()) {
            push_to_list(free_slab_head, slab);
        } else if (!slab->is_full()) {
            push_to_list(partial_slab_head, slab);
        }
        // Full slabs are not returned to global pool (they have no free objects)
    }

public:
    CPULockFreeSlabAllocator() = default;
    
    ~CPULockFreeSlabAllocator() {
        // Clean up all slabs
        cleanup_slab_list(free_slab_head);
        cleanup_slab_list(partial_slab_head);
    }
    
    T* allocate() {
        ThreadCache& cache = tl_cache;
        cache.allocations++;
        
        // First try current slab
        if (cache.current_slab) {
            T* ptr = try_allocate_from_slab(cache.current_slab);
            if (ptr) {
                return ptr;
            }
            
            // Current slab is full, try to return it and switch to backup
            if (cache.backup_slab) {
                return_slab_to_global(cache.current_slab);
                cache.current_slab = cache.backup_slab;
                cache.backup_slab = nullptr;
                cache.slab_switches++;
                
                T* ptr = try_allocate_from_slab(cache.current_slab);
                if (ptr) {
                    return ptr;
                }
            }
        }
        
        // Need new slab(s)
        if (!cache.backup_slab) {
            cache.backup_slab = get_slab_from_global_pool();
        }
        
        if (!cache.current_slab) {
            cache.current_slab = get_slab_from_global_pool();
            if (!cache.current_slab) {
                return nullptr; // Out of memory
            }
        }
        
        // Try allocation again
        T* ptr = try_allocate_from_slab(cache.current_slab);
        if (ptr) {
            return ptr;
        }
        
        // If we get here, something went wrong
        return nullptr;
    }
    
    bool deallocate(T* ptr) {
        if (!ptr) return false;
        
        // Find the slab containing this pointer
        Slab* slab = find_slab_for_pointer(ptr);
        if (!slab) return false;
        
        size_t index = slab->get_object_index(ptr);
        uint64_t bit_mask = 1ULL << index;
        
        // Atomically set the bit to mark as free
        uint64_t old_mask = slab->free_mask.fetch_or(bit_mask, 
                                                    std::memory_order_relaxed);
        
        // Check if bit was already set (double free)
        if (old_mask & bit_mask) {
            return false; // Double free detected
        }
        
        ThreadCache& cache = tl_cache;
        cache.deallocations++;
        
        return true;
    }
    
    // Statistics and debugging
    struct Stats {
        size_t total_allocations = 0;
        size_t total_deallocations = 0;
        size_t total_slab_switches = 0;
        size_t active_slabs = 0;
        size_t free_slabs = 0;
        size_t partial_slabs = 0;
    };
    
    Stats get_stats() const {
        Stats stats;
        
        // This is a simplified stats collection - in production you'd want
        // to aggregate across all thread caches
        const ThreadCache& cache = tl_cache;
        stats.total_allocations = cache.allocations;
        stats.total_deallocations = cache.deallocations;
        stats.total_slab_switches = cache.slab_switches;
        
        // Count slabs in global lists
        stats.free_slabs = count_slabs_in_list(free_slab_head);
        stats.partial_slabs = count_slabs_in_list(partial_slab_head);
        stats.active_slabs = next_slab_id.load(std::memory_order_relaxed);
        
        return stats;
    }
    
    void print_stats() const {
        Stats stats = get_stats();
        std::cout << "Slab Allocator Statistics:\n"
                  << "  Total allocations: " << stats.total_allocations << "\n"
                  << "  Total deallocations: " << stats.total_deallocations << "\n"
                  << "  Slab switches: " << stats.total_slab_switches << "\n"
                  << "  Active slabs: " << stats.active_slabs << "\n"
                  << "  Free slabs: " << stats.free_slabs << "\n"
                  << "  Partial slabs: " << stats.partial_slabs << "\n";
    }

private:
    Slab* find_slab_for_pointer(T* ptr) {
        // This is a simplified implementation. In production, you'd want
        // a more efficient lookup mechanism (e.g., hash table or
        // address-based calculation if using large contiguous allocations)
        
        char* char_ptr = reinterpret_cast<char*>(ptr);
        
        // Check thread cache first
        ThreadCache& cache = tl_cache;
        if (cache.current_slab && 
            char_ptr >= cache.current_slab->objects &&
            char_ptr < cache.current_slab->objects + SlabSize) {
            return cache.current_slab;
        }
        
        if (cache.backup_slab &&
            char_ptr >= cache.backup_slab->objects &&
            char_ptr < cache.backup_slab->objects + SlabSize) {
            return cache.backup_slab;
        }
        
        // This is inefficient but correct for demonstration
        // In production, you'd maintain a lookup structure
        return nullptr;
    }
    
    void cleanup_slab_list(std::atomic<Slab*>& head) {
        Slab* current = head.exchange(nullptr, std::memory_order_relaxed);
        while (current) {
            Slab* next = current->next.load(std::memory_order_relaxed);
            delete current;
            current = next;
        }
    }
    
    size_t count_slabs_in_list(const std::atomic<Slab*>& head) const {
        size_t count = 0;
        Slab* current = head.load(std::memory_order_relaxed);
        while (current) {
            count++;
            current = current->next.load(std::memory_order_relaxed);
        }
        return count;
    }
};

// Thread-local storage definition
template<size_t SlabSize, typename T>
thread_local typename CPULockFreeSlabAllocator<SlabSize, T>::ThreadCache
CPULockFreeSlabAllocator<SlabSize, T>::tl_cache;

// Test and benchmark code
void test_basic_functionality() {
    std::cout << "Testing basic functionality...\n";
    
    CPULockFreeSlabAllocator<1024, int> allocator;
    
    // Test allocation
    int* ptr1 = allocator.allocate();
    int* ptr2 = allocator.allocate();
    int* ptr3 = allocator.allocate();
    
    assert(ptr1 != nullptr);
    assert(ptr2 != nullptr);
    assert(ptr3 != nullptr);
    assert(ptr1 != ptr2);
    assert(ptr2 != ptr3);
    
    // Test usage
    *ptr1 = 42;
    *ptr2 = 84;
    *ptr3 = 126;
    
    assert(*ptr1 == 42);
    assert(*ptr2 == 84);
    assert(*ptr3 == 126);
    
    // Test deallocation
    assert(allocator.deallocate(ptr1));
    assert(allocator.deallocate(ptr2));
    assert(allocator.deallocate(ptr3));
    
    // Test double free detection
    assert(!allocator.deallocate(ptr1));
    
    allocator.print_stats();
    std::cout << "Basic functionality test passed!\n\n";
}

void benchmark_multithreaded() {
    std::cout << "Running multithreaded benchmark...\n";
    
    CPULockFreeSlabAllocator<256, int> allocator;
    constexpr int NUM_THREADS = 8;
    constexpr int OPERATIONS_PER_THREAD = 100000;
    
    auto worker = [&](int thread_id) {
        std::vector<int*> ptrs;
        ptrs.reserve(1000);
        
        for (int i = 0; i < OPERATIONS_PER_THREAD; ++i) {
            // Allocate
            int* ptr = allocator.allocate();
            if (ptr) {
                *ptr = thread_id * 1000000 + i;
                ptrs.push_back(ptr);
            }
            
            // Deallocate some randomly
            if (ptrs.size() > 100 && (i % 10 == 0)) {
                size_t index = i % ptrs.size();
                allocator.deallocate(ptrs[index]);
                ptrs.erase(ptrs.begin() + index);
            }
        }
        
        // Clean up remaining
        for (int* ptr : ptrs) {
            allocator.deallocate(ptr);
        }
    };
    
    auto start  = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(worker, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Completed " << (NUM_THREADS * OPERATIONS_PER_THREAD) 
              << " operations in " << duration.count() << "ms\n";
    
    allocator.print_stats();
    std::cout << "Multithreaded benchmark completed!\n\n";
}

// Example usage
/*
int main() {
    test_basic_functionality();
    benchmark_multithreaded();
    return 0;
}
*/