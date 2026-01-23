#include <iostream>
#include <vector>
#include <chrono>
#include <atomic>
#include <bit>

// Mock implementation of intr:: namespace for testing
namespace intr {
    namespace atomic {
        template<typename T>
        T CAS_system(T* ptr, T expected, T desired) {
            T current = *ptr;
            if (current == expected) {
                *ptr = desired;
            }
            return current;
        }
        
        template<typename T>
        T exch_system(T* ptr, T desired) {
            T old = *ptr;
            *ptr = desired;
            return old;
        }
        
        template<typename T>
        T add_system(T* ptr, T value) {
            T old = *ptr;
            *ptr += value;
            return old;
        }
        
        template<typename T>
        T or_system(T* ptr, T value) {
            T old = *ptr;
            *ptr |= value;
            return old;
        }
        
        template<typename T>
        T and_system(T* ptr, T value) {
            T old = *ptr;
            *ptr &= value;
            return old;
        }
    }
    
    namespace bitwise {
        template<typename T>
        int population_count(T x) {
            return __builtin_popcountll(static_cast<unsigned long long>(x));
        }
        
        template<typename T>
        int first_set(T x) {
            if (x == 0) return sizeof(T) * 8; // Return bit width if no bits set
            return __builtin_ctzll(static_cast<unsigned long long>(x));
        }
    }
}

#include "allocator.h" // Your enhanced allocator header

// Enhanced test function with better error reporting
void testBasicAllocation() {
    std::cout << "=== Basic Allocation Test ===" << std::endl;
    
    // Create arena and allocator
    TestSlabArena arena;
    TestAllocator allocator(arena, 64); // 64-byte objects
    
    std::cout << "Arena created, slab count: " << TestSlabArena::SLAB_COUNT << std::endl;
    std::cout << "Slab size: " << TestSlabArena::slabType::SIZE << " bytes" << std::endl;
    
    // Test allocation count tracking
    size_t initialCount = allocator.getAllocatedCount();
    std::cout << "Initial allocated count: " << initialCount << std::endl;
    
    // Test allocation
    void* ptr1 = allocator.alloc();
    if (ptr1) {
        std::cout << "\nFirst allocation successful: " << ptr1 << std::endl;
        std::cout << "Allocated count after first alloc: " << allocator.getAllocatedCount() << std::endl;
    } else {
        std::cout << "\n:( - First allocation failed!" << std::endl;
        return;
    }
    
    void* ptr2 = allocator.alloc();
    if (ptr2) {
        std::cout << "Second allocation successful: " << ptr2 << std::endl;
        std::cout << "Allocated count after second alloc: " << allocator.getAllocatedCount() << std::endl;
    } else {
        std::cout << ":( - Second allocation failed!" << std::endl;
    }
    
    // Test that pointers are different and valid
    if (ptr1 != ptr2) {
        std::cout << "\nAllocated different addresses" << std::endl;
    } else {
        std::cout << "\n:( - Same address allocated twice!" << std::endl;
    }
    
    // Test pointer validation
    if (allocator.isValidPtr(ptr1)) {
        std::cout << "ptr1 validates as valid pointer" << std::endl;
    } else {
        std::cout << ":( - ptr1 failed validation!" << std::endl;
    }
    
    // Test writing to allocated memory
    if (ptr1) {
        char* charPtr = static_cast<char*>(ptr1);
        for (int i = 0; i < 64; i++) {
            charPtr[i] = (char)(i % 256);
        }
        std::cout << "Successfully wrote to allocated memory" << std::endl;
        
        // Verify the data
        bool dataOk = true;
        for (int i = 0; i < 64; i++) {
            if (charPtr[i] != (char)(i % 256)) {
                dataOk = false;
                std::cout << "Data mismatch at byte " << i << ": expected " 
                         << (i % 256) << ", got " << (int)charPtr[i] << std::endl;
                break;
            }
        }
        if (dataOk) {
            std::cout << "Data integrity verified" << std::endl;
        } else {
            std::cout << "\n:( - Data corruption detected!" << std::endl;
        }
    }
    
    // Test freeing
    if (allocator.free(ptr1)) {
        std::cout << "\nptr1 freed successfully..." << std::endl;
        std::cout << "Allocated count after first free: " << allocator.getAllocatedCount() << std::endl;
    } else {
        std::cout << "\n:( - First free failed!" << std::endl;
    }
    
    if (allocator.free(ptr2)) {
        std::cout << "ptr2 freed successfully..." << std::endl;
        std::cout << "Allocated count after second free: " << allocator.getAllocatedCount() << std::endl;
    } else {
        std::cout << "\n:( - Second free failed!" << std::endl;
    }
    
    std::cout << std::endl;
}

// Updated zero-size test
bool testZeroSizeAllocation(){
    std::cout << "=== Zero Size Allocation Test ===" << std::endl;

    TestSlabArena arena;
    TestAllocator allocator(arena, 0); // This might not work as expected

    void* ptr = allocator.alloc();
    std::cout << "Zero-size allocation returned: " << ptr << std::endl;

    if (ptr == nullptr) {
        std::cout << "Zero-size allocation returned null (expected behavior)" << std::endl;
        return true;
    }

    bool valid = allocator.isValidPtr(ptr);
    if(valid){
        std::cout << "Zero-sized allocation tracked successfully!" << std::endl;
        bool freed = allocator.free(ptr);
        if(freed)
            std::cout << "Zero-size allocation freed successfully" << std::endl;
        else
            std::cout << ":( - Zero-size free failed!" << std::endl;
    } else {
        std::cout << ":( - Zero-sized allocation invalid but not null!!" << std::endl;
        return false;
    }

    return true;
}

void testMultipleAllocations() {
    std::cout << "\n=== Multiple Allocations Test ===" << std::endl;
    
    TestSlabArena arena;
    TestAllocator allocator(arena, 32); // 32-byte objects
    
    const int numAllocs = 100;
    std::vector<void*> ptrs;
    
    // Show initial state
    std::cout << "Initial allocated count: " << allocator.getAllocatedCount() << std::endl;
    
    // Allocate multiple objects
    for (int i = 0; i < numAllocs; i++) {
        void* ptr = allocator.alloc();
        if (ptr) {
            ptrs.push_back(ptr);
        } else {
            std::cout << "Allocation failed at iteration " << i << std::endl;
            std::cout << "Allocated count at failure: " << allocator.getAllocatedCount() << std::endl;
            break;
        }
        
        // Show progress every 10 allocations
        if ((i + 1) % 10 == 0) {
            std::cout << "Allocated " << (i + 1) << " objects, count: " 
                     << allocator.getAllocatedCount() << std::endl;
        }
    }
    
    std::cout << "\nSuccessfully allocated " << ptrs.size() << " objects" << std::endl;
    std::cout << "Final allocated count: " << allocator.getAllocatedCount() << std::endl;
    
    // Verify all pointers are unique
    bool allUnique = true;
    for (size_t i = 0; i < ptrs.size() && allUnique; i++) {
        for (size_t j = i + 1; j < ptrs.size(); j++) {
            if (ptrs[i] == ptrs[j]) {
                allUnique = false;
                std::cout << ":( - Duplicate pointer found at indices " << i << " and " << j << std::endl;
                break;
            }
        }
    }
    
    if (allUnique) {
        std::cout << "All allocated pointers are unique" << std::endl;
    }
    
    // Free all objects
    int freedCount = 0;
    for (size_t i = 0; i < ptrs.size(); i++) {
        if (allocator.free(ptrs[i])) {
            freedCount++;
        } else {
            std::cout << ":( - Free failed for pointer " << i << std::endl;
        }
        
        // Show progress every 10 frees
        if ((i + 1) % 10 == 0) {
            std::cout << "Freed " << (i + 1) << " objects, count: " 
                     << allocator.getAllocatedCount() << std::endl;
        }
    }
    
    std::cout << "Successfully freed " << freedCount << "/" << ptrs.size() << " objects" << std::endl;
    std::cout << "Final allocated count: " << allocator.getAllocatedCount() << std::endl;
    std::cout << std::endl;
}

void testDifferentSizes() {
    std::cout << "=== Different Size Test ===" << std::endl;
    
    TestSlabArena arena;
    
    std::vector<size_t> sizes = {1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    
    for (size_t size : sizes) {
        TestAllocator allocator(arena, size);
        void* ptr = allocator.alloc();
        
        if (ptr) {
            std::cout << "Size " << size << " bytes: allocation successful" << std::endl;
            
            // Validate the pointer
            if (!allocator.isValidPtr(ptr)) {
                std::cout << ":( - Size " << size << " bytes: pointer validation failed!" << std::endl;
                continue;
            }
            
            // Test writing (be careful with size)
            char* charPtr = static_cast<char*>(ptr);
            for (size_t i = 0; i < size && i < 1024; i++) { // Cap at 1024 for safety
                charPtr[i] = (char)(i % 256);
            }
            
            if (!allocator.free(ptr)) {
                std::cout << ":( - Size " << size << " bytes: free failed!" << std::endl;
            }
        } else {
            std::cout << ":( - Size " << size << " bytes: allocation failed" << std::endl;
        }
    }
    
    std::cout << std::endl;
}

void testSlabReuse() {
    std::cout << "=== Slab Reuse Test ===" << std::endl;
    
    TestSlabArena arena;
    TestAllocator allocator(arena, 64);
    
    // Get reuse statistics
    size_t activeSlabs, reusableSlabs, totalObjects;
    allocator.getReuseStats(activeSlabs, reusableSlabs, totalObjects);
    std::cout << "Initial state - Active: " << activeSlabs 
              << ", Reusable: " << reusableSlabs 
              << ", Objects: " << totalObjects << std::endl;
    
    // Allocate some objects
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; i++) {
        void* ptr = allocator.alloc();
        if (ptr) ptrs.push_back(ptr);
    }
    
    allocator.getReuseStats(activeSlabs, reusableSlabs, totalObjects);
    std::cout << "After 10 allocs - Active: " << activeSlabs 
              << ", Reusable: " << reusableSlabs 
              << ", Objects: " << totalObjects << std::endl;
    
    // Free all objects (should create reusable slabs)
    for (void* ptr : ptrs) {
        allocator.free(ptr);
    }
    
    allocator.getReuseStats(activeSlabs, reusableSlabs, totalObjects);
    std::cout << "After freeing all - Active: " << activeSlabs 
              << ", Reusable: " << reusableSlabs 
              << ", Objects: " << totalObjects << std::endl;
    
    // Allocate again (should reuse slabs)
    for (int i = 0; i < 5; i++) {
        void* ptr = allocator.alloc();
        if (ptr) ptrs.push_back(ptr);
    }
    
    allocator.getReuseStats(activeSlabs, reusableSlabs, totalObjects);
    std::cout << "After reuse allocs - Active: " << activeSlabs 
              << ", Reusable: " << reusableSlabs 
              << ", Objects: " << totalObjects << std::endl;
    
    std::cout << std::endl;
}

void performanceTest() {
    std::cout << "=== Performance Test ===" << std::endl;
    
    TestSlabArena arena;
    TestAllocator allocator(arena, 64);
    
    const int iterations = 10000;
    std::vector<void*> ptrs;
    ptrs.reserve(iterations);
    
    // Time allocations
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        void* ptr = allocator.alloc();
        if (ptr) {
            ptrs.push_back(ptr);
        } else {
            std::cout << "Allocation failed at iteration " << i << std::endl;
            break;
        }
    }
    
    auto mid = std::chrono::high_resolution_clock::now();
    
    // Time deallocations
    for (void* ptr : ptrs) {
        allocator.free(ptr);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    auto allocTime = std::chrono::duration_cast<std::chrono::microseconds>(mid - start);
    auto freeTime = std::chrono::duration_cast<std::chrono::microseconds>(end - mid);
    
    std::cout << "Allocated " << ptrs.size() << " objects in " << allocTime.count() << " μs" << std::endl;
    std::cout << "Freed " << ptrs.size() << " objects in " << freeTime.count() << " μs" << std::endl;
    
    if (!ptrs.empty()) {
        std::cout << "\nAverage allocation time: " << (double)allocTime.count() / ptrs.size() << " μs" << std::endl;
        std::cout << "Average free time: " << (double)freeTime.count() / ptrs.size() << " μs" << std::endl;
    }
    
    std::cout << std::endl;
}

int main() {
    std::cout << "Enhanced Slab Allocator Test" << std::endl;
    std::cout << "============================" << std::endl << std::endl;
    
    try {
        testBasicAllocation();
        testZeroSizeAllocation();
        testMultipleAllocations();
        testDifferentSizes();
        testSlabReuse();  // New test for reuse functionality
        performanceTest();
        
        std::cout << "All tests completed! Exiting..." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception caught!" << std::endl;
        return 1;
    }
    
    return 0;
}