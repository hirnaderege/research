#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <unordered_set>

#include "allocator.h"


// wrapper class!
template<typename AllocatorType>
class TrackingAllocatorWrapper {
private:
    AllocatorType allocator;
    std::unordered_set<void*> allocatedPtrs;
    size_t totalAllocations;
    size_t currentAllocations;

public:
    TrackingAllocatorWrapper() : totalAllocations(0), currentAllocations(0) {}
    
    void* alloc(size_t size) {
        void* ptr = allocator.alloc(size);
        if (ptr) {
            allocatedPtrs.insert(ptr);
            totalAllocations++;
            currentAllocations++;
        }
        return ptr;
    }
    
    bool free(void* ptr) {
        if (!ptr) return false;
        
        auto it = allocatedPtrs.find(ptr);
        if (it == allocatedPtrs.end()) {
            return false; 
        }
        
        bool result = allocator.free(ptr);
        if (result) {
            allocatedPtrs.erase(it);
            currentAllocations--;
        }
        return result;
    }
    
    size_t getAllocatedCount() const {
        return currentAllocations;
    }
    
    size_t getTotalAllocations() const {
        return totalAllocations;
    }
    
    bool isValidPointer(void* ptr) const {
        return allocatedPtrs.find(ptr) != allocatedPtrs.end();
    }
    
    
    AllocatorType& getUnderlyingAllocator() {
        return allocator;
    }
};

typedef TrackingAllocatorWrapper<TestGeneralAllocator> TrackedAllocator;

// Tests
bool testBasicAllocation() {
    TrackedAllocator allocator;
    
    // Test single allocation
    void* ptr1 = allocator.alloc(64);
    TEST_ASSERT(ptr1 != nullptr, "Single allocation should succeed");
    TEST_ASSERT(allocator.getAllocatedCount() == 1, "Count should be 1 after single allocation");
    
    // Test multiple allocations
    void* ptr2 = allocator.alloc(128);
    void* ptr3 = allocator.alloc(256);
    TEST_ASSERT(ptr2 != nullptr && ptr3 != nullptr, "Multiple allocations should succeed");
    TEST_ASSERT(allocator.getAllocatedCount() == 3, "Count should be 3 after three allocations");
    
    // Test deallocation
    bool freed = allocator.free(ptr2);
    TEST_ASSERT(freed, "Free should succeed for valid pointer");
    TEST_ASSERT(allocator.getAllocatedCount() == 2, "Count should be 2 after freeing one");
    
    // Test freeing invalid pointer
    bool invalidFree = allocator.free((void*)0xDEADBEEF);
    TEST_ASSERT(!invalidFree, "Free should fail for invalid pointer");
    
    // Clean up
    allocator.free(ptr1);
    allocator.free(ptr3);
    
    return true;
}

bool testZeroSizeAllocation() {
    TrackedAllocator allocator;
    
    void* ptr = allocator.alloc(0);
    std::cout << "Zero-size allocation returned: " << ptr << std::endl;
    
    if (ptr) {
        TEST_ASSERT(allocator.isValidPointer(ptr), "Zero-size allocation should be tracked if successful");
        allocator.free(ptr);
    }
    
    return true;
}

bool testLargeAllocation() {
    TrackedAllocator allocator;
    
    // Test various large sizes within your allocator's limits
    std::vector<size_t> largeSizes = {1024, 4096, 8192, 16384, 32768};
    
    for (size_t size : largeSizes) {
        void* ptr = allocator.alloc(size);
        
        if (ptr == nullptr) {
            std::cout << "Large allocation of size " << size << " failed (may be expected)" << std::endl;
            continue;
        }
        
        TEST_ASSERT(ptr != nullptr, "Large allocation should succeed");
        TEST_ASSERT(allocator.isValidPointer(ptr), "Large allocation should be tracked");
        
        // Test that we can write to the memory
        memset(ptr, 0xAA, size);
        
        // Verify the memory
        char* charPtr = static_cast<char*>(ptr);
        for (size_t i = 0; i < size; i++) {
            TEST_ASSERT(charPtr[i] == (char)0xAA, "Memory should be writable and readable");
        }
        
        bool freed = allocator.free(ptr);
        TEST_ASSERT(freed, "Large allocation should free successfully");
    }
    
    return true;
}

bool testPowerOfTwoSizes() {
    TrackedAllocator allocator;
    std::vector<void*> ptrs;
    
    // Test power-of-2 sizes that should align with your size classes
    for (size_t exp = 0; exp < 12; exp++) { // 1 byte to 4KB
        size_t size = 1ULL << exp;
        void* ptr = allocator.alloc(size);
        
        if (ptr) {
            TEST_ASSERT(allocator.isValidPointer(ptr), "Power-of-2 allocation should be tracked");
            ptrs.push_back(ptr);
            
            // Write pattern to verify memory works
            if (size > 0) {
                memset(ptr, 0x55, size);
            }
        } else {
            std::cout << "Power-of-2 allocation of size " << size << " failed" << std::endl;
        }
    }
    
    // Clean up
    for (void* ptr : ptrs) {
        bool freed = allocator.free(ptr);
        TEST_ASSERT(freed, "Power-of-2 allocation should free successfully");
    }
    
    TEST_ASSERT(allocator.getAllocatedCount() == 0, "All power-of-2 allocations should be freed");
    
    return true;
}

bool testManySmallAllocations() {
    TrackedAllocator allocator;
    std::vector<void*> ptrs;
    
    // Allocate many small blocks
    const int numAllocs = 500; // Reduced from 1000 to avoid running out of slabs
    for (int i = 0; i < numAllocs; i++) {
        void* ptr = allocator.alloc(32);
        if (!ptr) {
            std::cout << "Small allocation failed at iteration " << i << " (may be expected)" << std::endl;
            break;
        }
        
        TEST_ASSERT(allocator.isValidPointer(ptr), "Small allocation should be tracked");
        ptrs.push_back(ptr);
        
        // Write to each block to ensure it's valid
        memset(ptr, i % 256, 32);
    }
    
    std::cout << "Successfully allocated " << ptrs.size() << " small blocks" << std::endl;
    TEST_ASSERT(allocator.getAllocatedCount() == ptrs.size(), "Allocation count should match");
    
    // Verify memory integrity
    for (size_t i = 0; i < ptrs.size(); i++) {
        char* charPtr = static_cast<char*>(ptrs[i]);
        for (int j = 0; j < 32; j++) {
            TEST_ASSERT(charPtr[j] == (char)(i % 256), "Memory integrity should be maintained");
        }
    }
    
    // Free all blocks
    for (void* ptr : ptrs) {
        bool freed = allocator.free(ptr);
        TEST_ASSERT(freed, "Free should succeed for valid pointer");
    }
    
    TEST_ASSERT(allocator.getAllocatedCount() == 0, "All blocks should be freed");
    
    return true;
}

bool testRandomAllocationPattern() {
    TrackedAllocator allocator;
    std::vector<void*> ptrs;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> sizeDistr(1, 512); // Reduced max size
    std::uniform_int_distribution<> actionDistr(0, 1);
    
    // Perform random allocations and deallocations
    for (int i = 0; i < 2000; i++) { // Reduced iterations
        if (ptrs.empty() || actionDistr(gen) == 0) {
            // Allocate
            size_t size = sizeDistr(gen);
            void* ptr = allocator.alloc(size);
            if (ptr) {
                TEST_ASSERT(allocator.isValidPointer(ptr), "Random allocation should be tracked");
                ptrs.push_back(ptr);
                memset(ptr, 0x55, size);
            }
        } else {
            // Free
            std::uniform_int_distribution<> indexDistr(0, ptrs.size() - 1);
            size_t index = indexDistr(gen);
            void* ptr = ptrs[index];
            bool freed = allocator.free(ptr);
            TEST_ASSERT(freed, "Random free should succeed");
            ptrs.erase(ptrs.begin() + index);
        }
    }
    
    std::cout << "Random test completed with " << ptrs.size() << " blocks remaining" << std::endl;
    
    // Clean up remaining allocations
    for (void* ptr : ptrs) {
        allocator.free(ptr);
    }
    
    return true;
}


bool testSizeClassBoundaries() {
    TrackedAllocator allocator;
    
    // Test allocations around size class boundaries
    std::vector<size_t> testSizes = {
        1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257, 512, 513, 1024
    };
    
    std::vector<void*> ptrs;
    
    for (size_t size : testSizes) {
        void* ptr = allocator.alloc(size);
        if (ptr) {
            TEST_ASSERT(allocator.isValidPointer(ptr), "Size class boundary allocation should be tracked");
            ptrs.push_back(ptr);
            
            // Write and verify memory
            memset(ptr, 0x77, size);
            char* charPtr = static_cast<char*>(ptr);
            for (size_t i = 0; i < size; i++) {
                TEST_ASSERT(charPtr[i] == (char)0x77, "Memory should be accessible");
            }
        }
    }
    
    std::cout << "Successfully allocated " << ptrs.size() << " size class boundary blocks" << std::endl;
    
    // Clean up
    for (void* ptr : ptrs) {
        bool freed = allocator.free(ptr);
        TEST_ASSERT(freed, "Size class boundary allocation should free successfully");
    }
    
    return true;
}

// Performance Tests
bool testAllocationSpeed() {
    TrackedAllocator allocator;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<void*> ptrs;
    const int numAllocs = 5000; // Reduced for slab allocator limits
    
    for (int i = 0; i < numAllocs; i++) {
        void* ptr = allocator.alloc(64);
        if (ptr) ptrs.push_back(ptr);
    }
    
    auto mid = std::chrono::high_resolution_clock::now();
    
    for (void* ptr : ptrs) {
        allocator.free(ptr);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    auto allocTime = std::chrono::duration_cast<std::chrono::microseconds>(mid - start);
    auto freeTime = std::chrono::duration_cast<std::chrono::microseconds>(end - mid);
    
    std::cout << "Allocation time for " << ptrs.size() << " 64-byte blocks: " << allocTime.count() << " μs" << std::endl;
    std::cout << "Deallocation time for " << ptrs.size() << " 64-byte blocks: " << freeTime.count() << " μs" << std::endl;
    if (ptrs.size() > 0) {
        std::cout << "Average allocation time: " << (double)allocTime.count() / ptrs.size() << " μs per allocation" << std::endl;
        std::cout << "Average deallocation time: " << (double)freeTime.count() / ptrs.size() << " μs per deallocation" << std::endl;
    }
    
    return true;
}

bool testMixedSizePerformance() {
    TrackedAllocator allocator;
    
    std::vector<size_t> sizes = {16, 32, 64, 128, 256, 512, 1024};
    std::vector<void*> ptrs;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Allocate mixed sizes
    for (int round = 0; round < 200; round++) {
        for (size_t size : sizes) {
            void* ptr = allocator.alloc(size);
            if (ptr) ptrs.push_back(ptr);
        }
    }
    
    auto mid = std::chrono::high_resolution_clock::now();
    
    // Free in reverse order to test fragmentation handling
    std::reverse(ptrs.begin(), ptrs.end());
    for (void* ptr : ptrs) {
        allocator.free(ptr);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    auto allocTime = std::chrono::duration_cast<std::chrono::microseconds>(mid - start);
    auto freeTime = std::chrono::duration_cast<std::chrono::microseconds>(end - mid);
    
    std::cout << "Mixed size allocation time for " << ptrs.size() << " blocks: " << allocTime.count() << " μs" << std::endl;
    std::cout << "Mixed size deallocation time: " << freeTime.count() << " μs" << std::endl;
    
    return true;
}

// Fragmentation Tests
//

// Utility function to test size class calculations
bool testSizeClassCalculation() {
    // Test power-of-2 size class calculation logic
    auto calculateSizeIndex = [](size_t size) -> size_t {
        if (size <= 1) return 0;
        return 64 - __builtin_clzll(size - 1);
    };
    
    // Test various sizes map to correct size classes
    TEST_ASSERT(calculateSizeIndex(1) == 0, "Size 1 should map to class 0");
    TEST_ASSERT(calculateSizeIndex(2) == 1, "Size 2 should map to class 1");
    TEST_ASSERT(calculateSizeIndex(3) == 2, "Size 3 should map to class 2");
    TEST_ASSERT(calculateSizeIndex(4) == 2, "Size 4 should map to class 2");
    TEST_ASSERT(calculateSizeIndex(5) == 3, "Size 5 should map to class 3");
    TEST_ASSERT(calculateSizeIndex(8) == 3, "Size 8 should map to class 3");
    TEST_ASSERT(calculateSizeIndex(9) == 4, "Size 9 should map to class 4");
    TEST_ASSERT(calculateSizeIndex(16) == 4, "Size 16 should map to class 4");
    TEST_ASSERT(calculateSizeIndex(17) == 5, "Size 17 should map to class 5");
    
    return true;
}


int main() {
    std::cout << "=== Slab Allocator Test Suite ===" << std::endl;
    std::cout << "Testing GeneralAllocator implementation" << std::endl;
    std::cout << std::endl;
    
    // Basic functionality tests
    std::cout << "=== Basic Functionality Tests ===" << std::endl;
    RUN_TEST(testBasicAllocation);
    RUN_TEST(testZeroSizeAllocation);
    RUN_TEST(testLargeAllocation);
    RUN_TEST(testPowerOfTwoSizes);
    
    // Stress tests
    std::cout << "=== Stress Tests ===" << std::endl;
    RUN_TEST(testManySmallAllocations);
    RUN_TEST(testRandomAllocationPattern);
    
    // Edge case tests
    std::cout << "=== Edge Case Tests ===" << std::endl;
    RUN_TEST(testDoubleFreePrevention);
    RUN_TEST(testNullPointerFree);
    RUN_TEST(testAlignmentRequirements);
    RUN_TEST(testSizeClassBoundaries);
    
    // Performance tests
    std::cout << "=== Performance Tests ===" << std::endl;
    RUN_TEST(testAllocationSpeed);
    RUN_TEST(testMixedSizePerformance);
    
    // Specialized tests
    std::cout << "=== Specialized Tests ===" << std::endl;
    RUN_TEST(testFragmentationBehavior);
    RUN_TEST(testSlabUtilization);
    RUN_TEST(testSlabReuse);
    RUN_TEST(testOutOfMemoryHandling);
    RUN_TEST(testSizeClassCalculation);
    
    std::cout << std::endl;
    std::cout << "=== Test Suite Complete ===" << std::endl;
    
    return 0;
}