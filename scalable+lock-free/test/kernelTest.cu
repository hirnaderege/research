#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <sstream>

#include "intr/mod.h"
#include "allocator.h"

void auto_throw(cudaError_t result) {
    if(result != cudaSuccess) {
        std::stringstream ss;
        ss << ":( - CUDA error: " << cudaGetErrorString(result) << std::endl;
        throw std::runtime_error(ss.str());
    }
}

// Test configuration struct
struct TestConfig {
    int threadsPerBlock;
    int numBlocks;
    int iterations;
    
    TestConfig() : threadsPerBlock(256), numBlocks(16), iterations(1000) {}
    
    void print() const {
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Threads per block: " << threadsPerBlock << std::endl;
        std::cout << "  Number of blocks: " << numBlocks << std::endl;
        std::cout << "  Total threads: " << (threadsPerBlock * numBlocks) << std::endl;
        std::cout << "  Iterations: " << iterations << std::endl << std::endl;
    }
};

// Helper function to get validated input
int getIntInput(const std::string& prompt, int minVal, int maxVal, int defaultVal) {
    std::cout << prompt << " [" << minVal << "-" << maxVal << "] (default: " << defaultVal << "): ";
    std::string input;
    std::getline(std::cin, input);
    
    if (input.empty()) {
        return defaultVal;
    }
    
    try {
        int value = std::stoi(input);
        if (value < minVal || value > maxVal) {
            std::cout << "Value out of range, using default: " << defaultVal << std::endl;
            return defaultVal;
        }
        return value;
    } catch (...) {
        std::cout << "Invalid input, using default: " << defaultVal << std::endl;
        return defaultVal;
    }
}

// Function to get test configuration from user
TestConfig getTestConfig(const cudaDeviceProp& prop) {
    TestConfig config;
    
    std::cout << "\n=== Test Configuration ===" << std::endl;
    std::cout << "Press Enter to use default values\n" << std::endl;
    
    config.threadsPerBlock = getIntInput(
        "Threads per block", 
        32, 
        prop.maxThreadsPerBlock, 
        256
    );
    
    // Validate threads per block is multiple of 32 (warp size)
    if (config.threadsPerBlock % 32 != 0) {
        config.threadsPerBlock = ((config.threadsPerBlock + 31) / 32) * 32;
        std::cout << "Adjusted to nearest multiple of 32: " << config.threadsPerBlock << std::endl;
    }
    
    config.numBlocks = getIntInput(
        "Number of blocks", 
        1, 
        prop.multiProcessorCount * 64,
        16
    );
    
    config.iterations = getIntInput(
        "Iterations per thread", 
        100, 
        10000, 
        1000
    );
    
    std::cout << std::endl;
    config.print();
    
    return config;
}

__device__ 
uint32_t simpleRand(uint32_t* seed){
    *seed = *seed * 1664525u + 1013904223u;
    return *seed;
}

class GPUTracker {
public:
    static const size_t MAX_TRACKED_OBJECTS = 1048576;
    uint32_t* trackingArena;
    uint32_t* stats;

    __device__ 
    bool recordAllocation(void* ptr, size_t size, uint32_t threadId, TestSlabArena* arena) {
        if (!ptr) 
            return false;

        size_t index = getIndexForPtr(ptr, arena);
        if (index >= MAX_TRACKED_OBJECTS){
            intr::atomic::add_system(&stats[2], 1u);
            return false;
        }
        
        uint32_t newValue = (static_cast<uint32_t>(size & 0xFFFF) << 16) | (threadId & 0xFFFF);
        uint32_t expected = 0;
        uint32_t old = intr::atomic::CAS_system(&trackingArena[index], expected, newValue);
        
        if (old == expected) {
            intr::atomic::add_system(&stats[0], 1u);
            return true;
        }
        
        intr::atomic::add_system(&stats[2], 1u);
        return false;
    }

    __device__ bool 
    recordFree(void* ptr, uint32_t threadId, TestSlabArena* arena) {
        if (!ptr) 
            return false;
        
        size_t index = getIndexForPtr(ptr, arena);
        if (index >= MAX_TRACKED_OBJECTS) 
            return false;
        
        uint32_t current = intr::atomic::load_relaxed(&trackingArena[index]);
        if ((current & 0xFFFF) != threadId || current == 0) {
            intr::atomic::add_system(&stats[2], 1u);
            return false;
        }
        
        uint32_t old = intr::atomic::CAS_system(&trackingArena[index], current, 0u);
        if (old == current) {
            intr::atomic::add_system(&stats[1], 1u);
            return true;
        }
        
        intr::atomic::add_system(&stats[2], 1u);
        return false;
    }

private:
    __device__ 
    size_t getIndexForPtr(void* ptr, TestSlabArena* arena) {
        if (!ptr) 
            return MAX_TRACKED_OBJECTS;
    
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        return (addr >> 3) % MAX_TRACKED_OBJECTS;      
    }
};

__global__ 
void allocatorTestKernel(TestSlabArena* arena, GPUTracker* tracker, int iterations, uint32_t* shouldStop) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t seed = tid + 12345u;
    
    void* localPtrs[64];
    size_t localSizes[64];
    uint32_t localIds[64];
    int localCount = 0;
    
    for(int i = 0; i < iterations && !(*shouldStop); i++) {
        uint32_t action = simpleRand(&seed) % 100;
        
        if(action < 70 || localCount == 0) {
            uint32_t sizeChoice = simpleRand(&seed) % 7;
            size_t objSize = 1 << (3 + sizeChoice);
            
            TestAllocator allocator(*arena, objSize);
            void* ptr = allocator.alloc();
            
            if(ptr && localCount < 64) {
                localPtrs[localCount] = ptr;
                localSizes[localCount] = objSize;
                localIds[localCount] = simpleRand(&seed) % 0xFFFFu;
                localCount++;
                
                tracker->recordAllocation(ptr, objSize, tid, arena);
                
                if(objSize >= 4) {
                    uint32_t* intPtr = static_cast<uint32_t*>(ptr);
                    uint32_t new_val = (tid << 16) | (localIds[localCount-1] & 0xFFFF);
                    intr::atomic::exch_system(intPtr, new_val);
                }
            }
        } else {
            if(localCount > 0) {
                uint32_t idx = simpleRand(&seed) % localCount;
                void* ptr = localPtrs[idx];
                size_t objSize = localSizes[idx];
                uint32_t vid = localIds[idx];
                
                __threadfence_system();
                
                if(objSize >= 4) {
                    uint32_t* intPtr = static_cast<uint32_t*>(ptr);
                    uint32_t expected = (tid << 16) | (vid & 0xFFFF);
                    uint32_t old_val = intr::atomic::CAS_system((unsigned int*)intPtr,(unsigned int)expected,0u);
                }
                
                TestAllocator allocator(*arena, objSize);
                if(allocator.free(ptr)) {
                    tracker->recordFree(ptr, tid, arena);
                }
                
                localPtrs[idx] = localPtrs[localCount-1];
                localSizes[idx] = localSizes[localCount-1];
                localIds[idx] = localIds[localCount-1];
                localCount--;
            }
        }
        
        if((simpleRand(&seed) % 100) == 0) {
            __syncthreads();
        }
    }
    
    for(int j = 0; j < localCount; j++) {
        TestAllocator allocator(*arena, localSizes[j]);
        allocator.free(localPtrs[j]); 
        tracker->recordFree(localPtrs[j], tid, arena); 
    }
}

void runGPUAllocatorTest(const TestConfig& config) {
    std::cout << "=== GPU Allocator Test ===" << std::endl;
    config.print();
    
    TestSlabArena* d_arena;
    GPUTracker* d_tracker;
    uint32_t* d_trackingArena;
    uint32_t* d_stats;
    uint32_t* d_shouldStop;
    
    cudaMalloc(&d_arena, sizeof(TestSlabArena));
    cudaMalloc(&d_tracker, sizeof(GPUTracker));
    cudaMalloc(&d_trackingArena, GPUTracker::MAX_TRACKED_OBJECTS * sizeof(uint32_t));
    cudaMalloc(&d_stats, 4 * sizeof(uint32_t));
    cudaMalloc(&d_shouldStop, sizeof(uint32_t));
    
    cudaMemset(d_trackingArena, 0, GPUTracker::MAX_TRACKED_OBJECTS * sizeof(uint32_t));
    cudaMemset(d_stats, 0, 4 * sizeof(uint32_t));
    
    uint32_t stopFlag = 0;
    cudaMemcpy(d_shouldStop, &stopFlag, sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    TestSlabArena* h_arena = new TestSlabArena();
    cudaMemcpy(d_arena, h_arena, sizeof(TestSlabArena), cudaMemcpyHostToDevice);
    delete h_arena;
    
    GPUTracker h_tracker;
    h_tracker.trackingArena = d_trackingArena;
    h_tracker.stats = d_stats;
    cudaMemcpy(d_tracker, &h_tracker, sizeof(GPUTracker), cudaMemcpyHostToDevice);
    
    std::cout << "Launching kernel..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    allocatorTestKernel<<<config.numBlocks, config.threadsPerBlock>>>(d_arena, d_tracker, config.iterations, d_shouldStop);
    
    cudaError_t result = cudaDeviceSynchronize();
    if(result != cudaSuccess) {
        std::cerr << ":( - CUDA error: " << cudaGetErrorString(result) << std::endl;
        return;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    uint32_t h_stats[4];
    cudaMemcpy(h_stats, d_stats, 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    uint32_t* h_trackingArena = new uint32_t[GPUTracker::MAX_TRACKED_OBJECTS];
    cudaMemcpy(h_trackingArena, d_trackingArena, 
               GPUTracker::MAX_TRACKED_OBJECTS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    uint32_t leaks = 0;
    for(size_t i = 0; i < GPUTracker::MAX_TRACKED_OBJECTS; i++) {
        if(h_trackingArena[i] != 0) leaks++;
    }
    
    float conversion = 1000000.0;
    
    std::cout << "\nGPU Results:" << std::endl;
    std::cout << "Duration: " << duration.count() << " μs" << std::endl;
    std::cout << "Threads: " << (config.numBlocks * config.threadsPerBlock) << std::endl;
    std::cout << "Allocs: " << h_stats[0] << std::endl;
    std::cout << "Frees: " << h_stats[1] << std::endl;
    std::cout << "Failures: " << h_stats[2] << std::endl;
    std::cout << "Throughput: " << ((h_stats[0] + h_stats[1]) * conversion) / duration.count() << " ops/sec" << std::endl;
    
    uint32_t totalOps = h_stats[0] + h_stats[1] + h_stats[2];
    if(totalOps > 0) {
        std::cout << "Success rate: " << (100.0 * (h_stats[0] + h_stats[1])) / totalOps << "%" << std::endl;
    }
    
    if(leaks == 0) {
        std::cout << ":) No leaks detected" << std::endl;
    } else {
        std::cout << ":( " << leaks << " leaks detected!" << std::endl;
    }
    
    delete[] h_trackingArena;
    cudaFree(d_arena);
    cudaFree(d_tracker);
    cudaFree(d_trackingArena);
    cudaFree(d_stats);
    cudaFree(d_shouldStop);
    
    std::cout << std::endl;
}

__global__ 
void stressTestKernel(TestSlabArena* arena, GPUTracker* tracker, int maxIterations, uint32_t* shouldStop) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t seed = tid + 54321u;
    
    void* localPtrs[32];
    size_t localSizes[32];
    int localCount = 0;
 
    for(int i = 0; i < maxIterations; i++) {
        if(i % 100 == 0 && *shouldStop) 
            break;
        
        uint32_t action = simpleRand(&seed) % 100;
        
        if(action < 80 || localCount == 0) {
            uint32_t sizeChoice = simpleRand(&seed) % 6;
            size_t objSize = 1 << (3 + sizeChoice);
            
            TestAllocator allocator(*arena, objSize);
            void* ptr = allocator.alloc();
            
            if(ptr && localCount < 32) {
                if(tracker->recordAllocation(ptr, objSize, tid, arena)) {
                    localPtrs[localCount] = ptr;
                    localSizes[localCount] = objSize;
                    localCount++;
                    
                    if(objSize >= 4) {
                        uint32_t* intPtr = static_cast<uint32_t*>(ptr);
                        *intPtr = (tid << 16) | (i & 0xFFFF);
                    }
                }
            }
        } else {
            if(localCount > 0) {
                uint32_t idx = simpleRand(&seed) % localCount;
                void* ptr = localPtrs[idx];
                size_t objSize = localSizes[idx];
                
                TestAllocator allocator(*arena, objSize);
                if(allocator.free(ptr)) {
                    tracker->recordFree(ptr, tid, arena);
                }
                
                localPtrs[idx] = localPtrs[localCount-1];
                localSizes[idx] = localSizes[localCount-1];
                localCount--;
            }
        }
        
        if((simpleRand(&seed) % 1000) == 0) {
            __syncthreads();
        }
    }
    
    for(int j = 0; j < localCount; j++) {
        TestAllocator allocator(*arena, localSizes[j]);
        if(allocator.free(localPtrs[j])) {
            tracker->recordFree(localPtrs[j], tid, arena);
        }
    }
}

void runGPUStressTest(const TestConfig& config) {
    std::cout << "=== GPU Stress Test ===" << std::endl;
    
    TestSlabArena* d_arena;
    GPUTracker* d_tracker;
    uint32_t* d_trackingArena;
    uint32_t* d_stats;
    uint32_t* d_shouldStop;
    
    cudaMalloc(&d_arena, sizeof(TestSlabArena));
    cudaMalloc(&d_tracker, sizeof(GPUTracker));
    cudaMalloc(&d_trackingArena, GPUTracker::MAX_TRACKED_OBJECTS * sizeof(uint32_t));
    cudaMalloc(&d_stats, 4 * sizeof(uint32_t));
    cudaMalloc(&d_shouldStop, sizeof(uint32_t));
    
    cudaMemset(d_trackingArena, 0, GPUTracker::MAX_TRACKED_OBJECTS * sizeof(uint32_t));
    cudaMemset(d_stats, 0, 4 * sizeof(uint32_t));
    
    uint32_t stopFlag = 0;
    cudaMemcpy(d_shouldStop, &stopFlag, sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    TestSlabArena* h_arena = new TestSlabArena();
    cudaMemcpy(d_arena, h_arena, sizeof(TestSlabArena), cudaMemcpyHostToDevice);
    delete h_arena;
    
    GPUTracker h_tracker;
    h_tracker.trackingArena = d_trackingArena;
    h_tracker.stats = d_stats;
    cudaMemcpy(d_tracker, &h_tracker, sizeof(GPUTracker), cudaMemcpyHostToDevice);
    
    // Use higher contention for stress test
    const int threadsPerBlock = 512;
    const int numBlocks = 32;
    const int maxIterations = config.iterations * 2;
    
    std::cout << "Launching " << (numBlocks * threadsPerBlock) << " threads for stress test..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    stressTestKernel<<<numBlocks, threadsPerBlock>>>(d_arena, d_tracker, maxIterations, d_shouldStop);
    
    auto sleepStart = std::chrono::high_resolution_clock::now();
    while(true) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - sleepStart);
        if(elapsed.count() >= 3) break;
    }    
    stopFlag = 1;
    auto_throw(cudaMemcpy(d_shouldStop, &stopFlag, sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    auto_throw(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    uint32_t h_stats[4];
    auto_throw(cudaMemcpy(h_stats, d_stats, 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    uint32_t* h_trackingArena = new uint32_t[GPUTracker::MAX_TRACKED_OBJECTS];
    auto_throw(cudaMemcpy(h_trackingArena, d_trackingArena, 
               GPUTracker::MAX_TRACKED_OBJECTS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    uint32_t leaks = 0;
    for(size_t i = 0; i < GPUTracker::MAX_TRACKED_OBJECTS; i++) {
        if(h_trackingArena[i] != 0) leaks++;
    }
    
    std::cout << "\nGPU Stress Results:" << std::endl;
    std::cout << "Duration: " << duration.count() << "ms" << std::endl;
    std::cout << "Total Threads: " << (numBlocks * threadsPerBlock) << std::endl;
    std::cout << "Allocs: " << h_stats[0] << std::endl;
    std::cout << "Frees: " << h_stats[1] << std::endl;
    std::cout << "Failures: " << h_stats[2] << std::endl;
    std::cout << "Leaks: " << leaks << std::endl;
    std::cout << "Throughput: " << ((h_stats[0] + h_stats[1]) * 1000) / duration.count() << " ops/sec" << std::endl;
    
    if(leaks == 0) {
        std::cout << ":) No leaks under extreme GPU contention" << std::endl;
    } else {
        std::cout << ":( Leaks detected under GPU stress!" << std::endl;
    }
    
    delete[] h_trackingArena;
    cudaFree(d_arena);
    cudaFree(d_tracker);
    cudaFree(d_trackingArena);
    cudaFree(d_stats);
    cudaFree(d_shouldStop);
    
    std::cout << std::endl;
}

__global__ 
void cudaMallocTestKernel(GPUTracker* tracker, int iterations, uint32_t* shouldStop) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t seed = tid + 12345u;
    
    void* localPtrs[64];
    size_t localSizes[64];
    uint32_t localIds[64];
    int localCount = 0;
    
    for(int i = 0; i < iterations && !(*shouldStop); i++) {
        uint32_t action = simpleRand(&seed) % 100;
        
        if(action < 70 || localCount == 0) {
            uint32_t sizeChoice = simpleRand(&seed) % 7;
            size_t objSize = 1 << (3 + sizeChoice);
            
            void* ptr = malloc(objSize);
            
            if(ptr && localCount < 64) {
                localPtrs[localCount] = ptr;
                localSizes[localCount] = objSize;
                localIds[localCount] = simpleRand(&seed) % 0xFFFFu;
                localCount++;
                
                intr::atomic::add_system(&tracker->stats[0], 1u);
                
                if(objSize >= 4) {
                    uint32_t* intPtr = static_cast<uint32_t*>(ptr);
                    uint32_t new_val = (tid << 16) | (localIds[localCount-1] & 0xFFFF);
                    intr::atomic::exch_system(intPtr, new_val);
                }
            } else {
                intr::atomic::add_system(&tracker->stats[2], 1u);
            }
        } else {
            if(localCount > 0) {
                uint32_t idx = simpleRand(&seed) % localCount;
                void* ptr = localPtrs[idx];
                size_t objSize = localSizes[idx];
                uint32_t vid = localIds[idx];
                
                __threadfence_system();
                
                if(objSize >= 4) {
                    uint32_t* intPtr = static_cast<uint32_t*>(ptr);
                    uint32_t expected = (tid << 16) | (vid & 0xFFFF);
                    uint32_t old_val = intr::atomic::CAS_system((unsigned int*)intPtr, (unsigned int)expected, 0u);
                }
                
                free(ptr);
                intr::atomic::add_system(&tracker->stats[1], 1u);
                
                localPtrs[idx] = localPtrs[localCount-1];
                localSizes[idx] = localSizes[localCount-1];
                localIds[idx] = localIds[localCount-1];
                localCount--;
            }
        }
        
        if((simpleRand(&seed) % 100) == 0) {
            __syncthreads();
        }
    }
    
    for(int j = 0; j < localCount; j++) {
        free(localPtrs[j]);
        intr::atomic::add_system(&tracker->stats[1], 1u);
    }
}

void runCudaMallocTest(const TestConfig& config) {
    std::cout << "=== CUDA Malloc/Free Test ===" << std::endl;
    config.print();
    
    size_t heapSize = 512 * 1024 * 1024;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);
    
    GPUTracker* d_tracker;
    uint32_t* d_stats;
    uint32_t* d_shouldStop;
    
    cudaMalloc(&d_tracker, sizeof(GPUTracker));
    cudaMalloc(&d_stats, 4 * sizeof(uint32_t));
    cudaMalloc(&d_shouldStop, sizeof(uint32_t));
    
    cudaMemset(d_stats, 0, 4 * sizeof(uint32_t));
    
    uint32_t stopFlag = 0;
    cudaMemcpy(d_shouldStop, &stopFlag, sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    GPUTracker h_tracker;
    h_tracker.trackingArena = nullptr;
    h_tracker.stats = d_stats;
    cudaMemcpy(d_tracker, &h_tracker, sizeof(GPUTracker), cudaMemcpyHostToDevice);
    
    std::cout << "Launching kernel..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    cudaMallocTestKernel<<<config.numBlocks, config.threadsPerBlock>>>(d_tracker, config.iterations, d_shouldStop);
    
    cudaError_t result = cudaDeviceSynchronize();
    if(result != cudaSuccess) {
        std::cerr << ":( - CUDA error: " << cudaGetErrorString(result) << std::endl;
        cudaFree(d_tracker);
        cudaFree(d_stats);
        cudaFree(d_shouldStop);
        return;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    uint32_t h_stats[4];
    cudaMemcpy(h_stats, d_stats, 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    float conversion = 1000000.0;
    
    std::cout << "\nCUDA Malloc Results:" << std::endl;
    std::cout << "Duration: " << duration.count() << " μs" << std::endl;
    std::cout << "Allocs: " << h_stats[0] << std::endl;
    std::cout << "Frees: " << h_stats[1] << std::endl;
    std::cout << "Failures: " << h_stats[2] << std::endl;
    std::cout << "Throughput: " << ((h_stats[0] + h_stats[1]) * conversion) / duration.count() << " ops/sec" << std::endl;
    
    uint32_t totalOps = h_stats[0] + h_stats[1] + h_stats[2];
    if(totalOps > 0) {
        std::cout << "Success rate: " << (100.0 * (h_stats[0] + h_stats[1])) / totalOps << "%" << std::endl;
    }
    
    cudaFree(d_tracker);
    cudaFree(d_stats);
    cudaFree(d_shouldStop);
    
    std::cout << std::endl;
}

void runFullComparison(const TestConfig& config) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "   COMPREHENSIVE ALLOCATOR COMPARISON" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    std::cout << "--- 1. GPU Slab Allocator ---" << std::endl;
    runGPUAllocatorTest(config);
    
    std::cout << "--- 2. GPU Standard CUDA malloc/free ---" << std::endl;
    runCudaMallocTest(config);
    
    std::cout << "========================================" << std::endl;
    std::cout << "   COMPARISON COMPLETE" << std::endl;
    std::cout << "========================================" << std::endl;
}

int main() {
    std::cout << "GPU Allocator Testing" << std::endl;
    std::cout << "=====================" << std::endl << std::endl;
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount == 0) {
        std::cout << "No CUDA devices found. Exiting." << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl << std::endl;
    
    try {
        TestConfig config = getTestConfig(prop);
        
        runFullComparison(config);
        
        std::cout << "\nRun stress test? (y/n): ";
        std::string response;
        std::getline(std::cin, response);
        if (!response.empty() && (response[0] == 'y' || response[0] == 'Y')) {
            runGPUStressTest(config);
        }
        
        std::cout << "All GPU tests completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}