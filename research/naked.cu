// allocator.cu - Implementation file structure
// This file shows how to organize your .cu implementation

#include "allocator.h"
#include "mod.h" 


template<typename T, size_t SIZE>
__host__ __device__ SimplePool<T, SIZE>::SimplePool() : count(0) {
    for(size_t i = 0; i < SIZE; i++) {
        used[i] = false;
        items[i] = static_cast<T>(i);
    }
}

template<typename T, size_t SIZE>
__host__ __device__ T SimplePool<T, SIZE>::takeIndex() {
    for(size_t i = 0; i < SIZE; i++) {
        if(!used[i]) {
            used[i] = true;
            count++;
            return items[i];
        }
    }
    return AdrInfo<T>::null();
}

template<typename T, size_t SIZE>
__host__ __device__ void SimplePool<T, SIZE>::giveIndex(T index) {
    if(index < SIZE && used[index]) {
        used[index] = false;
        count--;
    }
}


template <typename SLAB_TYPE>
__host__ __device__ 
size_t defaultSlabProxy<SLAB_TYPE>::getSize() const 
{ return (allocState & SIZE_MASK) >> OFFSET; }

template <typename SLAB_TYPE>
__host__ __device__ 
bool defaultSlabProxy<SLAB_TYPE>::bindSize(unsigned int newSize) {
    allocMaskElem longSize = newSize;
    allocMaskElem prev = intr::atomic::CAS_system(&allocState, 0llu, longSize << OFFSET);
    return (prev == 0);
}

template <typename SLAB_TYPE>
__host__ __device__ 
bool defaultSlabProxy<SLAB_TYPE>::clearAllocState() {
    if (allocState == 0)
        return false;
    
    allocMaskElem prev = intr::atomic::exch_system(&allocState, 0llu);
    return ((prev & SIZE_MASK) != 0) && ((prev & COUNT_MASK) == 0);
}

template <typename SLAB_TYPE>
__host__ __device__ 
size_t defaultSlabProxy<SLAB_TYPE>::slabObjCountNoMask(size_t objectSize) const {
    return SLAB_SIZE / objectSize;
}

template <typename SLAB_TYPE>
__host__ __device__ 
size_t defaultSlabProxy<SLAB_TYPE>::slabObjCountWithMask(size_t objectSize) const {
    size_t objectBitSize = objectSize * 8;
    size_t bitCostPerObj = (objectBitSize + 1);
    size_t totalBits = SLAB_ELEM_COUNT * SLAB_ELEM_BIT_SIZE;
    size_t idealTOBC = (totalBits * objectBitSize) / bitCostPerObj;
    size_t maxExcluTOBC = (idealTOBC / SLAB_ELEM_BIT_SIZE) * SLAB_ELEM_BIT_SIZE;
    
    size_t objectCount = maxExcluTOBC / objectBitSize;
    return objectCount;
}

template <typename SLAB_TYPE>
__host__ __device__ 
size_t defaultSlabProxy<SLAB_TYPE>::slabObjCount(size_t objectSize) const {
    size_t result = slabObjCountNoMask(objectSize);
    if(result > 64)
        result = slabObjCountWithMask(objectSize);
    return result;
}

template <typename SLAB_TYPE>
__host__ __device__ 
void defaultSlabProxy<SLAB_TYPE>::clearSlab(SLAB_TYPE* slab, size_t maskElemCount) {
    for (size_t i = 0; i < maskElemCount; i++)
        slab->data[i] = 0;
}

template <typename SLAB_TYPE>
__host__ __device__ 
bool defaultSlabProxy<SLAB_TYPE>::claim(SLAB_TYPE* slab, size_t objectSize) {
    AllocState startingState = ((AllocState)objectSize) << OFFSET;
    
    if(intr::atomic::CAS_system(&allocState, (AllocState)0, startingState) != 0)
        return false;
    
    size_t objectCount = slabObjCount(objectSize);
    if(objectCount == 0)
        return false;
    
    allocMask = 0;
    if(objectCount > SLAB_ELEM_BIT_SIZE) {
        size_t maskElemCount = (objectCount + SLAB_ELEM_BIT_SIZE - 1) / SLAB_ELEM_BIT_SIZE;
        clearSlab(slab, maskElemCount);
    }
    return true;
}

template <typename SLAB_TYPE>
__host__ __device__ 
bool defaultSlabProxy<SLAB_TYPE>::attemptMaskAlloc(allocMaskElem& elem, size_t& result) {
    allocMaskElem maskCopy = allocMask;
    
    while(intr::atomic::population_count(maskCopy) < SLAB_ELEM_BIT_SIZE) {
        size_t target = intr::bitwise::first_set(~maskCopy);
        allocMaskElem targetMask = 1ULL << target;
        maskCopy = intr::atomic::or_system(&allocMask, targetMask);
        
        if((maskCopy & targetMask) == 0) {
            result = target;
            return true;
        }
    }
    return false;
}

template <typename SLAB_TYPE>
__host__ __device__ 
void* defaultSlabProxy<SLAB_TYPE>::indexToPtr(SLAB_TYPE* slab, size_t objectSize, size_t maskElemCount, size_t ind) {
    char* bytePtr = static_cast<char*>(static_cast<void*>(slab));
    bytePtr += sizeof(allocMaskElem) * maskElemCount;
    bytePtr += objectSize * ind;
    return static_cast<void*>(bytePtr);
}

template <typename SLAB_TYPE>
__host__ __device__ 
void* defaultSlabProxy<SLAB_TYPE>::alloc(SLAB_TYPE* slab, bool& slabFilled) {
    allocMaskElem prev = intr::atomic::add_system(&allocState, (allocMaskElem)1);
    
    allocMaskElem prevCount = prev & COUNT_MASK;
    allocMaskElem objectSize = (prev & SIZE_MASK) >> OFFSET;
    
    allocMaskElem maxObjCount = slabObjCount(objectSize);
    if(prevCount >= maxObjCount) {
        intr::atomic::add_system(&allocState, ((AllocState)0) - ((AllocState)1));
        return nullptr;
    }
    
    slabFilled = (prevCount == (maxObjCount - 1));
    
    size_t maskElemCount = (maxObjCount + SLAB_ELEM_BIT_SIZE - 1) / SLAB_ELEM_BIT_SIZE;
    if(maxObjCount <= SLAB_ELEM_BIT_SIZE) {
        size_t index;
        if(attemptMaskAlloc(allocMask, index)) {
            return indexToPtr(slab, objectSize, maskElemCount, index);
        } else 
            return nullptr;
    } else {
        for(size_t i = 0; i < 2048; i++) {
            for(size_t j = 0; j < SLAB_ELEM_COUNT; j++) {
                size_t index;
                if(attemptMaskAlloc(slab->data[j], index)) {
                    size_t fullIndex = SLAB_ELEM_BIT_SIZE * j + index;
                    return indexToPtr(slab, objectSize, maskElemCount, fullIndex);
                }
            }
        }
        return nullptr;
    }
}

template <typename SLAB_TYPE>
__host__ __device__ 
bool defaultSlabProxy<SLAB_TYPE>::free(SLAB_TYPE* slab, void* objPtr, bool& slabEmptied) {
    char* objBytePtr = static_cast<char*>(objPtr);
    char* slabBytePtr = static_cast<char*>(static_cast<void*>(slab));
    size_t byteOffset = objBytePtr - slabBytePtr;
    
    allocMaskElem prev = intr::atomic::add_system(&allocState, ((AllocState)0) - ((AllocState)1));
    allocMaskElem prevCount = prev & COUNT_MASK;
    
    if(prevCount == 0) {
        intr::atomic::add_system(&allocState, 1llu);
        return false;
    }
    
    slabEmptied = (prevCount == 1);
    
    size_t objectSize = (prev & SIZE_MASK) >> OFFSET;
    size_t maxObjCount = slabObjCount(objectSize);
    
    size_t maskCount = (maxObjCount + sizeof(allocMaskElem) - 1) / sizeof(allocMaskElem);
    size_t maskSize = maskCount * sizeof(allocMaskElem);
    size_t firstObjOffset = maskSize;
    
    size_t objOffset = byteOffset - firstObjOffset;
    if(objOffset % objectSize != 0)
        return false;
    
    size_t objIndex = objOffset / objectSize;
    if(objIndex >= maxObjCount)
        return false;
    
    size_t maskIndex = objIndex / SLAB_ELEM_BIT_SIZE;
    size_t targetBit = objIndex % SLAB_ELEM_BIT_SIZE;
    allocMaskElem targetMask = ((allocMaskElem)1) << ((allocMaskElem)targetBit);
    allocMaskElem prevMask = 0;
    
    if(maskCount <= 1) {
        prevMask = intr::atomic::and_system(&allocMask, ~targetMask);
    } else 
        prevMask = intr::atomic::and_system(&(slab->data[maskIndex]), ~targetMask);
    
    return ((prevMask & targetMask) != 0);
}

template <typename ARENA_SIZE, template<typename> typename SLAB_PROXY_TYPE, typename SLAB_TYPE, typename SLAB_ADDR_TYPE>
__host__ __device__ SlabArena<ARENA_SIZE, SLAB_PROXY_TYPE, SLAB_TYPE, SLAB_ADDR_TYPE>::SlabArena() {
    // Initialize the free slab pool

}

template <typename ARENA_SIZE, template<typename> typename SLAB_PROXY_TYPE, typename SLAB_TYPE, typename SLAB_ADDR_TYPE>
__host__ __device__ typename SlabArena<ARENA_SIZE, SLAB_PROXY_TYPE, SLAB_TYPE, SLAB_ADDR_TYPE>::slabAddrType 
SlabArena<ARENA_SIZE, SLAB_PROXY_TYPE, SLAB_TYPE, SLAB_ADDR_TYPE>::alloc() {
    return freeSlabs.takeIndex();
}

template <typename ARENA_SIZE, template<typename> typename SLAB_PROXY_TYPE, typename SLAB_TYPE, typename SLAB_ADDR_TYPE>
__host__ __device__ void SlabArena<ARENA_SIZE, SLAB_PROXY_TYPE, SLAB_TYPE, SLAB_ADDR_TYPE>::free(slabAddrType addr) {
    freeSlabs.giveIndex(addr);
}

template <typename ARENA_SIZE, template<typename> typename SLAB_PROXY_TYPE, typename SLAB_TYPE, typename SLAB_ADDR_TYPE>
__host__ __device__ 
typename SlabArena<ARENA_SIZE, SLAB_PROXY_TYPE, SLAB_TYPE, SLAB_ADDR_TYPE>::slabAddrType 
SlabArena<ARENA_SIZE, SLAB_PROXY_TYPE, SLAB_TYPE, SLAB_ADDR_TYPE>::slabIndexFor(void* ptr) const {
    char* bytePtr = static_cast<char*>(ptr);
    char* baseBytePtr = static_cast<char*>(static_cast<void*>(const_cast<slabType*>(slabs.arena)));
    
    size_t ptrOffset = bytePtr - baseBytePtr;
    size_t slabIndex = ptrOffset / SLAB_TYPE::SIZE;
    
    return slabIndex;
}

template <typename ARENA_SIZE, template<typename> typename SLAB_PROXY_TYPE, typename SLAB_TYPE, typename SLAB_ADDR_TYPE>
__host__ __device__ 
typename SlabArena<ARENA_SIZE, SLAB_PROXY_TYPE, SLAB_TYPE, SLAB_ADDR_TYPE>::slabType& 
SlabArena<ARENA_SIZE, SLAB_PROXY_TYPE, SLAB_TYPE, SLAB_ADDR_TYPE>::slabAt(slabAddrType slabIndex) {
    return slabs.arena[slabIndex];
}

template <typename ARENA_SIZE, template<typename> typename SLAB_PROXY_TYPE, typename SLAB_TYPE, typename SLAB_ADDR_TYPE>
__host__ __device__ 
const typename SlabArena<ARENA_SIZE, SLAB_PROXY_TYPE, SLAB_TYPE, SLAB_ADDR_TYPE>::slabType& 
SlabArena<ARENA_SIZE, SLAB_PROXY_TYPE, SLAB_TYPE, SLAB_ADDR_TYPE>::slabAt(slabAddrType slabIndex) const {
    return slabs.arena[slabIndex];
}

template <typename ARENA_SIZE, template<typename> typename SLAB_PROXY_TYPE, typename SLAB_TYPE, typename SLAB_ADDR_TYPE>
__host__ __device__ 
typename SlabArena<ARENA_SIZE, SLAB_PROXY_TYPE, SLAB_TYPE, SLAB_ADDR_TYPE>::proxyNodeType& 
SlabArena<ARENA_SIZE, SLAB_PROXY_TYPE, SLAB_TYPE, SLAB_ADDR_TYPE>::proxyAt(slabAddrType slabIndex) {
    return proxies.arena[slabIndex];
}

template <typename ARENA_SIZE, template<typename> typename SLAB_PROXY_TYPE, typename SLAB_TYPE, typename SLAB_ADDR_TYPE>
__host__ __device__ 
const typename SlabArena<ARENA_SIZE, SLAB_PROXY_TYPE, SLAB_TYPE, SLAB_ADDR_TYPE>::proxyNodeType& 
SlabArena<ARENA_SIZE, SLAB_PROXY_TYPE, SLAB_TYPE, SLAB_ADDR_TYPE>::proxyAt(slabAddrType slabIndex) const {
    return proxies.arena[slabIndex];
}

template <typename ARENA_SIZE, template<typename> typename SLAB_PROXY_TYPE, typename SLAB_TYPE, typename SLAB_ADDR_TYPE>
__host__ __device__ typename SlabArena<ARENA_SIZE, SLAB_PROXY_TYPE, SLAB_TYPE, SLAB_ADDR_TYPE>::slabType& 
SlabArena<ARENA_SIZE, SLAB_PROXY_TYPE, SLAB_TYPE, SLAB_ADDR_TYPE>::slabFor(void* ptr) {
    slabAddrType slabIndex = slabIndexFor(ptr);
    return slabs.arena[slabIndex];
}

template <typename ARENA_SIZE, template<typename> typename SLAB_PROXY_TYPE, typename SLAB_TYPE, typename SLAB_ADDR_TYPE>
__host__ __device__ typename SlabArena<ARENA_SIZE, SLAB_PROXY_TYPE, SLAB_TYPE, SLAB_ADDR_TYPE>::proxyNodeType& 
SlabArena<ARENA_SIZE, SLAB_PROXY_TYPE, SLAB_TYPE, SLAB_ADDR_TYPE>::proxyFor(void* ptr) {
    slabAddrType slabIndex = slabIndexFor(ptr);
    return proxies.arena[slabIndex];
}

// =============================================================================
// SIZED ALLOCATOR IMPLEMENTATION
// =============================================================================

template<typename SLAB_ALLOCATOR_TYPE, size_t POOL_SIZE>
__host__ __device__ SizedAllocator<SLAB_ALLOCATOR_TYPE, POOL_SIZE>::SizedAllocator(
    SlabAllocatorType& slabAllocator, 
    size_t objectSize
) : slabAllocator(slabAllocator), objectSize(objectSize) {
    // Constructor body
}

template<typename SLAB_ALLOCATOR_TYPE, size_t POOL_SIZE>
__host__ __device__ void* SizedAllocator<SLAB_ALLOCATOR_TYPE, POOL_SIZE>::alloc() {
    SlabAddrType slabAddr = pool.takeIndex();
    
    if(AdrInfo<SlabAddrType>::isNull(slabAddr)) {
        // No available slab in pool, get a new one
        slabAddr = slabAllocator.alloc();
        if(AdrInfo<SlabAddrType>::isNull(slabAddr)) {
            return nullptr;
        }
        
        SlabType& newSlab = slabAllocator.slabAt(slabAddr);
        defaultSlabProxyType& defaultslabProxy = slabAllocator.proxyAt(slabAddr).data;
        
        if(!defaultslabProxy.claim(&newSlab, objectSize)) {
            slabAllocator.free(slabAddr);
            return nullptr;
        }
    }
    
    SlabType& slab = slabAllocator.slabAt(slabAddr);
    defaultSlabProxyType& defaultslabProxy = slabAllocator.proxyAt(slabAddr).data;
    bool slabFilled = false;
    void* result = defaultslabProxy.alloc(&slab, slabFilled);
    
    if(result != nullptr && !slabFilled) {
        // Slab still has space, return it to pool
        pool.giveIndex(slabAddr);
    }
    
    return result;
}

template<typename SLAB_ALLOCATOR_TYPE, size_t POOL_SIZE>
__host__ __device__ bool SizedAllocator<SLAB_ALLOCATOR_TYPE, POOL_SIZE>::free(void* ptr) {
    if(!ptr) return false;
    
    SlabAddrType slabAddr = slabAllocator.slabIndexFor(ptr);
    bool slabEmptied = false;
    SlabType& slab = slabAllocator.slabAt(slabAddr);
    defaultSlabProxyType& defaultslabProxy = slabAllocator.proxyAt(slabAddr).data;
    
    if(!defaultslabProxy.free(&slab, ptr, slabEmptied))
        return false;
        
    if(slabEmptied) {
        // Slab is now empty, return it to the main allocator
        slabAllocator.free(slabAddr);
    } else {
        // Slab has free space, add it back to our pool
        pool.giveIndex(slabAddr);
    }
    
    return true;
}

// =============================================================================
// GENERAL ALLOCATOR IMPLEMENTATION
// =============================================================================

template<typename SLAB_ALLOCATOR_TYPE, size_t POOL_SIZE, size_t ALLOC_LIMIT>
__host__ GeneralAllocator<SLAB_ALLOCATOR_TYPE, POOL_SIZE, ALLOC_LIMIT>::GeneralAllocator() {
    for(size_t i = 0; i < ALLOC_LIMIT; i++) {
        size_t allocSize = 1ULL << i;
        cache[i] = new SizedAllocatorType(slabAllocator, allocSize);
    }
}

template<typename SLAB_ALLOCATOR_TYPE, size_t POOL_SIZE, size_t ALLOC_LIMIT>
__host__ GeneralAllocator<SLAB_ALLOCATOR_TYPE, POOL_SIZE, ALLOC_LIMIT>::~GeneralAllocator() {
    for(size_t i = 0; i < ALLOC_LIMIT; i++) {
        delete cache[i];
    }
}

template<typename SLAB_ALLOCATOR_TYPE, size_t POOL_SIZE, size_t ALLOC_LIMIT>
__host__ __device__ size_t GeneralAllocator<SLAB_ALLOCATOR_TYPE, POOL_SIZE, ALLOC_LIMIT>::getSizeClassIndex(size_t size) const {
    if (size <= MIN_SIZE) return 0;
    
    // Find the smallest power of 2 >= size
    size_t scaledSize = (size + MIN_SIZE - 1) / MIN_SIZE;
    return 64 - intr::bitwise::leading_zeros((unsigned long long int)(scaledSize - 1));
}

template<typename SLAB_ALLOCATOR_TYPE, size_t POOL_SIZE, size_t ALLOC_LIMIT>
__host__ __device__ size_t GeneralAllocator<SLAB_ALLOCATOR_TYPE, POOL_SIZE, ALLOC_LIMIT>::roundUpToPowerOfTwo(size_t size) const {
    if (size <= MIN_SIZE) return MIN_SIZE;
    
    size_t result = MIN_SIZE;
    while (result < size) {
        result <<= 1;
    }
    return result;
}

template<typename SLAB_ALLOCATOR_TYPE, size_t POOL_SIZE, size_t ALLOC_LIMIT>
__host__ __device__ void* GeneralAllocator<SLAB_ALLOCATOR_TYPE, POOL_SIZE, ALLOC_LIMIT>::alloc(size_t size) {
    if(size == 0 || size > MAX_SIZE)
        return nullptr;
    
    size_t sizeIndex = getSizeClassIndex(size);
    
    if(sizeIndex >= ALLOC_LIMIT)
        return nullptr;
    
    return cache[sizeIndex]->alloc();
}

template<typename SLAB_ALLOCATOR_TYPE, size_t POOL_SIZE, size_t ALLOC_LIMIT>
__host__ __device__ bool GeneralAllocator<SLAB_ALLOCATOR_TYPE, POOL_SIZE, ALLOC_LIMIT>::free(void* ptr) {
    if(!ptr) return false;
    
    // Find which slab this pointer belongs to
    SlabType& slab = slabAllocator.slabFor(ptr);
    defaultSlabProxyType& proxy = slabAllocator.proxyFor(ptr).data;
    
    size_t objectSize = proxy.getSize();
    if(objectSize == 0) return false;
    
    size_t sizeIndex = getSizeClassIndex(objectSize);
    
    if(sizeIndex >= ALLOC_LIMIT)
        return false;
    
    return cache[sizeIndex]->free(ptr);
}

template<typename SLAB_ALLOCATOR_TYPE, size_t POOL_SIZE, size_t ALLOC_LIMIT>
__host__ __device__ size_t GeneralAllocator<SLAB_ALLOCATOR_TYPE, POOL_SIZE, ALLOC_LIMIT>::getTotalSlabCount() const {
    return slabAllocator.getSlabCount();
}

template<typename SLAB_ALLOCATOR_TYPE, size_t POOL_SIZE, size_t ALLOC_LIMIT>
__host__ __device__ size_t GeneralAllocator<SLAB_ALLOCATOR_TYPE, POOL_SIZE, ALLOC_LIMIT>::getUsedSlabCount() const {
    return slabAllocator.getUsedSlabCount();
}

template<typename SLAB_ALLOCATOR_TYPE, size_t POOL_SIZE, size_t ALLOC_LIMIT>
__host__ __device__ size_t GeneralAllocator<SLAB_ALLOCATOR_TYPE, POOL_SIZE, ALLOC_LIMIT>::getFreeSlabCount() const {
    return slabAllocator.getFreeSlabCount();
}

template<typename SLAB_ALLOCATOR_TYPE, size_t POOL_SIZE, size_t ALLOC_LIMIT>
__host__ __device__ size_t GeneralAllocator<SLAB_ALLOCATOR_TYPE, POOL_SIZE, ALLOC_LIMIT>::getSizeForClass(size_t classIndex) const {
    if(classIndex >= ALLOC_LIMIT) return 0;
    return 1ULL << classIndex;
}

// =============================================================================
// CUDA DEVICE ALLOCATOR INITIALIZATION
// =============================================================================

#ifdef __CUDA_ARCH__

__host__ cudaError_t initDeviceAllocator(GeneralAllocator16MB* allocator) {
    return cudaMemcpyToSymbol(g_deviceAllocator, &allocator, sizeof(GeneralAllocator16MB*));
}

#endif

// =============================================================================
// EXPLICIT TEMPLATE INSTANTIATIONS
// =============================================================================

// Instantiate commonly used types to reduce compilation time
template class SimplePool<defaultSlabAddr, 32>;
template class SimplePool<defaultSlabAddr, 64>;
template class SimplePool<defaultSlabAddr, 128>;

template class defaultSlabProxy<DefaultSlab>;

template class SlabArena<Arena1MB>;
template class SlabArena<Arena16MB>;
template class SlabArena<Arena64MB>;

template class SizedAllocator<SlabArena1MB, 32>;
template class SizedAllocator<SlabArena16MB, 64>;
template class SizedAllocator<SlabArena64MB, 128>;

template class GeneralAllocator<SlabArena1MB, 32, 12>;
template class GeneralAllocator<SlabArena16MB, 64, 16>;
template class GeneralAllocator<SlabArena64MB, 128, 20>;

// =============================================================================
// DEBUGGING AND TESTING FUNCTIONS
// =============================================================================

#ifdef SLAB_ALLOCATOR_DEBUG

template<typename AllocatorType>
__host__ void printAllocatorStats(const AllocatorType& allocator) {
    printf("=== Allocator Statistics ===\n");
    printf("Total slabs: %zu\n", allocator.getTotalSlabCount());
    printf("Used slabs: %zu\n", allocator.getUsedSlabCount());
    printf("Free slabs: %zu\n", allocator.getFreeSlabCount());
    printf("Max allocation size: %zu bytes\n", allocator.getMaxSize());
    printf("Size classes: %zu\n", allocator.getSizeClassCount());
    printf("============================\n");
}

template<typename AllocatorType>
__host__ bool testBasicAllocation(AllocatorType& allocator) {
    printf("Testing basic allocation...\n");
    
    void* ptr1 = allocator.alloc(64);
    if (!ptr1) {
        printf("FAIL: Basic allocation failed\n");
        return false;
    }
    
    void* ptr2 = allocator.alloc(128);
    if (!ptr2) {
        printf("FAIL: Second allocation failed\n");
        allocator.free(ptr1);
        return false;
    }
    
    if (!allocator.free(ptr1)) {
        printf("FAIL: First free failed\n");
        return false;
    }
    
    if (!allocator.free(ptr2)) {
        printf("FAIL: Second free failed\n");
        return false;
    }
    
    printf("PASS: Basic allocation test\n");
    return true;
}

#endif // SLAB_ALLOCATOR_DEBUG

// =============================================================================
// USAGE EXAMPLE FUNCTIONS
// =============================================================================

// Example host function
void exampleHostUsage() {
    GeneralAllocator16MB allocator;
    
    // Allocate various sizes
    void* small = allocator.alloc(32);
    void* medium = allocator.alloc(1024);
    void* large = allocator.alloc(8192);
    
    // Use the memory...
    if (small) memset(small, 0xAA, 32);
    if (medium) memset(medium, 0xBB, 1024);
    if (large) memset(large, 0xCC, 8192);
    
    // Free the memory
    allocator.free(small);
    allocator.free(medium);
    allocator.free(large);
}

// Example kernel
__global__ void exampleKernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread allocates some memory
    void* ptr = device_malloc(256);
    if (ptr) {
        // Use the memory
        int* intPtr = static_cast<int*>(ptr);
        intPtr[0] = tid;
        
        // Free the memory
        device_free(ptr);
    }
}

// Example launch function
void launchExampleKernel() {
    GeneralAllocator16MB* deviceAllocator;
    
    // Allocate device allocator
    cudaMalloc(&deviceAllocator, sizeof(GeneralAllocator16MB));
    
    // Initialize device allocator (this would need proper initialization)
    // new(deviceAllocator) GeneralAllocator16MB();
    
    // Set up device symbol
    initDeviceAllocator(deviceAllocator);
    
    // Launch kernel
    exampleKernel<<<256, 256>>>();
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Cleanup
    cudaFree(deviceAllocator);
}