#pragma once

typedef unsigned long long int defaultAllocMaskElem;
typedef unsigned long long int defaultSlabAddr;

const size_t DEFAULT_SLAB_SIZE = 1 << 16;  // 64KB default slab size

template <
    size_t SLAB_SIZE = DEFAULT_SLAB_SIZE, 
    typename ELEM_TYPE = defaultAllocMaskElem
>

struct Slab {
    typedef ELEM_TYPE allocMaskElem;
    
    // constexpr...
    static const size_t ELEM_SIZE  = sizeof(ELEM_TYPE);
    static const size_t SIZE       = SLAB_SIZE;
    static const size_t ELEM_COUNT = (SLAB_SIZE + ELEM_SIZE - 1) / ELEM_SIZE;
    
    // The actual data storage
    ELEM_TYPE data[ELEM_COUNT];
};

/////

template <typename SLAB_TYPE>
struct defaultSlabProxy {
    typedef SLAB_TYPE slabType;
    typedef typename SLAB_TYPE::AllocMaskElem allocMaskElem;
    typedef unsigned long long int AllocState;
    
    // constexpr...
    static const AllocState OFFSET              = sizeof(AllocState) * 8 / 2;
    static const AllocState COUNT_MASK          = (((AllocState)1) << OFFSET) - (AllocState)1;
    static const AllocState SIZE_MASK           = ~COUNT_MASK;
    static const size_t     SLAB_SIZE           = SLAB_TYPE::SIZE;
    static const size_t     SLAB_ELEM_COUNT     = SLAB_TYPE::ELEM_COUNT;
    static const size_t     SLAB_ELEM_BIT_SIZE  = sizeof(allocMaskElem);    // * 8;
    
    
    allocMaskElem allocMask;    
    AllocState allocState;      

    // major leagues
    __host__ __device__ size_t getSize() const;
    __host__ __device__ bool bindSize(unsigned int);
    __host__ __device__ bool clearAllocState();
    __host__ __device__ bool claim(SLAB_TYPE*, size_t);
    __host__ __device__ void* alloc(SLAB_TYPE*, bool&);
    __host__ __device__ bool free(SLAB_TYPE*, void*, bool&);
    
    // cherry on top
    __host__ __device__ size_t slabObjCount(size_t) const;
    __host__ __device__ size_t slabObjCountNoMask(size_t) const;
    __host__ __device__ size_t slabObjCountWithMask(size_t) const;
    
private:
    // helpers
    __host__ __device__ void clearSlab(SLAB_TYPE*, size_t);
    __host__ __device__ bool attemptMaskAlloc(allocMaskElem&, size_t&);
    __host__ __device__ void* indexToPtr(SLAB_TYPE*, size_t, size_t, size_t);
};




template <
    typename ARENA_SIZE,
    template<typename> typename SLAB_PROXY_TYPE = deafualtSlabProxy,
    typename SLAB_TYPE = Slab<DEFAULT_SLAB_SIZE>,
    typename SLAB_ADDR_TYPE = defaultSlabAddr
>
class SlabArena {
public:
    typedef SLAB_ADDR_TYPE slabAddrType;
    typedef SLAB_TYPE slabType;
    typedef SLAB_PROXY_TYPE<SLAB_TYPE> slabProxyType;
    typedef Node<slabProxyType, slabAddrType, Size<2>> proxyNodeType;
    
    // what value is this - i think i know the size
    static const size_t SLAB_COUNT = (ARENA_SIZE::VALUE + SLAB_TYPE::SIZE - 1) / SLAB_TYPE::SIZE;
    
    typedef DirectArena <
        slabType, 
        slabAddrType, 
        Size<SLAB_COUNT>
    > BackingArenaType;

    typedef DirectArena  <
        proxyNodeType, 
        slabAddrType, 
        Size<SLAB_COUNT>
    > ProxyArenaType;
    
private:
    BackingArenaType slabs;
    ProxyArenaType proxies;
    SimplePool<slabAddrType, SLAB_COUNT> freeSlabs;
    
public:
    __host__ __device__ SlabArena();
    
    __host__ __device__ slabAddrType alloc();
    __host__ __device__ void free(slabAddrType addr);
    
    __host__ __device__ slabAddrType slabIndexFor(void* ptr) const;
    __host__ __device__ slabType& slabAt(slabAddrType slabIndex);
    __host__ __device__ proxyNodeType& proxyAt(slabAddrType slabIndex);
    __host__ __device__ slabType& slabFor(void* ptr);
    __host__ __device__ proxyNodeType& proxyFor(void* ptr);
    
    __host__ __device__ size_t getSlabCount() const { return SLAB_COUNT; }
    __host__ __device__ size_t getFreeSlabCount() const { return freeSlabs.getCount(); }
    __host__ __device__ size_t getUsedSlabCount() const { return SLAB_COUNT - freeSlabs.getCount(); }
};


template<typename SLAB_ALLOCATOR_TYPE, size_t POOL_SIZE>
class SizedAllocator {
public:
    typedef SLAB_ALLOCATOR_TYPE SlabAllocatorType;
    typedef typename SlabAllocatorType::slabType SlabType;
    typedef typename SlabAllocatorType::slabAddrType SlabAddrType;
    typedef typename SlabAllocatorType::slabProxyType SlabProxyType;
    typedef SimplePool<SlabAddrType, POOL_SIZE> PoolType;
    typedef typename SlabProxyType::allocMaskElem AllocMaskElem;
    
private:
    SlabAllocatorType& slabAllocator;
    size_t objectSize;
    PoolType pool;
    
public:
    __host__ __device__ SizedAllocator(SlabAllocatorType& slabAllocator, size_t objectSize);
    
    __host__ __device__ void* alloc();
    __host__ __device__ bool free(void* ptr);
    
    __host__ __device__ size_t getObjectSize() const { return objectSize; }
    __host__ __device__ size_t getAvailableSlabCount() const { return pool.getCount(); }
};


template<typename SLAB_ALLOCATOR_TYPE, size_t POOL_SIZE, size_t ALLOC_LIMIT>
class GeneralAllocator {
public:
    typedef SLAB_ALLOCATOR_TYPE SlabAllocatorType;
    typedef typename SlabAllocatorType::slabType SlabType;
    typedef typename SlabAllocatorType::slabProxyType SlabProxyType;
    typedef typename SlabProxyType::allocMaskElem AllocMaskElem;
    typedef SizedAllocator<SLAB_ALLOCATOR_TYPE, POOL_SIZE> SizedAllocatorType;
    
    static const size_t MAX_SIZE = 1 << ALLOC_LIMIT;
    static const size_t MIN_SIZE = 1;
    
private:
    SlabAllocatorType slabAllocator;
    SizedAllocatorType* cache[ALLOC_LIMIT];
    
public:
    __host__ GeneralAllocator();
    __host__ ~GeneralAllocator();
    
    __host__ __device__ void* alloc(size_t size);
    __host__ __device__ bool free(void* ptr);
    
    __host__ __device__ size_t getMaxSize() const { return MAX_SIZE; }
    __host__ __device__ size_t getMinSize() const { return MIN_SIZE; }
    __host__ __device__ size_t getSizeClassCount() const { return SIZE_CLASS_COUNT; }
    
    __host__ __device__ size_t getTotalSlabCount() const;
    __host__ __device__ size_t getUsedSlabCount() const;
    __host__ __device__ size_t getFreeSlabCount() const;
    
    __host__ __device__ size_t getSizeForClass(size_t classIndex) const;
    __host__ __device__ size_t getClassForSize(size_t size) const { return getSizeClassIndex(size); }
};



