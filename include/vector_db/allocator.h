#ifndef VECTOR_DB_ALLOCATOR_H
#define VECTOR_DB_ALLOCATOR_H

#define _GNU_SOURCE
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#define CACHE_LINE_SIZE 64
#define HUGE_PAGE_SIZE (2 * 1024 * 1024)
#define MAX_THREADS 128
#define ARENA_BLOCK_SIZE (16 * 1024 * 1024)

typedef enum {
    ALLOC_SUCCESS = 0,
    ALLOC_ERR_NOMEM = -1,
    ALLOC_ERR_INVAL = -2,
    ALLOC_ERR_MMAP = -3,
    ALLOC_ERR_OVERFLOW = -4
} AllocStatus;

typedef struct ArenaChunk {
    uint8_t *memory;
    size_t capacity;
    size_t offset;
    struct ArenaChunk *next;
} ArenaChunk;

typedef struct {
    ArenaChunk *head;
    ArenaChunk *current;
    size_t chunk_size;
    pthread_mutex_t lock;
    size_t total_allocated;
} ThreadArena;

typedef struct {
    ThreadArena arenas[MAX_THREADS];
    _Atomic size_t active_threads;
    bool use_huge_pages;
} ArenaManager;

static inline void *cache_align_ptr(void *ptr) {
    uintptr_t addr = (uintptr_t)ptr;
    uintptr_t aligned = (addr + (CACHE_LINE_SIZE - 1)) & ~(CACHE_LINE_SIZE - 1);
    return (void *)aligned;
}

static inline size_t cache_align_size(size_t size) {
    return (size + (CACHE_LINE_SIZE - 1)) & ~(CACHE_LINE_SIZE - 1);
}

static inline ArenaChunk *arena_chunk_create(size_t capacity, bool use_huge_pages) {
    ArenaChunk *chunk = (ArenaChunk *)malloc(sizeof(ArenaChunk));
    if (!chunk) return NULL;

    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_HUGETLB
    if (use_huge_pages) {
        flags |= MAP_HUGETLB;
    }
#endif

    void *mem = mmap(NULL, capacity, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (mem == MAP_FAILED) {
#ifdef MAP_HUGETLB
        if (use_huge_pages) {
            flags &= ~MAP_HUGETLB;
            mem = mmap(NULL, capacity, PROT_READ | PROT_WRITE, flags, -1, 0);
        }
#endif
        if (mem == MAP_FAILED) {
            free(chunk);
            return NULL;
        }
    }

    chunk->memory = (uint8_t *)mem;
    chunk->capacity = capacity;
    chunk->offset = 0;
    chunk->next = NULL;
    return chunk;
}

static inline void arena_chunk_destroy(ArenaChunk *chunk) {
    while (chunk) {
        ArenaChunk *next = chunk->next;
        munmap(chunk->memory, chunk->capacity);
        free(chunk);
        chunk = next;
    }
}

static inline AllocStatus arena_manager_init(ArenaManager *manager, size_t chunk_size, bool use_huge_pages) {
    if (!manager) return ALLOC_ERR_INVAL;
    manager->chunk_size = cache_align_size(chunk_size ? chunk_size : ARENA_BLOCK_SIZE);
    manager->use_huge_pages = use_huge_pages;
    atomic_init(&manager->active_threads, 0);

    for (int i = 0; i < MAX_THREADS; i++) {
        manager->arenas[i].head = NULL;
        manager->arenas[i].current = NULL;
        manager->arenas[i].chunk_size = manager->chunk_size;
        manager->arenas[i].total_allocated = 0;
        if (pthread_mutex_init(&manager->arenas[i].lock, NULL) != 0) {
            for (int j = 0; j < i; j++) {
                pthread_mutex_destroy(&manager->arenas[j].lock);
                arena_chunk_destroy(manager->arenas[j].head);
            }
            return ALLOC_ERR_NOMEM;
        }
    }
    return ALLOC_SUCCESS;
}

static inline void arena_manager_destroy(ArenaManager *manager) {
    if (!manager) return;
    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_mutex_lock(&manager->arenas[i].lock);
        arena_chunk_destroy(manager->arenas[i].head);
        manager->arenas[i].head = NULL;
        manager->arenas[i].current = NULL;
        pthread_mutex_unlock(&manager->arenas[i].lock);
        pthread_mutex_destroy(&manager->arenas[i].lock);
    }
}

static inline void *arena_alloc_thread(ThreadArena *arena, size_t size) {
    size_t aligned_size = cache_align_size(size);
    if (aligned_size > arena->chunk_size) {
        ArenaChunk *large_chunk = arena_chunk_create(aligned_size + CACHE_LINE_SIZE, false);
        if (!large_chunk) return NULL;
        
        pthread_mutex_lock(&arena->lock);
        large_chunk->next = arena->head;
        arena->head = large_chunk;
        arena->total_allocated += large_chunk->capacity;
        
        void *ptr = cache_align_ptr(large_chunk->memory);
        large_chunk->offset = ((uint8_t *)ptr - large_chunk->memory) + aligned_size;
        pthread_mutex_unlock(&arena->lock);
        return ptr;
    }

    pthread_mutex_lock(&arena->lock);
    if (!arena->current || (arena->current->capacity - arena->current->offset < aligned_size)) {
        ArenaChunk *new_chunk = arena_chunk_create(arena->chunk_size, false);
        if (!new_chunk) {
            pthread_mutex_unlock(&arena->lock);
            return NULL;
        }
        new_chunk->next = arena->head;
        arena->head = new_chunk;
        arena->current = new_chunk;
        arena->total_allocated += new_chunk->capacity;
    }

    void *ptr = cache_align_ptr(arena->current->memory + arena->current->offset);
    arena->current->offset = ((uint8_t *)ptr - arena->current->memory) + aligned_size;
    pthread_mutex_unlock(&arena->lock);

    return ptr;
}

static inline void *arena_alloc(ArenaManager *manager, int thread_id, size_t size) {
    if (!manager || thread_id < 0 || thread_id >= MAX_THREADS) return NULL;
    return arena_alloc_thread(&manager->arenas[thread_id], size);
}

static inline void arena_reset_thread(ThreadArena *arena) {
    pthread_mutex_lock(&arena->lock);
    ArenaChunk *chunk = arena->head;
    while (chunk) {
        chunk->offset = 0;
        chunk = chunk->next;
    }
    arena->current = arena->head;
    pthread_mutex_unlock(&arena->lock);
}

static inline void arena_reset(ArenaManager *manager, int thread_id) {
    if (!manager || thread_id < 0 || thread_id >= MAX_THREADS) return;
    arena_reset_thread(&manager->arenas[thread_id]);
}

static inline void *vector_alloc_aligned(size_t size) {
    size_t aligned_size = cache_align_size(size);
    void *ptr = NULL;
    int ret = posix_memalign(&ptr, CACHE_LINE_SIZE, aligned_size);
    if (ret != 0 || !ptr) {
        return NULL;
    }
    memset(ptr, 0, aligned_size);
    return ptr;
}

static inline void vector_free_aligned(void *ptr) {
    if (ptr) {
        free(ptr);
    }
}

static inline bool mprotect_segment(void *addr, size_t size, bool read_only) {
    uintptr_t start = (uintptr_t)addr;
    uintptr_t page_start = start & ~(sysconf(_SC_PAGESIZE) - 1);
    size_t diff = start - page_start;
    size_t total_size = cache_align_size(size + diff);
    
    int prot = read_only ? PROT_READ : (PROT_READ | PROT_WRITE);
    return mprotect((void *)page_start, total_size, prot) == 0;
}

#endif /* VECTOR_DB_ALLOCATOR_H */