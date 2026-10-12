#include "core/column.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/mman.h>
#include <errno.h>
#include <immintrin.h>

#define CACHE_LINE_SIZE 64
#define DEFAULT_CHUNK_ROWS 65536
#define VECTOR_BATCH_SIZE 64

typedef enum {
    DT_INT8 = 1,
    DT_INT16 = 2,
    DT_INT32 = 3,
    DT_INT64 = 4,
    DT_FLOAT32 = 5,
    DT_FLOAT64 = 6,
    DT_BINARY = 7
} DataType;

typedef struct {
    uint64_t seq;
    bool lock;
} __attribute__((aligned(CACHE_LINE_SIZE))) Seqlock;

struct ColumnChunk {
    DataType data_type;
    uint32_t row_count;
    uint32_t capacity_rows;
    size_t data_size;
    void* data;
    uint8_t* null_bitmap;
    size_t bitmap_size;
    uint8_t min_val[8];
    uint8_t max_val[8];
    uint32_t compression_flags;
    bool is_immutable;
    Seqlock seqlock;
    void* mmap_base;
    size_t mmap_len;
} __attribute__((aligned(CACHE_LINE_SIZE)));

struct Dictionary {
    DataType key_type;
    size_t unique_count;
    size_t capacity;
    void* sorted_keys;
    uint32_t* hash_index;
    size_t hash_mask;
} __attribute__((aligned(CACHE_LINE_SIZE)));

struct BitmapIndex {
    uint32_t word_count;
    uint64_t* words;
    size_t byte_size;
} __attribute__((aligned(CACHE_LINE_SIZE)));

struct VectorBatch {
    uint32_t size;
    void* data_ptrs[VECTOR_BATCH_SIZE];
    uint16_t selection_vector[VECTOR_BATCH_SIZE];
    uint16_t selection_size;
    bool use_selection;
} __attribute__((aligned(CACHE_LINE_SIZE)));

static inline size_t get_type_size(DataType dt) {
    switch (dt) {
        case DT_INT8: return 1;
        case DT_INT16: return 2;
        case DT_INT32:
        case DT_FLOAT32: return 4;
        case DT_INT64:
        case DT_FLOAT64: return 8;
        case DT_BINARY: return sizeof(void*);
        default: return 0;
    }
}

static inline size_t align_to_cache_line(size_t size) {
    return (size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
}

ColumnChunk* column_chunk_create(DataType dt, uint32_t capacity_rows) {
    size_t type_sz = get_type_size(dt);
    size_t raw_data_sz = type_sz * capacity_rows;
    size_t padded_data_sz = align_to_cache_line(raw_data_sz + CACHE_LINE_SIZE);
    
    size_t bitmap_sz = align_to_cache_line((capacity_rows + 7) / 8);
    size_t total_sz = sizeof(ColumnChunk) + padded_data_sz + bitmap_sz;

    total_sz = (total_sz + 4095) & ~4095;

    void* mem = mmap(NULL, total_sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        return NULL;
    }

    ColumnChunk* chunk = (ColumnChunk*)mem;
    chunk->data_type = dt;
    chunk->row_count = 0;
    chunk->capacity_rows = capacity_rows;
    chunk->data_size = padded_data_sz;
    chunk->data = (uint8_t*)mem + sizeof(ColumnChunk);
    
    chunk->null_bitmap = (uint8_t*)chunk->data + padded_data_sz;
    chunk->bitmap_size = bitmap_sz;
    memset(chunk->null_bitmap, 0xFF, bitmap_sz);

    memset(chunk->min_val, 0, 8);
    memset(chunk->max_val, 0, 8);
    chunk->compression_flags = 0;
    chunk->is_immutable = false;
    
    chunk->seqlock.seq = 0;
    chunk->seqlock.lock = false;

    chunk->mmap_base = mem;
    chunk->mmap_len = total_sz;

    return chunk;
}

void column_chunk_destroy(ColumnChunk* chunk) {
    if (!chunk) return;
    if (chunk->is_immutable) {
        mprotect(chunk->mmap_base, chunk->mmap_len, PROT_READ | PROT_WRITE);
    }
    munmap(chunk->mmap_base, chunk->mmap_len);
}

static void update_bounds(ColumnChunk* chunk, const void* val) {
    size_t sz = get_type_size(chunk->data_type);
    if (chunk->row_count == 0) {
        memcpy(chunk->min_val, val, sz);
        memcpy(chunk->max_val, val, sz);
        return;
    }

    switch (chunk->data_type) {
        case DT_INT8: {
            int8_t v = *(const int8_t*)val;
            int8_t cur_min = *(int8_t*)chunk->min_val;
            int8_t cur_max = *(int8_t*)chunk->max_val;
            if (v < cur_min) *(int8_t*)chunk->min_val = v;
            if (v > cur_max) *(int8_t*)chunk->max_val = v;
            break;
        }
        case DT_INT16: {
            int16_t v = *(const int16_t*)val;
            int16_t cur_min = *(int16_t*)chunk->min_val;
            int16_t cur_max = *(int16_t*)chunk->max_val;
            if (v < cur_min) *(int16_t*)chunk->min_val = v;
            if (v > cur_max) *(int16_t*)chunk->max_val = v;
            break;
        }
        case DT_INT32: {
            int32_t v = *(const int32_t*)val;
            int32_t cur_min = *(int32_t*)chunk->min_val;
            int32_t cur_max = *(int32_t*)chunk->max_val;
            if (v < cur_min) *(int32_t*)chunk->min_val = v;
            if (v > cur_max) *(int32_t*)chunk->max_val = v;
            break;
        }
        case DT_INT64: {
            int64_t v = *(const int64_t*)val;
            int64_t cur_min = *(int64_t*)chunk->min_val;
            int64_t cur_max = *(int64_t*)chunk->max_val;
            if (v < cur_min) *(int64_t*)chunk->min_val = v;
            if (v > cur_max) *(int64_t*)chunk->max_val = v;
            break;
        }
        case DT_FLOAT32: {
            float v = *(const float*)val;
            float cur_min = *(float*)chunk->min_val;
            float cur_max = *(float*)chunk->max_val;
            if (v < cur_min) *(float*)chunk->min_val = *(const uint32_t*)val;
            if (v > cur_max) *(float*)chunk->max_val = *(const uint32_t*)val;
            break;
        }
        case DT_FLOAT64: {
            double v = *(const double*)val;
            double cur_min = *(double*)chunk->min_val;
            double cur_max = *(double*)chunk->max_val;
            if (v < cur_min) *(double*)chunk->min_val = *(const uint64_t*)val;
            if (v > cur_max) *(double*)chunk->max_val = *(const uint64_t*)val;
            break;
        }
        default:
            break;
    }
}

int column_chunk_append(ColumnChunk* chunk, const void* val, bool is_null) {
    if (!chunk || chunk->is_immutable || chunk->row_count >= chunk->capacity_rows) {
        return -EINVAL;
    }

    while (__atomic_test_and_set(&chunk->seqlock.lock, __ATOMIC_ACQUIRE)) {
        __builtin_ia32_pause();
    }
    __atomic_add_fetch(&chunk->seqlock.seq, 1, __ATOMIC_RELEASE);

    size_t idx = chunk->row_count;
    size_t type_sz = get_type_size(chunk->data_type);

    if (is_null) {
        size_t byte_idx = idx / 8;
        size_t bit_idx = idx % 8;
        chunk->null_bitmap[byte_idx] &= ~(1U << bit_idx);
        memset((uint8_t*)chunk->data + (idx * type_sz), 0, type_sz);
    } else {
        size_t byte_idx = idx / 8;
        size_t bit_idx = idx % 8;
        chunk->null_bitmap[byte_idx] |= (1U << bit_idx);
        memcpy((uint8_t*)chunk->data + (idx * type_sz), val, type_sz);
        update_bounds(chunk, val);
    }

    chunk->row_count++;

    __atomic_add_fetch(&chunk->seqlock.seq, 1, __ATOMIC_RELEASE);
    __atomic_clear(&chunk->seqlock.lock, __ATOMIC_RELEASE);

    return 0;
}

int column_chunk_freeze(ColumnChunk* chunk) {
    if (!chunk || chunk->is_immutable) return -EINVAL;

    while (__atomic_test_and_set(&chunk->seqlock.lock, __ATOMIC_ACQUIRE)) {
        __builtin_ia32_pause();
    }
    __atomic_add_fetch(&chunk->seqlock.seq, 1, __ATOMIC_RELEASE);

    chunk->is_immutable = true;

    if (mprotect(chunk->mmap_base, chunk->mmap_len, PROT_READ) != 0) {
        chunk->is_immutable = false;
        __atomic_clear(&chunk->seqlock.lock, __ATOMIC_RELEASE);
        return -errno;
    }

    __atomic_add_fetch(&chunk->seqlock.seq, 1, __ATOMIC_RELEASE);
    __atomic_clear(&chunk->seqlock.lock, __ATOMIC_RELEASE);

    return 0;
}

uint64_t column_chunk_begin_read(const ColumnChunk* chunk) {
    uint64_t seq;
    do {
        seq = __atomic_load_n(&chunk->seqlock.seq, __ATOMIC_ACQUIRE);
        if (seq & 1) {
            __builtin_ia32_pause();
            continue;
        }
        break;
    } while (true);
    return seq;
}

bool column_chunk_validate_read(const ColumnChunk* chunk, uint64_t start_seq) {
    uint64_t end_seq = __atomic_load_n(&chunk->seqlock.seq, __ATOMIC_ACQUIRE);
    return start_seq == end_seq && (end_seq & 1) == 0;
}

Dictionary* dictionary_create(DataType key_type, size_t capacity) {
    size_t total_sz = sizeof(Dictionary);
    Dictionary* dict = (Dictionary*)malloc(total_sz);
    if (!dict) return NULL;

    dict->key_type = key_type;
    dict->unique_count = 0;
    dict->capacity = capacity;
    
    size_t key_sz = get_type_size(key_type);
    dict->sorted_keys = aligned_alloc(CACHE_LINE_SIZE, align_to_cache_line(key_sz * capacity));
    
    size_t hash_cap = 1;
    while (hash_cap < capacity * 2) hash_cap <<= 1;
    dict->hash_index = (uint32_t*)malloc(hash_cap * sizeof(uint32_t));
    memset(dict->hash_index, 0xFF, hash_cap * sizeof(uint32_t));
    dict->hash_mask = hash_cap - 1;

    if (!dict->sorted_keys || !dict->hash_index) {
        free(dict->sorted_keys);
        free(dict->hash_index);
        free(dict);
        return NULL;
    }

    return dict;
}

void dictionary_destroy(Dictionary* dict) {
    if (!dict) return;
    free(dict->sorted_keys);
    free(dict->hash_index);
    free(dict);
}

static uint32_t hash_key(const void* key, DataType dt) {
    uint32_t h = 2166136261u;
    size_t len = get_type_size(dt);
    const uint8_t* p = (const uint8_t*)key;
    for (size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= 16777619u;
    }
    return h;
}

int dictionary_lookup_or_insert(Dictionary* dict, const void* key, uint32_t* out_id) {
    if (!dict || !key || !out_id) return -EINVAL;

    uint32_t h = hash_key(key, dict->key_type);
    size_t idx = h & dict->hash_mask;
    size_t key_sz = get_type_size(dict->key_type);
    uint8_t* keys_base = (uint8_t*)dict->sorted_keys;

    for (size_t i = 0; i <= dict->hash_mask; ++i) {
        size_t probe = (idx + i) & dict->hash_mask;
        uint32_t existing_id = dict->hash_index[probe];
        
        if (existing_id == 0xFFFFFFFF) {
            if (dict->unique_count >= dict->capacity) return -ENOMEM;
            uint32_t new_id = (uint32_t)dict->unique_count;
            memcpy(keys_base + (new_id * key_sz), key, key_sz);
            dict->hash_index[probe] = new_id;
            dict->unique_count++;
            *out_id = new_id;
            return 0;
        }

        if (memcmp(keys_base + (existing_id * key_sz), key, key_sz) == 0) {
            *out_id = existing_id;
            return 0;
        }
    }

    return -ENOMEM;
}

BitmapIndex* bitmap_index_create(uint32_t word_count) {
    BitmapIndex* bm = (BitmapIndex*)malloc(sizeof(BitmapIndex));
    if (!bm) return NULL;

    bm->word_count = word_count;
    bm->byte_size = align_to_cache_line(word_count * sizeof(uint64_t));
    bm->words = (uint64_t*)aligned_alloc(CACHE_LINE_SIZE, bm->byte_size);