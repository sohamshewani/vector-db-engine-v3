#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <pthread.h>
#include <sys/mman.h>
#include <errno.h>
#include <immintrin.h>

#define VECTOR_BATCH_SIZE 64
#define CACHE_LINE_SIZE 64
#define CHUNK_ROW_CAPACITY 65536

typedef enum {
    DTYPE_INT32 = 0,
    DTYPE_INT64 = 1,
    DTYPE_FLOAT32 = 2,
    DTYPE_FLOAT64 = 3
} DataType;

typedef enum {
    OP_EQ = 0,
    OP_NEQ = 1,
    OP_LT = 2,
    OP_LTE = 3,
    OP_GT = 4,
    OP_GTE = 5
} ComparisonOp;

typedef struct {
    DataType type;
    uint64_t row_count;
    uint8_t* null_bitmap;
    uint8_t* data;
    uint32_t compression_flags;
    int64_t min_val_i64;
    int64_t max_val_i64;
    pthread_rwlock_t rwlock;
    uint32_t seqlock;
} ColumnChunk;

typedef struct {
    uint16_t indices[VECTOR_BATCH_SIZE];
    uint16_t size;
    bool is_filtered;
} SelectionVector;

typedef struct {
    void* data_ptrs[VECTOR_BATCH_SIZE];
    DataType types[VECTOR_BATCH_SIZE];
    uint32_t column_count;
    SelectionVector sel_vector;
    uint32_t row_count;
} VectorBatch;

typedef struct {
    uint64_t* words;
    uint32_t word_count;
    uint32_t capacity;
} BitmapIndex;

static inline void* arena_alloc_aligned(size_t size) {
    size_t aligned_size = (size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
    void* ptr = NULL;
    int ret = posix_memalign(&ptr, CACHE_LINE_SIZE, aligned_size);
    if (ret != 0 || ptr == NULL) {
        return NULL;
    }
    memset(ptr, 0, aligned_size);
    return ptr;
}

static inline void arena_free_aligned(void* ptr) {
    free(ptr);
}

ColumnChunk* vector_engine_create_chunk(DataType type, uint64_t row_count) {
    ColumnChunk* chunk = (ColumnChunk*)arena_alloc_aligned(sizeof(ColumnChunk));
    if (!chunk) return NULL;

    chunk->type = type;
    chunk->row_count = row_count;
    chunk->compression_flags = 0;
    chunk->min_val_i64 = INT64_MAX;
    chunk->max_val_i64 = INT64_MIN;
    chunk->seqlock = 0;

    if (pthread_rwlock_init(&chunk->rwlock, NULL) != 0) {
        arena_free_aligned(chunk);
        return NULL;
    }

    size_t bitmap_size = (row_count + 63) / 64 * sizeof(uint64_t);
    chunk->null_bitmap = (uint8_t*)arena_alloc_aligned(bitmap_size);
    if (!chunk->null_bitmap) {
        pthread_rwlock_destroy(&chunk->rwlock);
        arena_free_aligned(chunk);
        return NULL;
    }
    memset(chunk->null_bitmap, 0xFF, bitmap_size);

    size_t type_size = 4;
    if (type == DTYPE_INT64 || type == DTYPE_FLOAT64) {
        type_size = 8;
    }
    
    size_t data_size = row_count * type_size;
    chunk->data = (uint8_t*)arena_alloc_aligned(data_size);
    if (!chunk->data) {
        arena_free_aligned(chunk->null_bitmap);
        pthread_rwlock_destroy(&chunk->rwlock);
        arena_free_aligned(chunk);
        return NULL;
    }

    return chunk;
}

void vector_engine_destroy_chunk(ColumnChunk* chunk) {
    if (!chunk) return;
    pthread_rwlock_destroy(&chunk->rwlock);
    if (chunk->null_bitmap) arena_free_aligned(chunk->null_bitmap);
    if (chunk->data) arena_free_aligned(chunk->data);
    arena_free_aligned(chunk);
}

VectorBatch* vector_engine_create_batch(uint32_t column_count) {
    if (column_count > VECTOR_BATCH_SIZE) return NULL;
    VectorBatch* batch = (VectorBatch*)arena_alloc_aligned(sizeof(VectorBatch));
    if (!batch) return NULL;

    batch->column_count = column_count;
    batch->row_count = 0;
    batch->sel_vector.size = 0;
    batch->sel_vector.is_filtered = false;

    for (uint32_t i = 0; i < column_count; i++) {
        batch->data_ptrs[i] = arena_alloc_aligned(VECTOR_BATCH_SIZE * sizeof(uint64_t));
        if (!batch->data_ptrs[i]) {
            for (uint32_t j = 0; j < i; j++) {
                arena_free_aligned(batch->data_ptrs[j]);
            }
            arena_free_aligned(batch);
            return NULL;
        }
        batch->types[i] = DTYPE_INT64;
    }
    return batch;
}

void vector_engine_destroy_batch(VectorBatch* batch) {
    if (!batch) return;
    for (uint32_t i = 0; i < batch->column_count; i++) {
        if (batch->data_ptrs[i]) {
            arena_free_aligned(batch->data_ptrs[i]);
        }
    }
    arena_free_aligned(batch);
}

BitmapIndex* vector_engine_create_bitmap(uint32_t capacity_bits) {
    BitmapIndex* bm = (BitmapIndex*)arena_alloc_aligned(sizeof(BitmapIndex));
    if (!bm) return NULL;

    bm->word_count = (capacity_bits + 63) / 64;
    bm->capacity = capacity_bits;
    bm->words = (uint64_t*)arena_alloc_aligned(bm->word_count * sizeof(uint64_t));
    if (!bm->words) {
        arena_free_aligned(bm);
        return NULL;
    }
    memset(bm->words, 0, bm->word_count * sizeof(uint64_t));
    return bm;
}

void vector_engine_destroy_bitmap(BitmapIndex* bm) {
    if (!bm) return;
    if (bm->words) arena_free_aligned(bm->words);
    arena_free_aligned(bm);
}

int vector_engine_filter_int64(const int64_t* restrict data, uint32_t count, 
                               int64_t threshold, ComparisonOp op, 
                               SelectionVector* restrict sel) {
    if (!data || !sel) return -EINVAL;
    
    uint16_t match_count = 0;
    uint32_t i = 0;

    const uint32_t simd_limit = count & ~7u;

    switch (op) {
        case OP_EQ: {
            __m512i v_thresh = _mm512_set1_epi64(threshold);
            for (; i < simd_limit; i += 8) {
                __m512i v_data = _mm512_loadu_si512((const __m512i*)(data + i));
                __mmask8 mask = _mm512_cmpeq_epi64_mask(v_data, v_thresh);
                for (int bit = 0; bit < 8; bit++) {
                    if ((mask >> bit) & 1) {
                        if (match_count < VECTOR_BATCH_SIZE) {
                            sel->indices[match_count++] = (uint16_t)(i + bit);
                        }
                    }
                }
            }
            break;
        }
        case OP_NEQ: {
            __m512i v_thresh = _mm512_set1_epi64(threshold);
            for (; i < simd_limit; i += 8) {
                __m512i v_data = _mm512_loadu_si512((const __m512i*)(data + i));
                __mmask8 mask = _mm512_cmp_epi64_mask(v_data, v_thresh, _MM_CMPINT_NE);
                for (int bit = 0; bit < 8; bit++) {
                    if ((mask >> bit) & 1) {
                        if (match_count < VECTOR_BATCH_SIZE) {
                            sel->indices[match_count++] = (uint16_t)(i + bit);
                        }
                    }
                }
            }
            break;
        }
        case OP_LT: {
            __m512i v_thresh = _mm512_set1_epi64(threshold);
            for (; i < simd_limit; i += 8) {
                __m512i v_data = _mm512_loadu_si512((const __m512i*)(data + i));
                __mmask8 mask = _mm512_cmplt_epi64_mask(v_data, v_thresh);
                for (int bit = 0; bit < 8; bit++) {
                    if ((mask >> bit) & 1) {
                        if (match_count < VECTOR_BATCH_SIZE) {
                            sel->indices[match_count++] = (uint16_t)(i + bit);
                        }
                    }
                }
            }
            break;
        }
        case OP_LTE: {
            __m512i v_thresh = _mm512_set1_epi64(threshold);
            for (; i < simd_limit; i += 8) {
                __m512i v_data = _mm512_loadu_si512((const __m512i*)(data + i));
                __mmask8 mask = _mm512_cmple_epi64_mask(v_data, v_thresh);
                for (int bit = 0; bit < 8; bit++) {
                    if ((mask >> bit) & 1) {
                        if (match_count < VECTOR_BATCH_SIZE) {
                            sel->indices[match_count++] = (uint16_t)(i + bit);
                        }
                    }
                }
            }
            break;
        }
        case OP_GT: {
            __m512i v_thresh = _mm512_set1_epi64(threshold);
            for (; i < simd_limit; i += 8) {
                __m512i v_data = _mm512_loadu_si512((const __m512i*)(data + i));
                __mmask8 mask = _mm512_cmpgt_epi64_mask(v_data, v_thresh);
                for (int bit = 0; bit < 8; bit++) {
                    if ((mask >> bit) & 1) {
                        if (match_count < VECTOR_BATCH_SIZE) {
                            sel->indices[match_count++] = (uint16_t)(i + bit);
                        }
                    }
                }
            }
            break;
        }
        case OP_GTE: {
            __m512i v_thresh = _mm512_set1_epi64(threshold);
            for (; i < simd_limit; i += 8) {
                __m512i v_data = _mm512_loadu_si512((const __m512i*)(data + i));
                __mmask8 mask = _mm512_cmpge_epi64_mask(v_data, v_thresh);
                for (int bit = 0; bit < 8; bit++) {
                    if ((mask >> bit) & 1) {
                        if (match_count < VECTOR_BATCH_SIZE) {
                            sel->indices[match_count++] = (uint16_t)(i + bit);
                        }
                    }
                }
            }
            break;
        }
    }

    for (; i < count; i++) {
        bool match = false;
        int64_t val = data[i];
        switch (op) {
            case OP_EQ:  match = (val == threshold); break;
            case OP_NEQ: match = (val != threshold); break;
            case OP_LT:  match = (val < threshold);  break;
            case OP_LTE: match = (val <= threshold); break;
            case OP_GT:  match = (val > threshold);  break;
            case OP_GTE: match = (val >= threshold); break;
        }
        if (match && match_count < VECTOR_BATCH_SIZE) {
            sel->indices[match_count++] = (uint16_t)i;
        }
    }

    sel->size = match_count;
    sel->is_filtered = true;
    return 0;
}

int vector_engine_bitmap_and(const BitmapIndex* restrict a, 
                               const BitmapIndex* restrict b, 
                               BitmapIndex* restrict dest) {
    if (!a || !b || !dest) return -EINVAL;
    if (a->word_count != b->word_count || a->word_count != dest->word_count) return -EINVAL;

    uint32_t i = 0;
    uint32_t simd_limit = a->word_count & ~7u;

    for (; i < simd_limit; i += 8) {
        __m512i va = _mm512_loadu_si512((const __m512i*)&a->words[i]);
        __m512i vb = _mm512_loadu_si512((const __m512i*)&b->words[i]);
        __m512i vres = _mm512_and_si512(va, vb);
        _mm512_storeu_si512((__m512i*)&dest->words[i], vres);
    }

    for (; i < a->word_count; i++) {
        dest->words[i] = a->words[i] & b->words[i];
    }

    return 0;
}

int vector_engine_bitmap_or(const BitmapIndex* restrict a, 
                              const BitmapIndex* restrict b, 
                              BitmapIndex* restrict dest) {
    if (!a || !b || !dest) return -EINVAL;
    if (a->word_count != b->word_count || a->word_count != dest->word_count) return -EINVAL;

    uint32_t i = 0;
    uint32_t simd_limit = a->word_count & ~7u;

    for (; i < simd_limit; i += 8) {
        __m512i va = _mm512_loadu_si512((const __m512i*)&a->words[i]);
        __m512i vb = _mm512_loadu_si512((const __m512i*)&b->words[i]);
        __m512i vres = _mm512_or_si512(va, vb);
        _mm512_storeu_si512((__m512i*)&dest->words[i], vres);
    }

    for (; i < a->word_count; i++) {
        dest->words[i] = a->words[i] | b