#ifndef VECTOR_DB_TYPES_H
#define VECTOR_DB_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <pthread.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VDB_CACHE_LINE_SIZE 64
#define VDB_DEFAULT_CHUNK_ROWS 65536
#define VDB_VECTOR_BATCH_SIZE 64
#define VDB_MAX_STRING_LEN 256

typedef enum {
    VDB_OK = 0,
    VDB_ERR_NOMEM = -1,
    VDB_ERR_INVAL = -2,
    VDB_ERR_OUT_OF_BOUNDS = -3,
    VDB_ERR_CORRUPT = -4,
    VDB_ERR_IO = -5,
    VDB_ERR_LOCKED = -6,
    VDB_ERR_NOT_FOUND = -7
} vdb_status_t;

typedef enum {
    VDB_TYPE_INT8 = 1,
    VDB_TYPE_INT16 = 2,
    VDB_TYPE_INT32 = 3,
    VDB_TYPE_INT64 = 4,
    VDB_TYPE_FLOAT32 = 5,
    VDB_TYPE_FLOAT64 = 6,
    VDB_TYPE_VECTOR_FLOAT32 = 7,
    VDB_TYPE_DICTIONARY = 8,
    VDB_TYPE_BOOLEAN = 9
} vdb_data_type_t;

typedef enum {
    VDB_COMPRESSION_NONE = 0,
    VDB_COMPRESSION_LZ4 = 1,
    VDB_COMPRESSION_ZSTD = 2,
    VDB_COMPRESSION_BITPACK = 3
} vdb_compression_codec_t;

typedef union {
    int8_t i8;
    int16_t i16;
    int32_t i32;
    int64_t i64;
    float f32;
    double f64;
    char raw_bytes[8];
} vdb_scalar_value_t;

typedef struct {
    vdb_data_type_t type;
    vdb_scalar_value_t min_val;
    vdb_scalar_value_t max_val;
    bool has_nulls;
} __attribute__((aligned(VDB_CACHE_LINE_SIZE))) vdb_zone_map_t;

typedef struct {
    uint32_t version;
    uint32_t row_count;
    vdb_data_type_t data_type;
    vdb_compression_codec_t codec;
    vdb_zone_map_t zone_map;
    size_t data_offset;
    size_t data_size;
    size_t null_bitmap_offset;
    size_t null_bitmap_size;
    uint32_t checksum;
    bool is_immutable;
} __attribute__((aligned(VDB_CACHE_LINE_SIZE))) vdb_column_chunk_metadata_t;

typedef struct {
    vdb_column_chunk_metadata_t metadata;
    uint64_t seqlock;
    uint8_t* raw_memory;
    size_t allocated_size;
    uint8_t* null_bitmap;
    uint8_t* data_vector;
    pthread_rwlock_t rwlock;
} __attribute__((aligned(VDB_CACHE_LINE_SIZE))) vdb_column_chunk_t;

typedef struct {
    uint32_t hash;
    uint32_t dict_index;
} __attribute__((aligned(VDB_CACHE_LINE_SIZE))) vdb_dict_hash_entry_t;

typedef struct {
    vdb_data_type_t value_type;
    uint32_t capacity;
    uint32_t size;
    void* dense_keys;
    vdb_dict_hash_entry_t* hash_index;
    uint32_t hash_mask;
    pthread_rwlock_t lock;
    uint8_t* arena_mem;
    size_t arena_size;
} __attribute__((aligned(VDB_CACHE_LINE_SIZE))) vdb_dictionary_t;

typedef struct {
    uint32_t word_count;
    uint64_t* words;
    bool is_compressed;
    size_t compressed_size;
    uint8_t* compressed_data;
} __attribute__((aligned(VDB_CACHE_LINE_SIZE))) vdb_bitmap_index_t;

typedef struct {
    uint32_t size;
    uint16_t indices[VDB_VECTOR_BATCH_SIZE];
} __attribute__((aligned(VDB_CACHE_LINE_SIZE))) vdb_selection_vector_t;

typedef struct {
    uint32_t row_count;
    vdb_data_type_t types[VDB_VECTOR_BATCH_SIZE];
    void* columns[VDB_VECTOR_BATCH_SIZE];
    vdb_selection_vector_t selection;
    bool use_selection;
} __attribute__((aligned(VDB_CACHE_LINE_SIZE))) vdb_vector_batch_t;

typedef struct vdb_arena_block {
    struct vdb_arena_block* next;
    size_t capacity;
    size_t used;
    uint8_t memory[];
} vdb_arena_block_t;

typedef struct {
    size_t block_size;
    vdb_arena_block_t* current_block;
    pthread_mutex_t lock;
    size_t total_allocated;
} __attribute__((aligned(VDB_CACHE_LINE_SIZE))) vdb_arena_t;

typedef struct {
    uint64_t epoch;
    vdb_arena_t* local_arena;
    bool active;
} __attribute__((aligned(VDB_CACHE_LINE_SIZE))) vdb_thread_context_t;

typedef struct {
    uint64_t global_epoch;
    pthread_rwlock_t epoch_lock;
    uint32_t active_threads;
} __attribute__((aligned(VDB_CACHE_LINE_SIZE))) vdb_epoch_gc_t;

typedef struct {
    uint32_t magic;
    uint32_t crc32;
    uint64_t lsn;
    uint32_t payload_size;
    uint32_t op_type;
} __attribute__((aligned(VDB_CACHE_LINE_SIZE))) vdb_wal_header_t;

typedef struct {
    int fd;
    uint64_t current_lsn;
    char path[1024];
    pthread_mutex_t lock;
} vdb_wal_t;

vdb_status_t vdb_column_chunk_init(vdb_column_chunk_t* chunk, vdb_data_type_t type, uint32_t row_capacity, vdb_compression_codec_t codec);
void vdb_column_chunk_free(vdb_column_chunk_t* chunk);
vdb_status_t vdb_column_chunk_append(vdb_column_chunk_t* chunk, const void* data, uint32_t count);
vdb_status_t vdb_column_chunk_seal(vdb_column_chunk_t* chunk);

vdb_status_t vdb_dictionary_init(vdb_dictionary_t* dict, vdb_data_type_t type, uint32_t capacity);
void vdb_dictionary_free(vdb_dictionary_t* dict);
vdb_status_t vdb_dictionary_lookup_or_insert(vdb_dictionary_t* dict, const void* value, uint32_t* out_index);
int32_t vdb_dictionary_lookup(vdb_dictionary_t* dict, const void* value);

vdb_status_t vdb_bitmap_index_init(vdb_bitmap_index_t* bitmap, uint32_t total_bits);
void vdb_bitmap_index_free(vdb_bitmap_index_t* bitmap);
vdb_status_t vdb_bitmap_set(vdb_bitmap_index_t* bitmap, uint32_t bit_index, bool value);
bool vdb_bitmap_get(const vdb_bitmap_index_t* bitmap, uint32_t bit_index);
vdb_status_t vdb_bitmap_and(vdb_bitmap_index_t* dest, const vdb_bitmap_index_t* src1, const vdb_bitmap_index_t* src2);
vdb_status_t vdb_bitmap_or(vdb_bitmap_index_t* dest, const vdb_bitmap_index_t* src1, const vdb_bitmap_index_t* src2);
vdb_status_t vdb_bitmap_not(vdb_bitmap_index_t* dest, const vdb_bitmap_index_t* src);

vdb_status_t vdb_vector_batch_init(vdb_vector_batch_t* batch, uint32_t row_count);
void vdb_vector_batch_free(vdb_vector_batch_t* batch);

vdb_status_t vdb_arena_init(vdb_arena_t* arena, size_t block_size);
void* vdb_arena_alloc(vdb_arena_t* arena, size_t size);
void vdb_arena_reset(vdb_arena_t* arena);
void vdb_arena_free(vdb_arena_t* arena);

vdb_status_t vdb_epoch_gc_init(vdb_epoch_gc_t* gc);
uint64_t vdb_epoch_enter(vdb_epoch_gc_t* gc);
void vdb_epoch_exit(vdb_epoch_gc_t* gc, uint64_t epoch);
void vdb_epoch_collect(vdb_epoch_gc_t* gc);

vdb_status_t vdb_wal_init(vdb_wal_t* wal, const char* filepath);
vdb_status_t vdb_wal_append(vdb_wal_t* wal, uint32_t op_type, const void* data, uint32_t size, uint64_t* out_lsn);
vdb_status_t vdb_wal_recover(vdb_wal_t* wal, void (*callback)(uint32_t op_type, const void* data, uint32_t size));
void vdb_wal_close(vdb_wal_t* wal);

#ifdef __cplusplus
}
#endif

#endif

#define _GNU_SOURCE
#include "include/vector_db/types.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>

static size_t vdb_type_size(vdb_data_type_t type) {
    switch (type) {
        case VDB_TYPE_INT8:
        case VDB_TYPE_BOOLEAN: return 1;
        case VDB_TYPE_INT16: return 2;
        case VDB_TYPE_INT32:
        case VDB_TYPE_FLOAT32: return 4;
        case VDB_TYPE_INT64:
        case VDB_TYPE_FLOAT64: return 8;
        case VDB_TYPE_VECTOR_FLOAT32: return 4 * 64;
        case VDB_TYPE_DICTIONARY: return 4;
        default: return 8;
    }
}

static uint32_t vdb_hash_bytes(const void* buf, size_t len) {
    const uint8_t* data = (const uint8_t*)buf;
    uint32_t hash = 2166136261u;
    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= 16777619;
    }
    return hash;
}

vdb_status_t vdb_column_chunk_init(vdb_column_chunk_t* chunk, vdb_data_type_t type, uint32_t row_capacity, vdb_compression_codec_t codec) {
    if (!chunk) return VDB_ERR_INVAL;

    size_t type_sz = vdb_type_size(type);
    size_t data_sz = type_sz * row_capacity;
    size_t bitmap_sz = (row_capacity + 63) / 64 * sizeof(uint64_t);
    size_t total_sz = (data_sz + bitmap_sz + VDB_CACHE_LINE_SIZE - 1) & ~(VDB_CACHE_LINE_SIZE - 1);

    chunk->raw_memory = (uint8_t*)mmap(NULL, total_sz + VDB_CACHE_LINE_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (chunk->raw_memory == MAP_FAILED) {
        return VDB_ERR_NOMEM;
    }

    chunk->allocated_size = total_sz + VDB_CACHE_LINE_SIZE;
    chunk->null_bitmap = chunk->raw_memory;
    chunk->data_vector = chunk->raw_memory + bitmap_sz;
    
    memset(chunk->raw_memory, 0, total_sz);

    chunk->metadata.version = 1;
    chunk->metadata.row_count = 0;
    chunk->metadata.data_type = type;
    chunk->metadata.codec = codec;
    chunk->metadata.data_offset = (size_t)(chunk->data_vector - chunk->raw_memory);
    chunk->metadata.data_size = data_sz;
    chunk->metadata.null_bitmap_offset = 0;
    chunk->metadata.null_bitmap_size = bitmap_sz;
    chunk->metadata.checksum = 0;
    chunk->metadata.is_immutable = false;
    chunk->metadata.zone_map.type = type;
    chunk->metadata.zone_map.has_nulls = false;

    chunk->seqlock = 0;
    pthread_rwlock_init(&chunk->rwlock, NULL);

    return VDB_OK;
}

void vdb_column_chunk_free(vdb_column_chunk_t* chunk) {
    if (!chunk || !chunk->raw_memory) return;
    if (chunk->metadata.is_immutable) {
        mprotect(chunk->raw_memory, chunk->allocated_size, PROT_READ | PROT_WRITE);
    }
    munmap(chunk->raw_memory, chunk->allocated_size);
    pthread_rwlock_destroy(&chunk->rwlock);
    memset(chunk, 0, sizeof(vdb_column_chunk_t));
}

vdb_status_t vdb_column_chunk_append(vdb_column_chunk_t* chunk, const void* data, uint32_t count) {
    if (!chunk || !data) return VDB_ERR_INVAL;
    if (chunk->metadata.is_immutable) return VDB_ERR_LOCKED;

    pthread_rwlock_wrlock(&chunk->rwlock);
    chunk->seqlock++;

    size_t type_sz = vdb_type_size(chunk->metadata.data_type);
    size_t offset = chunk->metadata.row_count * type_sz;

    if ((chunk->metadata.row_count + count) * type_sz > chunk->metadata.data_size) {
        pthread_rwlock_unlock(&chunk->rwlock);
        return VDB_ERR_OUT_OF_BOUNDS;
    }

    memcpy(chunk->data_vector + offset, data, count * type_sz);
    chunk->metadata.row_count += count;

    chunk->seqlock++;
    pthread_rwlock_unlock(&chunk->rwlock);

    return VDB_OK;
}

vdb_status_t vdb_column_chunk_seal(vdb_column_chunk_t* chunk) {
    if (!chunk) return VDB_ERR_INVAL;

    pthread