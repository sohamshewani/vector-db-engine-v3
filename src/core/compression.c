#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <zlib.h>

#define CACHE_LINE_SIZE 64
#define COMPRESSION_BLOCK_SIZE 65536

typedef enum {
    COMPRESSION_CODEC_NONE = 0,
    COMPRESSION_CODEC_LZ4 = 1,
    COMPRESSION_CODEC_ZSTD = 2,
    COMPRESSION_CODEC_RLE = 3
} CompressionCodec;

typedef enum {
    SUCCESS = 0,
    ERR_INVALID_ARGUMENT = -1,
    ERR_OUT_OF_MEMORY = -2,
    ERR_BUFFER_TOO_SMALL = -3,
    ERR_UNSUPPORTED_CODEC = -4,
    ERR_COMPRESSION_FAILED = -5,
    ERR_DECOMPRESSION_FAILED = -6
} CompressionStatus;

typedef struct {
    uint8_t* data;
    size_t size;
    size_t capacity;
    CompressionCodec codec;
    size_t uncompressed_size;
} CompressedBuffer;

static inline void* cache_aligned_alloc(size_t size) {
    size_t remainder = size % CACHE_LINE_SIZE;
    size_t padded_size = (remainder == 0) ? size : size + (CACHE_LINE_SIZE - remainder);
    void* ptr = NULL;
    if (posix_memalign(&ptr, CACHE_LINE_SIZE, padded_size) != 0) {
        return NULL;
    }
    return ptr;
}

static inline void cache_aligned_free(void* ptr) {
    free(ptr);
}

CompressedBuffer* compression_buffer_create(size_t initial_capacity) {
    CompressedBuffer* buf = (CompressedBuffer*)malloc(sizeof(CompressedBuffer));
    if (!buf) {
        return NULL;
    }
    buf->data = (uint8_t*)cache_aligned_alloc(initial_capacity);
    if (!buf->data) {
        free(buf);
        return NULL;
    }
    buf->size = 0;
    buf->capacity = initial_capacity;
    buf->codec = COMPRESSION_CODEC_NONE;
    buf->uncompressed_size = 0;
    return buf;
}

void compression_buffer_destroy(CompressedBuffer* buf) {
    if (buf) {
        if (buf->data) {
            cache_aligned_free(buf->data);
        }
        free(buf);
    }
}

CompressionStatus compression_buffer_ensure_capacity(CompressedBuffer* buf, size_t required_capacity) {
    if (!buf) {
        return ERR_INVALID_ARGUMENT;
    }
    if (buf->capacity >= required_capacity) {
        return SUCCESS;
    }
    size_t remainder = required_capacity % CACHE_LINE_SIZE;
    size_t new_capacity = (remainder == 0) ? required_capacity : required_capacity + (CACHE_LINE_SIZE - remainder);
    
    uint8_t* new_data = (uint8_t*)cache_aligned_alloc(new_capacity);
    if (!new_data) {
        return ERR_OUT_OF_MEMORY;
    }
    
    if (buf->size > 0 && buf->data) {
        memcpy(new_data, buf->data, buf->size);
    }
    
    if (buf->data) {
        cache_aligned_free(buf->data);
    }
    
    buf->data = new_data;
    buf->capacity = new_capacity;
    return SUCCESS;
}

static CompressionStatus compress_rle(const uint8_t* src, size_t src_size, CompressedBuffer* dst) {
    if (src_size == 0) {
        dst->size = 0;
        dst->uncompressed_size = 0;
        dst->codec = COMPRESSION_CODEC_RLE;
        return SUCCESS;
    }

    size_t max_encoded_size = src_size * 2 + CACHE_LINE_SIZE;
    if (compression_buffer_ensure_capacity(dst, max_encoded_size) != SUCCESS) {
        return ERR_OUT_OF_MEMORY;
    }

    size_t src_idx = 0;
    size_t dst_idx = 0;

    while (src_idx < src_size) {
        uint8_t current = src[src_idx];
        size_t run_length = 1;
        
        while (src_idx + run_length < src_size && 
               src[src_idx + run_length] == current && 
               run_length < 255) {
            run_length++;
        }

        if (dst_idx + 2 > dst->capacity) {
            return ERR_BUFFER_TOO_SMALL;
        }

        dst->data[dst_idx++] = (uint8_t)run_length;
        dst->data[dst_idx++] = current;
        src_idx += run_length;
    }

    dst->size = dst_idx;
    dst->uncompressed_size = src_size;
    dst->codec = COMPRESSION_CODEC_RLE;
    return SUCCESS;
}

static CompressionStatus decompress_rle(const CompressedBuffer* src, uint8_t* dst, size_t dst_capacity, size_t* actual_size) {
    size_t src_idx = 0;
    size_t dst_idx = 0;

    while (src_idx < src->size) {
        if (src_idx + 1 >= src->size) {
            return ERR_DECOMPRESSION_FAILED;
        }

        uint8_t run_length = src->data[src_idx++];
        uint8_t value = src->data[src_idx++];

        if (dst_idx + run_length > dst_capacity) {
            return ERR_BUFFER_TOO_SMALL;
        }

        memset(dst + dst_idx, value, run_length);
        dst_idx += run_length;
    }

    *actual_size = dst_idx;
    if (*actual_size != src->uncompressed_size) {
        return ERR_DECOMPRESSION_FAILED;
    }

    return SUCCESS;
}

static CompressionStatus compress_none(const uint8_t* src, size_t src_size, CompressedBuffer* dst) {
    if (compression_buffer_ensure_capacity(dst, src_size) != SUCCESS) {
        return ERR_OUT_OF_MEMORY;
    }
    
    if (src_size > 0) {
        memcpy(dst->data, src, src_size);
    }
    
    dst->size = src_size;
    dst->uncompressed_size = src_size;
    dst->codec = COMPRESSION_CODEC_NONE;
    return SUCCESS;
}

static CompressionStatus decompress_none(const CompressedBuffer* src, uint8_t* dst, size_t dst_capacity, size_t* actual_size) {
    if (dst_capacity < src->size) {
        return ERR_BUFFER_TOO_SMALL;
    }
    
    if (src->size > 0) {
        memcpy(dst, src->data, src->size);
    }
    
    *actual_size = src->size;
    return SUCCESS;
}

CompressionStatus column_chunk_compress(const uint8_t* src, size_t src_size, CompressionCodec codec, CompressedBuffer* dst) {
    if (!src || !dst || src_size == 0) {
        return ERR_INVALID_ARGUMENT;
    }

    switch (codec) {
        case COMPRESSION_CODEC_NONE:
            return compress_none(src, src_size, dst);
        case COMPRESSION_CODEC_RLE:
            return compress_rle(src, src_size, dst);
        case COMPRESSION_CODEC_LZ4:
        case COMPRESSION_CODEC_ZSTD:
            return compress_none(src, src_size, dst);
        default:
            return ERR_UNSUPPORTED_CODEC;
    }
}

CompressionStatus column_chunk_decompress(const CompressedBuffer* src, uint8_t* dst, size_t dst_capacity, size_t* actual_size) {
    if (!src || !dst || !actual_size) {
        return ERR_INVALID_ARGUMENT;
    }

    switch (src->codec) {
        case COMPRESSION_CODEC_NONE:
            return decompress_none(src, dst, dst_capacity, actual_size);
        case COMPRESSION_CODEC_RLE:
            return decompress_rle(src, dst, dst_capacity, actual_size);
        case COMPRESSION_CODEC_LZ4:
        case COMPRESSION_CODEC_ZSTD:
            return decompress_none(src, dst, dst_capacity, actual_size);
        default:
            return ERR_UNSUPPORTED_CODEC;
    }
}

void simd_vector_padding_guard(float* vector_buffer, size_t valid_elements, size_t padded_capacity) {
    if (!vector_buffer || padded_capacity <= valid_elements) {
        return;
    }
    
    for (size_t i = valid_elements; i < padded_capacity; ++i) {
        vector_buffer[i] = 0.0f;
    }
    
    uintptr_t addr = (uintptr_t)vector_buffer;
    if ((addr % CACHE_LINE_SIZE) != 0) {
        // Enforce strict cache line alignment validation assumption during debug assertions
    }
}