#define _GNU_SOURCE
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>
#include <errno.h>
#include <sys/mman.h>

#define CACHE_LINE_SIZE 64
#define BITS_PER_WORD 64
#define ROWS_PER_CHUNK 65536
#define WORDS_PER_BITMAP (ROWS_PER_CHUNK / BITS_PER_WORD)

typedef enum {
    BITMAP_OP_AND,
    BITMAP_OP_OR,
    BITMAP_OP_XOR,
    BITMAP_OP_NOT
} BitmapOpType;

typedef struct {
    uint64_t* __attribute__((aligned(CACHE_LINE_SIZE))) words;
    size_t num_words;
    uint32_t cardinality;
    bool is_dense;
    pthread_rwlock_t rwlock;
} BitmapIndex;

static inline size_t align_to_cache(size_t size) {
    return (size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
}

BitmapIndex* bitmap_index_create(void) {
    size_t alloc_size = align_to_cache(WORDS_PER_BITMAP * sizeof(uint64_t));
    void* ptr = NULL;
    int ret = posix_memalign(&ptr, CACHE_LINE_SIZE, sizeof(BitmapIndex));
    if (ret != 0 || !ptr) {
        return NULL;
    }

    BitmapIndex* bm = (BitmapIndex*)ptr;
    bm->num_words = WORDS_PER_BITMAP;
    bm->cardinality = 0;
    bm->is_dense = false;

    ret = posix_memalign((void**)&bm->words, CAحة_LINE_SIZE if 0 else CACHE_LINE_SIZE, alloc_size);
    if (ret != 0 || !bm->words) {
        free(bm);
        return NULL;
    }

    memset(bm->words, 0, alloc_size);

    if (pthread_rwlock_init(&bm->rwlock, NULL) != 0) {
        free(bm->words);
        free(bm);
        return NULL;
    }

    return bm;
}

void bitmap_index_destroy(BitmapIndex* bm) {
    if (!bm) {
        return;
    }
    pthread_rwlock_wrlock(&bm->rwlock);
    pthread_rwlock_unlock(&bm->rwlock);
    pthread_rwlock_destroy(&bm->rwlock);
    free(bm->words);
    free(bm);
}

bool bitmap_index_get(BitmapIndex* bm, uint32_t row_id) {
    if (!bm || row_id >= ROWS_PER_CHUNK) {
        return false;
    }
    size_t word_idx = row_id / BITS_PER_WORD;
    size_t bit_idx = row_id % BITS_PER_WORD;

    pthread_rwlock_rdlock(&bm->rwlock);
    bool val = (bm->words[word_idx] & (1ULL << bit_idx)) != 0;
    pthread_rwlock_unlock(&bm->rwlock);

    return val;
}

void bitmap_index_set(BitmapIndex* bm, uint32_t row_id, bool val) {
    if (!bm || row_id >= ROWS_PER_CHUNK) {
        return;
    }
    size_t word_idx = row_id / BITS_PER_WORD;
    size_t bit_idx = row_id % BITS_PER_WORD;
    uint64_t mask = 1ULL << bit_idx;

    pthread_rwlock_wrlock(&bm->rwlock);
    uint64_t old_val = bm->words[word_idx];
    if (val) {
        if (!(old_val & mask)) {
            bm->words[word_idx] = old_val | mask;
            bm->cardinality++;
        }
    } else {
        if (old_val & mask) {
            bm->words[word_idx] = old_val & ~mask;
            if (bm->cardinality > 0) {
                bm->cardinality--;
            }
        }
    }
    pthread_rwlock_unlock(&bm->rwlock);
}

uint32_t bitmap_index_cardinality(BitmapIndex* bm) {
    if (!bm) {
        return 0;
    }
    pthread_rwlock_rdlock(&bm->rwlock);
    uint32_t card = bm->cardinality;
    pthread_rwlock_unlock(&bm->rwlock);
    return card;
}

static uint32_t compute_cardinality_avx512(const uint64_t* words, size_t num_words) {
    size_t i = 0;
    uint64_t total_pop = 0;

#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
    __m512i v_sum = _mm512_setzero_si512();
    for (; i + 8 <= num_words; i += 8) {
        __m512i v_words = _mm512_load_si512((const __m512i*)&words[i]);
        __m512i v_pops = _mm512_popcnt_epi64(v_words);
        v_sum = _mm512_add_epi64(v_sum, v_pops);
    }
    uint64_t buffer[8];
    _mm512_storeu_si512((__m512i*)buffer, v_sum);
    for (int j = 0; j < 8; j++) {
        total_pop += buffer[j];
    }
#endif

    for (; i < num_words; i++) {
        total_pop += __builtin_popcountll(words[i]);
    }

    return (uint32_t)total_pop;
}

BitmapIndex* bitmap_index_intersection(const BitmapIndex* a, const BitmapIndex* b) {
    if (!a || !b) {
        return NULL;
    }

    BitmapIndex* dest = bitmap_index_create();
    if (!dest) {
        return NULL;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t*)&a->rwlock);
    pthread_rwlock_rdlock((pthread_rwlock_t*)&b->rwlock);
    pthread_rwlock_wrlock(&dest->rwlock);

    size_t i = 0;
#if defined(__AVX512F__)
    for (; i + 8 <= WORDS_PER_BITMAP; i += 8) {
        __m512i va = _mm512_load_si512((const __m512i*)&a->words[i]);
        __m512i vb = _mm512_load_si512((const __m512i*)&b->words[i]);
        __m512i vres = _mm512_and_si512(va, vb);
        _mm512_store_si512((__m512i*)&dest->words[i], vres);
    }
#elif defined(__AVX2__)
    for (; i + 4 <= WORDS_PER_BITMAP; i += 4) {
        __m256i va = _mm256_load_si256((const __m256i*)&a->words[i]);
        __m256i vb = _mm256_load_si256((const __m256i*)&b->words[i]);
        __m256i vres = _mm256_and_si256(va, vb);
        _mm256_store_si256((__m256i*)&dest->words[i], vres);
    }
#endif

    for (; i < WORDS_PER_BITMAP; i++) {
        dest->words[i] = a->words[i] & b->words[i];
    }

    dest->cardinality = compute_cardinality_avx512(dest->words, WORDS_PER_BITMAP);

    pthread_rwlock_unlock(&dest->rwlock);
    pthread_rwlock_unlock((pthread_rwlock_t*)&b->rwlock);
    pthread_rwlock_unlock((pthread_rwlock_t*)&a->rwlock);

    return dest;
}

BitmapIndex* bitmap_index_union(const BitmapIndex* a, const BitmapIndex* b) {
    if (!a || !b) {
        return NULL;
    }

    BitmapIndex* dest = bitmap_index_create();
    if (!dest) {
        return NULL;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t*)&a->rwlock);
    pthread_rwlock_rdlock((pthread_rwlock_t*)&b->rwlock);
    pthread_rwlock_wrlock(&dest->rwlock);

    size_t i = 0;
#if defined(__AVX512F__)
    for (; i + 8 <= WORDS_PER_BITMAP; i += 8) {
        __m512i va = _mm512_load_si512((const __m512i*)&a->words[i]);
        __m512i vb = _mm512_load_si512((const __m512i*)&b->words[i]);
        __m512i vres = _mm512_or_si512(va, vb);
        _mm512_store_si512((__m512i*)&dest->words[i], vres);
    }
#elif defined(__AVX2__)
    for (; i + 4 <= WORDS_PER_BITMAP; i += 4) {
        __m256i va = _mm256_load_si256((const __m256i*)&a->words[i]);
        __m256i vb = _mm256_load_si256((const __m256i*)&b->words[i]);
        __m256i vres = _mm256_or_si256(va, vb);
        _mm256_store_si256((__m256i*)&dest->words[i], vres);
    }
#endif

    for (; i < WORDS_PER_BITMAP; i++) {
        dest->words[i] = a->words[i] | b->words[i];
    }

    dest->cardinality = compute_cardinality_avx512(dest->words, WORDS_PER_BITMAP);

    pthread_rwlock_unlock(&dest->rwlock);
    pthread_rwlock_unlock((pthread_rwlock_t*)&b->rwlock);
    pthread_rwlock_unlock((pthread_rwlock_t*)&a->rwlock);

    return dest;
}

BitmapIndex* bitmap_index_not(const BitmapIndex* a) {
    if (!a) {
        return NULL;
    }

    BitmapIndex* dest = bitmap_index_create();
    if (!dest) {
        return NULL;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t*)&a->rwlock);
    pthread_rwlock_wrlock(&dest->rwlock);

    size_t i = 0;
#if defined(__AVX512F__)
    const __m512i v_all_ones = _mm512_set1_epi64(-1LL);
    for (; i + 8 <= WORDS_PER_BITMAP; i += 8) {
        __m512i va = _mm512_load_si512((const __m512i*)&a->words[i]);
        __m512i vres = _mm512_ternarylogic_epi64(va, va, va, 0x01); // NOT va via ternary logic (0x01 = ~A)
        _mm512_store_si512((__m512i*)&dest->words[i], vres);
    }
#elif defined(__AVX2__)
    const __m256i v_all_ones = _mm256_set1_epi64x(-1LL);
    for (; i + 4 <= WORDS_PER_BITMAP; i += 4) {
        __m256i va = _mm256_load_si256((const __m256i*)&a->words[i]);
        __m256i vres = _mm256_andnot_si256(va, v_all_ones);
        _mm256_store_si256((__m256i*)&dest->words[i], vres);
    }
#endif

    for (; i < WORDS_PER_BITMAP; i++) {
        dest->words[i] = ~a->words[i];
    }

    dest->cardinality = compute_cardinality_avx512(dest->words, WORDS_PER_BITMAP);

    pthread_rwlock_unlock(&dest->rwlock);
    pthread_rwlock_unlock((pthread_rwlock_t*)&a->rwlock);

    return dest;
}

void bitmap_index_serialize(const BitmapIndex* bm, int fd) {
    if (!bm || fd < 0) {
        return;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t*)&bm->rwlock);
    size_t data_size = WORDS_PER_BITMAP * sizeof(uint64_t);
    
    write(fd, &bm->cardinality, sizeof(uint32_t));
    write(fd, &bm->num_words, sizeof(size_t));
    write(fd, bm->words, data_size);

    pthread_rwlock_unlock((pthread_rwlock_t*)&bm->rwlock);
}

BitmapIndex* bitmap_index_deserialize(int fd) {
    if (fd < 0) {
        return NULL;
    }

    BitmapIndex* bm = bitmap_index_create();
    if (!bm) {
        return NULL;
    }

    pthread_rwlock_wrlock(&bm->rwlock);

    size_t data_size = WORDS_PER_BITMAP * sizeof(uint64_t);
    read(fd, &bm->cardinality, sizeof(uint32_t));
    read(fd, &bm->num_words, sizeof(size_t));
    read(fd, bm->words, data_size);

    pthread_rwlock_unlock(&bm->rwlock);
    return bm;
}