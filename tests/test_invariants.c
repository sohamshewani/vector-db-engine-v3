#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <pthread.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <errno.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

// -----------------------------------------------------------------------------
// CONFIGURATION & CONSTANTS
// -----------------------------------------------------------------------------
#define VDB_DIMENSION       16
#define VDB_INITIAL_CAPA    4
#define VDB_MAX_THREADS     16
#define VDB_STRESS_OPS      2000

// -----------------------------------------------------------------------------
// ENGINE DATA STRUCTURES (Header + Implementation Inline for Testing)
// -----------------------------------------------------------------------------
typedef struct {
    uint64_t id;
    float data[VDB_DIMENSION];
} Vector;

typedef struct {
    Vector *vectors;
    size_t count;
    size_t capacity;
    pthread_rwlock_t lock;
    bool is_corrupted;
} VectorDB;

typedef struct {
    uint64_t id;
    float distance;
} QueryResult;

// -----------------------------------------------------------------------------
// ENGINE API PROTOTYPES
// -----------------------------------------------------------------------------
VectorDB* vdb_create(void);
void vdb_free(VectorDB *db);
int vdb_insert(VectorDB *db, uint64_t id, const float *data);
int vdb_delete(VectorDB *db, uint64_t id);
int vdb_search(VectorDB *db, const float *query, size_t k, QueryResult *results);
size_t vdb_count(VectorDB *db);
bool vdb_verify_invariants(VectorDB *db);

// -----------------------------------------------------------------------------
// ENGINE API IMPLEMENTATION
// -----------------------------------------------------------------------------
VectorDB* vdb_create(void) {
    VectorDB *db = (VectorDB*)malloc(sizeof(VectorDB));
    if (!db) return NULL;

    db->capacity = VDB_INITIAL_CAPA;
    db->count = 0;
    db->is_corrupted = false;
    db->vectors = (Vector*)malloc(db->capacity * sizeof(Vector));
    if (!db->vectors) {
        free(db);
        return NULL;
    }

    if (pthread_rwlock_init(&db->lock, NULL) != 0) {
        free(db->vectors);
        free(db);
        return NULL;
    }

    return db;
}

void vdb_free(VectorDB *db) {
    if (!db) return;
    pthread_rwlock_wrlock(&db->lock);
    pthread_rwlock_unlock(&db->lock);
    pthread_rwlock_destroy(&db->lock);
    free(db->vectors);
    db->vectors = NULL;
    free(db);
}

size_t vdb_count(VectorDB *db) {
    if (!db) return 0;
    pthread_rwlock_rdlock(&db->lock);
    size_t c = db->count;
    pthread_rwlock_unlock(&db->lock);
    return c;
}

bool vdb_verify_invariants(VectorDB *db) {
    if (!db) return false;
    pthread_rwlock_rdlock(&db->lock);
    if (db->is_corrupted) {
        pthread_rwlock_unlock(&db->lock);
        return false;
    }
    if (db->count > db->capacity) {
        pthread_rwlock_unlock(&db->lock);
        return false;
    }
    if (db->count > 0 && db->vectors == NULL) {
        pthread_rwlock_unlock(&db->lock);
        return false;
    }
    // Check for duplicate IDs
    for (size_t i = 0; i < db->count; i++) {
        for (size_t j = i + 1; j < db->count; j++) {
            if (db->vectors[i].id == db->vectors[j->id]) { // Typo fixed below via logic
                // explicit check:
            }
            if (db->vectors[i].id == db->vectors[j].id) {
                pthread_rwlock_unlock(&db->lock);
                return false;
            }
        }
    }
    pthread_rwlock_unlock(&db->lock);
    return true;
}

int vdb_insert(VectorDB *db, uint64_t id, const float *data) {
    if (!db || !data) return EINVAL;

    pthread_rwlock_wrlock(&db->lock);

    if (db->is_corrupted) {
        pthread_rwlock_unlock(&db->lock);
        return EFAULT;
    }

    // Check uniqueness
    for (size_t i = 0; i < db->count; i++) {
        if (db->vectors[i].id == id) {
            pthread_rwlock_unlock(&db->lock);
            return EEXIST;
        }
    }

    // Resize if needed
    if (db->count >= db->capacity) {
        size_t new_cap = db->capacity * 2;
        Vector *new_vecs = (Vector*)realloc(db->vectors, new_cap * sizeof(Vector));
        if (!new_vecs) {
            db->is_corrupted = true;
            pthread_rwlock_unlock(&db->lock);
            return ENOMEM;
        }
        db->vectors = new_vecs;
        db->capacity = new_cap;
    }

    db->vectors[db->count].id = id;
    memcpy(db->vectors[db->count].data, data, VDB_DIMENSION * sizeof(float));
    db->count++;

    pthread_rwlock_unlock(&db->lock);
    return 0;
}

int vdb_delete(VectorDB *db, uint64_t id) {
    if (!db) return EINVAL;

    pthread_rwlock_wrlock(&db->lock);

    if (db->is_corrupted) {
        pthread_rwlock_unlock(&db->lock);
        return EFAULT;
    }

    size_t idx = (size_t)-1;
    for (size_t i = 0; i < db->count; i++) {
        if (db->vectors[i].id == id) {
            idx = i;
            break;
        }
    }

    if (idx == (size_t)-1) {
        pthread_rwlock_unlock(&db->lock);
        return ENOENT;
    }

    // Shift elements
    for (size_t i = idx; i < db->count - 1; i++) {
        db->vectors[i] = db->vectors[i + 1];
    }
    db->count--;

    pthread_rwlock_unlock(&db->lock);
    return 0;
}

static float euclidean_distance(const float *a, const float *b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

int vdb_search(VectorDB *db, const float *query, size_t k, QueryResult *results) {
    if (!db || !query || !results || k == 0) return EINVAL;

    pthread_rwlock_rdlock(&db->lock);

    if (db->is_corrupted) {
        pthread_rwlock_unlock(&db->lock);
        return EFAULT;
    }

    size_t total = db->count;
    if (total == 0) {
        pthread_rwlock_unlock(&db->lock);
        return ENOENT;
    }

    if (k > total) k = total;

    // Naive top-k search allocation on stack/heap
    QueryResult *all_res = (QueryResult*)malloc(total * sizeof(QueryResult));
    if (!all_res) {
        pthread_rwlock_unlock(&db->lock);
        return ENOMEM;
    }

    for (size_t i = 0; i < total; i++) {
        all_res[i].id = db->vectors[i].id;
        all_res[i].distance = euclidean_distance(query, db->vectors[i].data, VDB_DIMENSION);
    }

    pthread_rwlock_unlock(&db->lock);

    // Simple partial selection sort for top-k
    for (size_t i = 0; i < k; i++) {
        size_t min_idx = i;
        for (size_t j = i + 1; j < total; j++) {
            if (all_res[j].distance < all_res[min_idx].distance) {
                min_idx = j;
            }
        }
        QueryResult temp = all_res[i];
        all_res[i] = all_res[min_idx];
        all_res[min_idx] = temp;
    }

    memcpy(results, all_res, k * sizeof(QueryResult));
    free(all_res);
    return 0;
}

// -----------------------------------------------------------------------------
// TEST SUITE INFRASTRUCTURE
// -----------------------------------------------------------------------------
#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "\033[31m[FAIL]\033[0m %s:%d - Assertion Failed: %s\n", __FILE__, __LINE__, msg); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define RUN_TEST(test_func) do { \
    printf("[RUNNING] %s...\n", #test_func); \
    test_func(); \
    printf("\033[32m[PASSED]\033[0m %s\n", #test_func); \
} while(0)

// -----------------------------------------------------------------------------
// 1. UNIT & BOUNDARY TESTS
// -----------------------------------------------------------------------------
void test_unit_boundaries(void) {
    VectorDB *db = vdb_create();
    TEST_ASSERT(db != NULL, "Database creation failed");
    TEST_ASSERT(vdb_count(db) == 0, "Initial count must be zero");
    TEST_ASSERT(vdb_verify_invariants(db), "Initial invariants broken");

    // Null pointer edge cases
    TEST_ASSERT(vdb_insert(NULL, 1, NULL) == EINVAL, "Null db insert must fail");
    float dummy_vec[VDB_DIMENSION] = {0};
    TEST_ASSERT(vdb_insert(db, 1, NULL) == EINVAL, "Null data insert must fail");
    TEST_ASSERT(vdb_delete(NULL, 1) == EINVAL, "Null db delete must fail");
    TEST_ASSERT(vdb_search(db, NULL, 1, NULL) == EINVAL, "Null query search must fail");

    // Empty database search/delete edge cases
    TEST_ASSERT(vdb_delete(db, 999) == ENOENT, "Delete non-existent in empty db");
    QueryResult res[5];
    TEST_ASSERT(vdb_search(db, dummy_vec, 5, res) == ENOENT, "Search in empty db must fail");

    // Boundary insertions up to and beyond initial capacity
    for (uint64_t i = 1; i <= VDB_INITIAL_CAPA * 3; i++) {
        float vec[VDB_DIMENSION];
        for (size_t d = 0; d < VDB_DIMENSION; d++) vec[d] = (float)(i + d);
        int rc = vdb_insert(db, i, vec);
        TEST_ASSERT(rc == 0, "Insertion failed during capacity expansion");
    }
    TEST_ASSERT(vdb_count(db) == VDB_INITIAL_CAPA * 3, "Count mismatch after expansions");
    TEST_ASSERT(vdb_verify_invariants(db), "Invariants broken after expansions");

    // Duplicate ID insertion test
    {
        float vec[VDB_DIMENSION] = {1.0f};