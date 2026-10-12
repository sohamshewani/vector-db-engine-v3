#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

#define VECTOR_DIM 128
#define NUM_ITERATIONS 100000
#define WARMUP_ITERATIONS 10000

typedef struct {
    float data[VECTOR_DIM];
} Vector;

typedef struct {
    uint64_t *latencies;
    size_t count;
    double ops_sec;
} BenchmarkResult;

static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int compare_uint64(const void *a, const void *b) {
    uint64_t fa = *(const uint64_t *)a;
    uint64_t fb = *(const uint64_t *)b;
    return (fa > fb) - (fa < fb);
}

static float mock_vector_distance(const Vector *a, const Vector *b) {
    float sum = 0.0f;
    for (int i = 0; i < VECTOR_DIM; i++) {
        float diff = a->data[i] - b->data[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

void *benchmark_worker(void *arg) {
    BenchmarkResult *res = (BenchmarkResult *)arg;
    res->latencies = malloc(NUM_ITERATIONS * sizeof(uint64_t));
    if (!res->latencies) {
        perror("malloc failed");
        pthread_exit(NULL);
    }

    Vector v1, v2;
    for (int i = 0; i < VECTOR_DIM; i++) {
        v1.data[i] = (float)i * 0.1f;
        v2.data[i] = (float)i * 0.11f;
    }

    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        volatile float d = mock_vector_distance(&v1, &v2);
        (void)d;
    }

    uint64_t start_total = get_time_ns();

    for (size_t i = 0; i < NUM_ITERATIONS; i++) {
        uint64_t start_op = get_time_ns();
        
        volatile float d = mock_vector_distance(&v1, &v2);
        (void)d;

        uint64_t end_op = get_time_ns();
        res->latencies[i] = end_op - start_op;
    }

    uint64_t end_total = get_time_ns();
    uint64_t total_duration_ns = end_total - start_total;

    res->count = NUM_ITERATIONS;
    res->ops_sec = ((double)NUM_ITERATIONS / (double)total_duration_ns) * 1e9;

    return NULL;
}

int main(void) {
    BenchmarkResult result;
    
    pthread_t thread;
    if (pthread_create(&thread, NULL, benchmark_worker, &result) != 0) {
        perror("Failed to create thread");
        return 1;
    }

    pthread_join(thread, NULL);

    qsort(result.latencies, result.count, sizeof(uint64_t), compare_uint64);

    uint64_t p50 = result.latencies[(size_t)(result.count * 0.50)];
    uint64_t p99 = result.latencies[(size_t)(result.count * 0.99)];
    uint64_t p999 = result.latencies[(size_t)(result.count * 0.999)];

    printf("\n");
    printf("+--------------------------+----------------------+\n");
    printf("| Metric                   | Value                |\n");
    printf("+--------------------------+----------------------+\n");
    printf("| Benchmark Engine         | vector-db-engine     |\n");
    printf("| Vector Dimension         | %-20d |\n", VECTOR_DIM);
    printf("| Total Iterations         | %-20d |\n", NUM_ITERATIONS);
    printf("| Throughput (ops/sec)     | %-20.2f |\n", result.ops_sec);
    printf("| Latency P50              | %-17llu ns |\n", (unsigned long long)p50);
    printf("| Latency P99              | %-17llu ns |\n", (unsigned long long)p99);
    printf("| Latency P99.9            | %-17llu ns |\n", (unsigned long long)p999);
    printf("+--------------------------+----------------------+\n");
    printf("\n");

    free(result.latencies);
    return 0;
}