#ifndef STATOR_C_COMMON_H
#define STATOR_C_COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ── Error codes ─────────────────────────────────────────────────────────── */
#define STATOR_OK         0
#define STATOR_ERR_INVAL  1   /* invalid argument */
#define STATOR_ERR_RANGE  2   /* out of range */
#define STATOR_ERR_LOGIC  3   /* logic error / bad state */
#define STATOR_ERR_IO     4   /* file I/O error */
#define STATOR_ERR_NOMEM  5   /* allocation failed */
#define STATOR_ERR_FORK   6   /* fork() failed */

/* ── Boolean ─────────────────────────────────────────────────────────────── */
#ifndef __cplusplus
#include <stdbool.h>
#endif

/* ── IntPair ─────────────────────────────────────────────────────────────── */
typedef struct {
    int first;
    int second;
} IntPair;

/* ── IntVec — growable int array ─────────────────────────────────────────── */
typedef struct {
    int*   data;
    int    len;
    int    cap;
} IntVec;

static inline void intvec_init(IntVec* v) {
    v->data = NULL; v->len = 0; v->cap = 0;
}

static inline int intvec_push(IntVec* v, int val) {
    if (v->len == v->cap) {
        int new_cap = v->cap ? v->cap * 2 : 8;
        int* tmp = (int*)realloc(v->data, (size_t)new_cap * sizeof(int));
        if (!tmp) return STATOR_ERR_NOMEM;
        v->data = tmp;
        v->cap  = new_cap;
    }
    v->data[v->len++] = val;
    return STATOR_OK;
}

static inline void intvec_free(IntVec* v) {
    free(v->data);
    v->data = NULL; v->len = 0; v->cap = 0;
}

/* ── PairVec — growable IntPair array ────────────────────────────────────── */
typedef struct {
    IntPair* data;
    int      len;
    int      cap;
} PairVec;

static inline void pairvec_init(PairVec* v) {
    v->data = NULL; v->len = 0; v->cap = 0;
}

static inline int pairvec_push(PairVec* v, int first, int second) {
    if (v->len == v->cap) {
        int new_cap = v->cap ? v->cap * 2 : 8;
        IntPair* tmp = (IntPair*)realloc(v->data, (size_t)new_cap * sizeof(IntPair));
        if (!tmp) return STATOR_ERR_NOMEM;
        v->data = tmp;
        v->cap  = new_cap;
    }
    v->data[v->len].first  = first;
    v->data[v->len].second = second;
    v->len++;
    return STATOR_OK;
}

static inline void pairvec_free(PairVec* v) {
    free(v->data);
    v->data = NULL; v->len = 0; v->cap = 0;
}

/* ── Error buffer size ───────────────────────────────────────────────────── */
#define STATOR_ERR_BUF 512

/* ── snprintf-based error fill helper ───────────────────────────────────── */
#define STATOR_SET_ERR(buf, len, ...) \
    do { if ((buf) && (len) > 0) snprintf((buf), (len), __VA_ARGS__); } while(0)

#endif /* STATOR_C_COMMON_H */
