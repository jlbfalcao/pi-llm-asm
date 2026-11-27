// math.s — float32 math kernels with NEON (AArch64 ASIMD)
// Convention: leaf functions; expf/sincosf only clobber v0-v7 and x9-x11.
.include "defs.inc"

.text
.global expf, sincosf, dot, matvec, axpy, vadd, vcopy, vzero, rmsnorm, softmax, silu_mul

// ---------------------------------------------------------------------------
// expf(s0) -> s0     (Cephes: ln2 range reduction + degree-6 polynomial, ~1 ulp)
// ---------------------------------------------------------------------------
expf:
    FCONST  s1, w9, F_XMAX
    fmin    s0, s0, s1
    FCONST  s1, w9, F_XMIN
    fmax    s0, s0, s1
    FCONST  s1, w9, F_LOG2E
    fmul    s2, s0, s1
    frintn  s2, s2                      // n = round(x*log2e)
    FCONST  s1, w9, F_C1
    fmsub   s0, s2, s1, s0              // r = x - n*C1
    FCONST  s1, w9, F_C2
    fmsub   s0, s2, s1, s0              //   - n*C2
    FCONST  s3, w9, F_P0
    FCONST  s1, w9, F_P1
    fmadd   s3, s3, s0, s1
    FCONST  s1, w9, F_P2
    fmadd   s3, s3, s0, s1
    FCONST  s1, w9, F_P3
    fmadd   s3, s3, s0, s1
    FCONST  s1, w9, F_P4
    fmadd   s3, s3, s0, s1
    FCONST  s1, w9, F_P5
    fmadd   s3, s3, s0, s1
    fmul    s4, s0, s0                  // r^2
    fmul    s3, s3, s4
    fadd    s3, s3, s0
    fmov    s1, #1.0
    fadd    s3, s3, s1                  // p = 1 + r + r^2*P(r)
    fcvtzs  w10, s2
    add     w10, w10, #127
    lsl     w10, w10, #23
    fmov    s1, w10                     // 2^n
    fmul    s0, s3, s1
    ret

// ---------------------------------------------------------------------------
// sincosf(s0 = a) -> s0 = sin(a), s1 = cos(a)
// Cody-Waite reduction by pi/2 (3 parts) + polynomials on [-pi/4, pi/4]
// ---------------------------------------------------------------------------
sincosf:
    FCONST  s1, w9, F_TWO_OVER_PI
    fmul    s2, s0, s1
    frintn  s2, s2                      // k
    FCONST  s1, w9, F_DP1
    fmsub   s0, s2, s1, s0
    FCONST  s1, w9, F_DP2
    fmsub   s0, s2, s1, s0
    FCONST  s1, w9, F_DP3
    fmsub   s0, s2, s1, s0              // r
    fcvtzs  w10, s2
    and     w10, w10, #3                // quadrant
    fmul    s3, s0, s0                  // z = r^2
    // sin(r) = r + r*z*(S1 + z*(S2 + z*S3))
    FCONST  s4, w9, F_S3
    FCONST  s1, w9, F_S2
    fmadd   s4, s4, s3, s1
    FCONST  s1, w9, F_S1
    fmadd   s4, s4, s3, s1
    fmul    s4, s4, s3
    fmadd   s4, s4, s0, s0              // s4 = sin(r)
    // cos(r) = 1 - z/2 + z^2*(K1 + z*(K2 + z*K3))
    FCONST  s5, w9, F_K3
    FCONST  s1, w9, F_K2
    fmadd   s5, s5, s3, s1
    FCONST  s1, w9, F_K1
    fmadd   s5, s5, s3, s1
    fmul    s6, s3, s3
    fmul    s5, s5, s6
    fmov    s1, #0.5
    fmsub   s5, s3, s1, s5              // z^2*K - z/2
    fmov    s1, #1.0
    fadd    s5, s5, s1                  // s5 = cos(r)
    // quadrant: 0:( s, c) 1:( c,-s) 2:(-s,-c) 3:(-c, s)
    fneg    s6, s4
    fneg    s7, s5
    cmp     w10, #1
    b.eq    1f
    cmp     w10, #2
    b.eq    2f
    cmp     w10, #3
    b.eq    3f
    fmov    s0, s4
    fmov    s1, s5
    ret
1:  fmov    s0, s5
    fmov    s1, s6
    ret
2:  fmov    s0, s6
    fmov    s1, s7
    ret
3:  fmov    s0, s7
    fmov    s1, s4
    ret

// ---------------------------------------------------------------------------
// dot(x0 = a, x1 = b, w2 = n [multiple of 16]) -> s0
// ---------------------------------------------------------------------------
dot:
    movi    v0.16b, #0
    movi    v1.16b, #0
    movi    v2.16b, #0
    movi    v3.16b, #0
1:  ld1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64
    ld1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x1], #64
    fmla    v0.4s, v4.4s, v16.4s
    fmla    v1.4s, v5.4s, v17.4s
    fmla    v2.4s, v6.4s, v18.4s
    fmla    v3.4s, v7.4s, v19.4s
    subs    w2, w2, #16
    b.ne    1b
    fadd    v0.4s, v0.4s, v1.4s
    fadd    v2.4s, v2.4s, v3.4s
    fadd    v0.4s, v0.4s, v2.4s
    faddp   v0.4s, v0.4s, v0.4s
    faddp   s0, v0.2s
    ret

// ---------------------------------------------------------------------------
// matvec(x0 = out[d], x1 = x[n], x2 = W[d][n], w3 = n, w4 = d)
// out[i] = sum_j W[i][j] * x[j]     (n multiple of 16)
// 2 rows per iteration (8 accumulators) + 1-row tail
// ---------------------------------------------------------------------------
matvec:
    uxtw    x3, w3
    lsl     x8, x3, #2                  // bytes per row
    cmp     w4, #2
    b.lo    3f
1:  // ---- row pair ----
    mov     x6, x1
    add     x5, x2, x8                  // row i+1
    mov     w7, w3
    movi    v0.16b, #0
    movi    v1.16b, #0
    movi    v2.16b, #0
    movi    v3.16b, #0
    movi    v20.16b, #0
    movi    v21.16b, #0
    movi    v22.16b, #0
    movi    v23.16b, #0
2:  ld1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x6], #64
    ld1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x2], #64
    ld1     {v24.4s, v25.4s, v26.4s, v27.4s}, [x5], #64
    fmla    v0.4s, v4.4s, v16.4s
    fmla    v1.4s, v5.4s, v17.4s
    fmla    v2.4s, v6.4s, v18.4s
    fmla    v3.4s, v7.4s, v19.4s
    fmla    v20.4s, v24.4s, v16.4s
    fmla    v21.4s, v25.4s, v17.4s
    fmla    v22.4s, v26.4s, v18.4s
    fmla    v23.4s, v27.4s, v19.4s
    subs    w7, w7, #16
    b.ne    2b
    fadd    v0.4s, v0.4s, v1.4s
    fadd    v2.4s, v2.4s, v3.4s
    fadd    v0.4s, v0.4s, v2.4s
    fadd    v20.4s, v20.4s, v21.4s
    fadd    v22.4s, v22.4s, v23.4s
    fadd    v20.4s, v20.4s, v22.4s
    faddp   v0.4s, v0.4s, v20.4s        // [a01, a23, b01, b23]
    faddp   v0.4s, v0.4s, v0.4s         // [a, b, a, b]
    str     d0, [x0], #8
    mov     x2, x5                      // skip row i+1, already consumed
    sub     w4, w4, #2
    cmp     w4, #2
    b.hs    1b
3:  cbz     w4, 5f
    // ---- last row (odd d) ----
    mov     x6, x1
    mov     w7, w3
    movi    v0.16b, #0
    movi    v1.16b, #0
    movi    v2.16b, #0
    movi    v3.16b, #0
4:  ld1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x2], #64
    ld1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x6], #64
    fmla    v0.4s, v4.4s, v16.4s
    fmla    v1.4s, v5.4s, v17.4s
    fmla    v2.4s, v6.4s, v18.4s
    fmla    v3.4s, v7.4s, v19.4s
    subs    w7, w7, #16
    b.ne    4b
    fadd    v0.4s, v0.4s, v1.4s
    fadd    v2.4s, v2.4s, v3.4s
    fadd    v0.4s, v0.4s, v2.4s
    faddp   v0.4s, v0.4s, v0.4s
    faddp   s0, v0.2s
    str     s0, [x0], #4
5:  ret

// axpy(x0 = y, x1 = x, s0 = a, w2 = n [mult. of 4]):  y += a*x
axpy:
    dup     v1.4s, v0.s[0]
1:  ld1     {v2.4s}, [x1], #16
    ld1     {v3.4s}, [x0]
    fmla    v3.4s, v2.4s, v1.4s
    st1     {v3.4s}, [x0], #16
    subs    w2, w2, #4
    b.ne    1b
    ret

// vadd(x0 = y, x1 = x, w2 = n): y += x
vadd:
1:  ld1     {v0.4s}, [x0]
    ld1     {v1.4s}, [x1], #16
    fadd    v0.4s, v0.4s, v1.4s
    st1     {v0.4s}, [x0], #16
    subs    w2, w2, #4
    b.ne    1b
    ret

// vcopy(x0 = dst, x1 = src, w2 = n)
vcopy:
1:  ld1     {v0.4s}, [x1], #16
    st1     {v0.4s}, [x0], #16
    subs    w2, w2, #4
    b.ne    1b
    ret

// vzero(x0 = dst, w1 = n)
vzero:
    movi    v0.16b, #0
1:  st1     {v0.4s}, [x0], #16
    subs    w1, w1, #4
    b.ne    1b
    ret

// rmsnorm(x0 = out, x1 = x, x2 = w, w3 = n): out = x / sqrt(mean(x^2)+eps) * w
rmsnorm:
    movi    v0.16b, #0
    mov     x4, x1
    mov     w5, w3
1:  ld1     {v1.4s}, [x4], #16
    fmla    v0.4s, v1.4s, v1.4s
    subs    w5, w5, #4
    b.ne    1b
    faddp   v0.4s, v0.4s, v0.4s
    faddp   s0, v0.2s
    ucvtf   s1, w3
    fdiv    s0, s0, s1
    FCONST  s1, w9, F_EPS
    fadd    s0, s0, s1
    fsqrt   s0, s0
    fmov    s1, #1.0
    fdiv    s0, s1, s0
    dup     v0.4s, v0.s[0]
2:  ld1     {v1.4s}, [x1], #16
    ld1     {v2.4s}, [x2], #16
    fmul    v1.4s, v1.4s, v0.4s
    fmul    v1.4s, v1.4s, v2.4s
    st1     {v1.4s}, [x0], #16
    subs    w3, w3, #4
    b.ne    2b
    ret

// softmax(x0 = x, w1 = n) in-place  (uses x12-x14, s16-s17; calls expf)
softmax:
    str     x30, [sp, #-16]!
    mov     x12, x0
    mov     w13, w1
    ldr     s16, [x12]
    mov     w14, #1
1:  cmp     w14, w13
    b.hs    2f
    ldr     s1, [x12, w14, uxtw #2]
    fmax    s16, s16, s1
    add     w14, w14, #1
    b       1b
2:  movi    v17.16b, #0
    mov     w14, #0
3:  ldr     s0, [x12, w14, uxtw #2]
    fsub    s0, s0, s16
    bl      expf
    str     s0, [x12, w14, uxtw #2]
    fadd    s17, s17, s0
    add     w14, w14, #1
    cmp     w14, w13
    b.lo    3b
    fmov    s1, #1.0
    fdiv    s17, s1, s17
    mov     w14, #0
4:  ldr     s0, [x12, w14, uxtw #2]
    fmul    s0, s0, s17
    str     s0, [x12, w14, uxtw #2]
    add     w14, w14, #1
    cmp     w14, w13
    b.lo    4b
    ldr     x30, [sp], #16
    ret

// silu_mul(x0 = h, x1 = g, w2 = n): h = h*sigmoid(h) * g   (SwiGLU)
silu_mul:
    str     x30, [sp, #-16]!
    mov     x12, x0
    mov     x13, x1
    mov     w14, w2
1:  ldr     s16, [x12]
    fneg    s0, s16
    bl      expf
    fmov    s1, #1.0
    fadd    s0, s0, s1
    fdiv    s0, s16, s0                 // silu(h) = h / (1 + e^-h)
    ldr     s1, [x13], #4
    fmul    s0, s0, s1
    str     s0, [x12], #4
    subs    w14, w14, #1
    b.ne    1b
    ldr     x30, [sp], #16
    ret
.ltorg
