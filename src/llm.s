// llm.s — model, transformer forward pass (RMSNorm + RoPE + GQA + SwiGLU),
//          sampling and interactive serial console.  All in EL1, no OS.
.include "defs.inc"

.equ MAX_LINE, 240

.text
.global kmain

// ---------------------------------------------------------------------------
// alloc(x0 = bytes) -> x0 (64-byte aligned; from the heap after .bss)
// ---------------------------------------------------------------------------
alloc:
    ldr     x1, =g_heap
    ldr     x2, [x1]
    add     x3, x2, x0
    add     x3, x3, #63
    and     x3, x3, #~63
    str     x3, [x1]
    mov     x0, x2
    ret

// ---------------------------------------------------------------------------
// model_init -> w0 = 0 ok, 1 error.  Reads the PIML header, sets up the weight
// pointers (zero-copy, straight from the image) and allocates buffers + KV cache.
// ---------------------------------------------------------------------------
model_init:
    PROLOGUE
    ldr     x19, =model_blob
    ldr     w0, [x19]
    ldr     w1, =0x4C4D4950             // "PIML"
    cmp     w0, w1
    b.ne    9f
    ldr     w21, [x19, #8]              // dim
    ldr     w23, [x19, #12]             // hidden
    ldr     w24, [x19, #16]             // layers
    ldr     w25, [x19, #20]             // heads
    ldr     w26, [x19, #24]             // kv_heads
    ldr     w27, [x19, #28]             // vocab
    ldr     w28, [x19, #32]             // seq
    STG     x21, g_dim, x0
    STG     x23, g_hidden, x0
    STG     x24, g_layers, x0
    STG     x25, g_heads, x0
    STG     x26, g_kvheads, x0
    STG     x27, g_vocab, x0
    STG     x28, g_seq, x0
    udiv    x1, x21, x25                // hd
    STG     x1, g_hd, x0
    mul     x22, x26, x1                // kvdim
    STG     x22, g_kvdim, x0
    udiv    x2, x25, x26
    STG     x2, g_kvmul, x0
    // NEON kernel requirements: multiples of 16
    orr     x3, x21, x23
    orr     x3, x3, x1
    orr     x3, x3, x22
    tst     x3, #15
    b.ne    9f

    // vocabulary: itos = blob+64 ; ctoi[256] = 0xFF except vocab chars
    add     x0, x19, #64
    STG     x0, g_itos, x1
    ldr     x1, =g_ctoi
    mov     x2, #0
    mov     w3, #0xFF
1:  strb    w3, [x1, x2]
    add     x2, x2, #1
    cmp     x2, #256
    b.lo    1b
    mov     x2, #0
2:  ldrb    w3, [x0, x2]
    strb    w2, [x1, x3]
    add     x2, x2, #1
    cmp     x2, x27
    b.lo    2b

    // weights (fp32, contiguous)
    add     x20, x19, #320
    STG     x20, g_emb, x0
    mul     x1, x27, x21
    add     x20, x20, x1, lsl #2
    STG     x20, g_rms_att, x0
    mul     x1, x24, x21
    add     x20, x20, x1, lsl #2
    STG     x20, g_wq, x0
    mul     x1, x21, x21
    mul     x1, x1, x24
    add     x20, x20, x1, lsl #2
    STG     x20, g_wk, x0
    mul     x1, x22, x21
    mul     x1, x1, x24
    add     x20, x20, x1, lsl #2
    STG     x20, g_wv, x0
    add     x20, x20, x1, lsl #2
    STG     x20, g_wo, x0
    mul     x1, x21, x21
    mul     x1, x1, x24
    add     x20, x20, x1, lsl #2
    STG     x20, g_rms_ffn, x0
    mul     x1, x24, x21
    add     x20, x20, x1, lsl #2
    STG     x20, g_w1, x0
    mul     x1, x23, x21
    mul     x1, x1, x24
    add     x20, x20, x1, lsl #2
    STG     x20, g_w2, x0
    add     x20, x20, x1, lsl #2
    STG     x20, g_w3, x0
    add     x20, x20, x1, lsl #2
    STG     x20, g_rms_final, x0
    add     x20, x20, x21, lsl #2
    ldr     x1, =model_blob_end
    cmp     x20, x1                     // size must match exactly
    b.ne    9f

    // activation buffers
    lsl     x0, x21, #2
    bl      alloc
    STG     x0, g_x, x1
    lsl     x0, x21, #2
    bl      alloc
    STG     x0, g_xb, x1
    lsl     x0, x21, #2
    bl      alloc
    STG     x0, g_xb2, x1
    lsl     x0, x21, #2
    bl      alloc
    STG     x0, g_q, x1
    lsl     x0, x23, #2
    bl      alloc
    STG     x0, g_hb, x1
    lsl     x0, x23, #2
    bl      alloc
    STG     x0, g_hb2, x1
    mul     x0, x25, x28
    lsl     x0, x0, #2
    bl      alloc
    STG     x0, g_att, x1
    lsl     x0, x27, #2
    add     x0, x0, #64
    bl      alloc
    STG     x0, g_logits, x1
    // KV cache: layers * seq * kvdim floats (each)
    mul     x0, x24, x28
    mul     x0, x0, x22
    lsl     x0, x0, #2
    bl      alloc
    STG     x0, g_kcache, x1
    mul     x0, x24, x28
    mul     x0, x0, x22
    lsl     x0, x0, #2
    bl      alloc
    STG     x0, g_vcache, x1
    // RoPE tables: seq * hd/2 (cos and sin)
    LDG     x1, g_hd
    lsr     x1, x1, #1
    mul     x0, x28, x1
    lsl     x0, x0, #2
    bl      alloc
    STG     x0, g_rope_cos, x1
    LDG     x1, g_hd
    lsr     x1, x1, #1
    mul     x0, x28, x1
    lsl     x0, x0, #2
    bl      alloc
    STG     x0, g_rope_sin, x1
    bl      rope_init
    mov     w0, #0
    EPILOGUE
9:  mov     w0, #1
    EPILOGUE

// rope_init: cos/sin[p][j] of p * theta^(-2j/hd), theta = 10000
rope_init:
    PROLOGUE
    LDG     x19, g_seq
    LDG     x20, g_hd
    lsr     x21, x20, #1                // half
    LDG     x22, g_rope_cos
    LDG     x23, g_rope_sin
    ucvtf   s9, x20                     // hd (float)
    FCONST  s10, w9, F_LN_THETA
    mov     x24, #0                     // p
1:  mov     x25, #0                     // j
    ucvtf   s11, x24
2:  lsl     x0, x25, #1
    ucvtf   s0, x0
    fdiv    s0, s0, s9                  // 2j/hd
    fmul    s0, s0, s10
    fneg    s0, s0
    bl      expf                        // freq = exp(-(2j/hd) ln theta)
    fmul    s0, s0, s11                 // angle = p * freq
    bl      sincosf
    str     s1, [x22], #4
    str     s0, [x23], #4
    add     x25, x25, #1
    cmp     x25, x21
    b.lo    2b
    add     x24, x24, #1
    cmp     x24, x19
    b.lo    1b
    EPILOGUE

// rope(x0 = vec, x1 = n floats, x2 = pos) — interleaved pairs per head
rope:
    LDG     x9, g_hd
    lsr     x10, x9, #1
    mul     x11, x2, x10
    LDG     x12, g_rope_cos
    add     x12, x12, x11, lsl #2
    LDG     x13, g_rope_sin
    add     x13, x13, x11, lsl #2
    udiv    x14, x1, x9                 // number of heads
1:  mov     x15, #0
2:  ldr     s0, [x12, x15, lsl #2]      // cos
    ldr     s1, [x13, x15, lsl #2]      // sin
    ldp     s2, s3, [x0]
    fmul    s4, s2, s0
    fmsub   s4, s3, s1, s4              // v0*c - v1*s
    fmul    s5, s2, s1
    fmadd   s5, s3, s0, s5              // v0*s + v1*c
    stp     s4, s5, [x0], #8
    add     x15, x15, #1
    cmp     x15, x10
    b.lo    2b
    subs    x14, x14, #1
    b.ne    1b
    ret

// ---------------------------------------------------------------------------
// attn_head(x0 = q_h, x1 = att_h, x2 = K base, x3 = V base, x4 = out_h,
//           x5 = npos, x6 = stride bytes (kvdim*4), x7 = hd, s0 = 1/sqrt(hd))
// ---------------------------------------------------------------------------
attn_head:
    PROLOGUE
    mov     x19, x0
    mov     x20, x1
    mov     x21, x2
    mov     x22, x3
    mov     x23, x4
    mov     x24, x5
    mov     x25, x6
    mov     x26, x7
    fmov    s8, s0
    mov     x27, #0
    mov     x28, x21
1:  mov     x0, x19
    mov     x1, x28
    mov     w2, w26
    bl      dot
    fmul    s0, s0, s8
    str     s0, [x20, x27, lsl #2]
    add     x28, x28, x25
    add     x27, x27, #1
    cmp     x27, x24
    b.lo    1b
    mov     x0, x20
    mov     w1, w24
    bl      softmax
    mov     x0, x23
    mov     w1, w26
    bl      vzero
    mov     x27, #0
    mov     x28, x22
2:  mov     x0, x23
    mov     x1, x28
    ldr     s0, [x20, x27, lsl #2]
    mov     w2, w26
    bl      axpy
    add     x28, x28, x25
    add     x27, x27, #1
    cmp     x27, x24
    b.lo    2b
    EPILOGUE

// ---------------------------------------------------------------------------
// forward(x0 = token, x1 = pos) -> g_logits
// ---------------------------------------------------------------------------
forward:
    PROLOGUE
    mov     x19, x0
    mov     x20, x1
    LDG     x21, g_dim
    LDG     x22, g_kvdim
    LDG     x23, g_hidden
    LDG     x24, g_seq
    LDG     x25, g_hd
    // x = emb[token]
    LDG     x0, g_x
    LDG     x1, g_emb
    mul     x2, x19, x21
    add     x1, x1, x2, lsl #2
    mov     w2, w21
    bl      vcopy
    ucvtf   s0, x25
    fsqrt   s0, s0
    fmov    s1, #1.0
    fdiv    s8, s1, s0                  // 1/sqrt(hd)
    mov     x19, #0                     // layer l
.Llayer:
    // xb = rmsnorm(x, rms_att[l])
    LDG     x0, g_xb
    LDG     x1, g_x
    LDG     x2, g_rms_att
    mul     x3, x19, x21
    add     x2, x2, x3, lsl #2
    mov     w3, w21
    bl      rmsnorm
    // q = Wq[l] xb
    LDG     x0, g_q
    LDG     x1, g_xb
    LDG     x2, g_wq
    mul     x3, x21, x21
    mul     x3, x3, x19
    add     x2, x2, x3, lsl #2
    mov     w3, w21
    mov     w4, w21
    bl      matvec
    // pointers to this position's K/V in the cache
    mul     x3, x19, x24
    add     x3, x3, x20
    mul     x3, x3, x22
    LDG     x26, g_kcache
    add     x26, x26, x3, lsl #2
    LDG     x27, g_vcache
    add     x27, x27, x3, lsl #2
    // k = Wk[l] xb ; v = Wv[l] xb  (written directly into the KV cache)
    mov     x0, x26
    LDG     x1, g_xb
    LDG     x2, g_wk
    mul     x3, x22, x21
    mul     x3, x3, x19
    add     x2, x2, x3, lsl #2
    mov     w3, w21
    mov     w4, w22
    bl      matvec
    mov     x0, x27
    LDG     x1, g_xb
    LDG     x2, g_wv
    mul     x3, x22, x21
    mul     x3, x3, x19
    add     x2, x2, x3, lsl #2
    mov     w3, w21
    mov     w4, w22
    bl      matvec
    // RoPE on q and k
    LDG     x0, g_q
    mov     x1, x21
    mov     x2, x20
    bl      rope
    mov     x0, x26
    mov     x1, x22
    mov     x2, x20
    bl      rope
    // multi-head attention with GQA
    mov     x26, #0                     // head h
.Lhead:
    LDG     x0, g_q
    mul     x1, x26, x25
    add     x0, x0, x1, lsl #2          // q_h
    LDG     x1, g_att
    mul     x2, x26, x24
    add     x1, x1, x2, lsl #2          // att_h
    LDG     x9, g_kvmul
    udiv    x9, x26, x9
    mul     x9, x9, x25                 // kv head offset (floats)
    mul     x10, x19, x24
    mul     x10, x10, x22
    add     x10, x10, x9
    LDG     x2, g_kcache
    add     x2, x2, x10, lsl #2
    LDG     x3, g_vcache
    add     x3, x3, x10, lsl #2
    LDG     x4, g_xb
    mul     x11, x26, x25
    add     x4, x4, x11, lsl #2
    add     x5, x20, #1
    lsl     x6, x22, #2
    mov     x7, x25
    fmov    s0, s8
    bl      attn_head
    add     x26, x26, #1
    LDG     x9, g_heads
    cmp     x26, x9
    b.lo    .Lhead
    // x += Wo[l] xb
    LDG     x0, g_xb2
    LDG     x1, g_xb
    LDG     x2, g_wo
    mul     x3, x21, x21
    mul     x3, x3, x19
    add     x2, x2, x3, lsl #2
    mov     w3, w21
    mov     w4, w21
    bl      matvec
    LDG     x0, g_x
    LDG     x1, g_xb2
    mov     w2, w21
    bl      vadd
    // FFN SwiGLU: x += W2 (silu(W1 xb) * W3 xb)
    LDG     x0, g_xb
    LDG     x1, g_x
    LDG     x2, g_rms_ffn
    mul     x3, x19, x21
    add     x2, x2, x3, lsl #2
    mov     w3, w21
    bl      rmsnorm
    LDG     x0, g_hb
    LDG     x1, g_xb
    LDG     x2, g_w1
    mul     x3, x23, x21
    mul     x3, x3, x19
    add     x2, x2, x3, lsl #2
    mov     w3, w21
    mov     w4, w23
    bl      matvec
    LDG     x0, g_hb2
    LDG     x1, g_xb
    LDG     x2, g_w3
    mul     x3, x23, x21
    mul     x3, x3, x19
    add     x2, x2, x3, lsl #2
    mov     w3, w21
    mov     w4, w23
    bl      matvec
    LDG     x0, g_hb
    LDG     x1, g_hb2
    mov     w2, w23
    bl      silu_mul
    LDG     x0, g_xb
    LDG     x1, g_hb
    LDG     x2, g_w2
    mul     x3, x21, x23
    mul     x3, x3, x19
    add     x2, x2, x3, lsl #2
    mov     w3, w23
    mov     w4, w21
    bl      matvec
    LDG     x0, g_x
    LDG     x1, g_xb
    mov     w2, w21
    bl      vadd
    add     x19, x19, #1
    LDG     x9, g_layers
    cmp     x19, x9
    b.lo    .Llayer
    // final norm + logits (tied embeddings)
    LDG     x0, g_x
    mov     x1, x0
    LDG     x2, g_rms_final
    mov     w3, w21
    bl      rmsnorm
    LDG     x0, g_logits
    LDG     x1, g_x
    LDG     x2, g_emb
    mov     w3, w21
    LDG     x4, g_vocab
    bl      matvec
    EPILOGUE
.ltorg

// ---------------------------------------------------------------------------
// sample -> x0 = token.  temp == 0: argmax; otherwise softmax(logits/temp) + PRNG
// ---------------------------------------------------------------------------
sample:
    PROLOGUE
    LDG     x19, g_logits
    LDG     x20, g_vocab
    ldr     x0, =g_temp
    ldr     s8, [x0]
    fcmp    s8, #0.0
    b.ne    3f
    // argmax
    ldr     s0, [x19]
    mov     x0, #0
    mov     x1, #1
1:  cmp     x1, x20
    b.hs    2f
    ldr     s1, [x19, x1, lsl #2]
    fcmp    s1, s0
    b.le    11f
    fmov    s0, s1
    mov     x0, x1
11: add     x1, x1, #1
    b       1b
2:  EPILOGUE
3:  // scale by 1/temp
    mov     x1, #0
4:  ldr     s0, [x19, x1, lsl #2]
    fdiv    s0, s0, s8
    str     s0, [x19, x1, lsl #2]
    add     x1, x1, #1
    cmp     x1, x20
    b.lo    4b
    mov     x0, x19
    mov     w1, w20
    bl      softmax
    // xorshift64 -> r in [0,1)
    ldr     x2, =g_rng
    ldr     x0, [x2]
    eor     x0, x0, x0, lsl #13
    eor     x0, x0, x0, lsr #7
    eor     x0, x0, x0, lsl #17
    str     x0, [x2]
    lsr     x0, x0, #40
    ucvtf   s1, x0
    FCONST  s2, w9, F_INV24
    fmul    s1, s1, s2
    movi    v2.16b, #0                  // cdf
    mov     x0, #0
5:  ldr     s3, [x19, x0, lsl #2]
    fadd    s2, s2, s3
    fcmp    s1, s2
    b.lt    6f
    add     x0, x0, #1
    cmp     x0, x20
    b.lo    5b
    sub     x0, x20, #1
6:  EPILOGUE

// now -> x0 = CNTPCT_EL0
now:
    isb
    mrs     x0, cntpct_el0
    ret

// rate(x0 = n, x1 = ticks) -> x0 = n/s * 100
rate:
    LDG     x2, g_freq
    mul     x0, x0, x2
    mov     x2, #100
    mul     x0, x0, x2
    cbz     x1, 1f
    udiv    x0, x0, x1
    ret
1:  mov     x0, #0
    ret

// ---------------------------------------------------------------------------
// generate(x0 = prompt text, NUL-terminated)
// ---------------------------------------------------------------------------
generate:
    PROLOGUE
    mov     x19, x0
    // encode chars -> tokens (skip chars not in the vocabulary)
    ldr     x20, =g_tokens
    ldr     x1, =g_ctoi
    LDG     x2, g_seq
    sub     x2, x2, #1
    mov     x21, #0                     // n_prompt
1:  ldrb    w3, [x19], #1
    cbz     w3, 2f
    ldrb    w4, [x1, x3]
    cmp     w4, #0xFF
    b.eq    1b
    strb    w4, [x20, x21]
    add     x21, x21, #1
    cmp     x21, x2
    b.lo    1b
2:  cbnz    x21, 3f
    ldr     x1, =g_ctoi                 // empty prompt -> '\n'
    ldrb    w4, [x1, #10]
    strb    w4, [x20]
    mov     x21, #1
3:
    // echo the (normalized) prompt and start the continuation on the same line
    mov     w0, #10
    bl      uart_putc
    LDG     x22, g_itos
    mov     x23, #0
4:  ldrb    w1, [x20, x23]
    ldrb    w0, [x22, x1]
    bl      uart_putc
    add     x23, x23, #1
    cmp     x23, x21
    b.lo    4b

    // prefill: one forward pass per prompt token
    bl      now
    mov     x24, x0
    mov     x23, #0
5:  ldrb    w0, [x20, x23]
    mov     x1, x23
    bl      forward
    add     x23, x23, #1
    cmp     x23, x21
    b.lo    5b
    bl      now
    mov     x25, x0                     // t end of prefill

    // decode
    mov     x26, #0                     // generated
    mov     x27, x21                    // pos
    LDG     x28, g_max_new
6:  bl      sample
    mov     x23, x0
    ldrb    w0, [x22, x23]
    bl      uart_putc
    add     x26, x26, #1
    cmp     x26, x28
    b.hs    7f
    LDG     x1, g_seq
    cmp     x27, x1
    b.hs    7f
    mov     x0, x23
    mov     x1, x27
    bl      forward
    add     x27, x27, #1
    b       6b
7:  bl      now
    sub     x23, x0, x25                // ticks decode
    sub     x24, x25, x24               // ticks prefill

    PUTS    s_stat1
    mov     x0, x21
    bl      print_u64
    PUTS    s_stat2
    mov     x0, x21
    mov     x1, x24
    bl      rate
    bl      print_2dp
    PUTS    s_stat3
    mov     x0, x26
    bl      print_u64
    PUTS    s_stat4
    mov     x0, x26
    mov     x1, x23
    bl      rate
    bl      print_2dp
    PUTS    s_stat5
    EPILOGUE

// readline -> x0 = g_line (NUL-terminated), with echo and backspace
readline:
    PROLOGUE
    ldr     x19, =g_line
    mov     x20, #0
1:  bl      uart_getc
    cmp     w0, #10
    b.ne    2f
    LDG     x1, g_last_cr               // ignore '\n' after '\r'
    STG     xzr, g_last_cr, x2
    cbnz    x1, 1b
    b       9f
2:  STG     xzr, g_last_cr, x2
    cmp     w0, #13
    b.ne    3f
    mov     x1, #1
    STG     x1, g_last_cr, x2
    b       9f
3:  cmp     w0, #8
    b.eq    4f
    cmp     w0, #127
    b.eq    4f
    cmp     w0, #32
    b.lo    1b
    cmp     w0, #126
    b.hi    1b
    cmp     x20, #MAX_LINE
    b.hs    1b
    strb    w0, [x19, x20]
    add     x20, x20, #1
    bl      uart_putc
    b       1b
4:  cbz     x20, 1b
    sub     x20, x20, #1
    mov     w0, #8
    bl      uart_putc
    mov     w0, #' '
    bl      uart_putc
    mov     w0, #8
    bl      uart_putc
    b       1b
9:  strb    wzr, [x19, x20]
    mov     x0, x19
    EPILOGUE

// starts_with(x0 = s, x1 = prefix) -> x0 = 1/0
starts_with:
1:  ldrb    w3, [x1], #1
    cbz     w3, 2f
    ldrb    w2, [x0], #1
    cmp     w2, w3
    b.eq    1b
    mov     x0, #0
    ret
2:  mov     x0, #1
    ret

// parse_uint(x0 = s) -> x0
parse_uint:
    mov     x1, #0
    mov     x3, #10
1:  ldrb    w2, [x0], #1
    sub     w2, w2, #'0'
    cmp     w2, #9
    b.hi    2f
    madd    x1, x1, x3, x2
    b       1b
2:  mov     x0, x1
    ret

// ---------------------------------------------------------------------------
// kmain
// ---------------------------------------------------------------------------
kmain:
    PROLOGUE
    ldr     x0, =__heap_start
    STG     x0, g_heap, x1
    bl      uart_init
    mrs     x0, cntfrq_el0
    cbnz    x0, 1f
    ldr     x0, =19200000
1:  STG     x0, g_freq, x1
    bl      now
    orr     x0, x0, #1
    STG     x0, g_rng, x1
    // defaults: temp 0.8, 200 tokens
    ldr     w0, =0x3F4CCCCD
    ldr     x1, =g_temp
    str     w0, [x1]
    mov     x0, #200
    STG     x0, g_max_new, x1

    PUTS    s_banner
    PUTS    s_cpu
    bl      cpu_part
    cmp     x0, #0xD08
    b.eq    2f
    PUTS    s_a53
    b       3f
2:  PUTS    s_a72
3:  PUTS    s_el
    LDG     x0, g_boot_el
    bl      print_u64
    PUTS    s_freq
    LDG     x0, g_freq
    bl      print_u64
    PUTS    s_nl

    bl      now
    mov     x19, x0
    bl      model_init
    cbz     w0, 4f
    PUTS    s_badmodel
5:  wfe
    b       5b
4:  bl      now
    sub     x19, x0, x19
    PUTS    s_model
    LDG     x0, g_dim
    bl      print_u64
    PUTS    s_m_l
    LDG     x0, g_layers
    bl      print_u64
    PUTS    s_m_h
    LDG     x0, g_heads
    bl      print_u64
    PUTS    s_m_kv
    LDG     x0, g_kvheads
    bl      print_u64
    PUTS    s_m_ff
    LDG     x0, g_hidden
    bl      print_u64
    PUTS    s_m_v
    LDG     x0, g_vocab
    bl      print_u64
    PUTS    s_m_ctx
    LDG     x0, g_seq
    bl      print_u64
    PUTS    s_m_bytes
    ldr     x0, =model_blob_end
    ldr     x1, =model_blob
    sub     x0, x0, x1
    bl      print_u64
    PUTS    s_m_mem
    LDG     x0, g_heap
    ldr     x1, =_start
    sub     x0, x0, x1
    lsr     x0, x0, #10
    bl      print_u64
    PUTS    s_m_init
    mov     x0, #1000
    mul     x0, x19, x0
    LDG     x1, g_freq
    udiv    x0, x0, x1
    bl      print_u64
    PUTS    s_m_end
    PUTS    s_help

.Lrepl:
    PUTS    s_prompt
    bl      readline
    mov     x19, x0
    ldrb    w1, [x19]
    cmp     w1, #'/'
    b.ne    .Lgen
    mov     x0, x19
    ldr     x1, =c_help
    bl      starts_with
    cbz     x0, 1f
    PUTS    s_help
    b       .Lrepl
1:  mov     x0, x19
    ldr     x1, =c_greedy
    bl      starts_with
    cbz     x0, 2f
    ldr     x1, =g_temp
    str     wzr, [x1]
    PUTS    s_ok
    b       .Lrepl
2:  mov     x0, x19
    ldr     x1, =c_temp
    bl      starts_with
    cbz     x0, 3f
    add     x0, x19, #2
    bl      parse_uint
    ucvtf   s0, x0
    FCONST  s1, w9, F_TEN
    fdiv    s0, s0, s1
    ldr     x1, =g_temp
    str     s0, [x1]
    PUTS    s_ok
    b       .Lrepl
3:  mov     x0, x19
    ldr     x1, =c_ntok
    bl      starts_with
    cbz     x0, 4f
    add     x0, x19, #2
    bl      parse_uint
    cmp     x0, #1
    b.lo    .Lrepl
    STG     x0, g_max_new, x1
    PUTS    s_ok
    b       .Lrepl
4:  mov     x0, x19
    ldr     x1, =c_bench
    bl      starts_with
    cbz     x0, 6f
    ldr     x20, =g_temp                // bench: greedy, 200 tokens
    ldr     w21, [x20]
    str     wzr, [x20]
    LDG     x22, g_max_new
    mov     x0, #200
    STG     x0, g_max_new, x1
    ldr     x0, =s_benchprompt
    bl      generate
    str     w21, [x20]
    STG     x22, g_max_new, x1
    b       .Lrepl
6:  PUTS    s_unknown
    b       .Lrepl
.Lgen:
    mov     x0, x19
    bl      generate
    b       .Lrepl
.ltorg

// ---------------------------------------------------------------------------
.section .rodata
s_banner:   .asciz "\n\n==============================================\n pi-llm.asm :: bare-metal AArch64 LLM (no OS)\n RMSNorm + RoPE + GQA + SwiGLU, 100% assembly\n==============================================\n"
s_cpu:      .asciz "cpu: "
s_a53:      .asciz "Cortex-A53 (Pi 3 / Zero 2 W)"
s_a72:      .asciz "Cortex-A72 (Pi 4)"
s_el:       .asciz " | boot EL: "
s_freq:     .asciz " | timer: "
s_nl:       .asciz " Hz\n"
s_badmodel: .asciz "ERROR: invalid PIML model\n"
s_model:    .asciz "model: dim="
s_m_l:      .asciz " layers="
s_m_h:      .asciz " heads="
s_m_kv:     .asciz " kv="
s_m_ff:     .asciz " ffn="
s_m_v:      .asciz " vocab="
s_m_ctx:    .asciz " ctx="
s_m_bytes:  .asciz "\nweights: "
s_m_mem:    .asciz " bytes | total memory: "
s_m_init:   .asciz " KiB | init: "
s_m_end:    .asciz " ms\n"
s_help:     .asciz "\nType the start of a text (e.g. ROMEO:) and press Enter.\n  /tN  temperature N/10 (e.g. /t8)   /greedy  deterministic\n  /nN  tokens to generate (e.g. /n300)   /bench   measure speed\n"
s_prompt:   .asciz "\n\npi-llm> "
s_ok:       .asciz "  ok"
s_unknown:  .asciz "  unknown command (/help)"
s_stat1:    .asciz "\n\n[prompt "
s_stat2:    .asciz " tok @ "
s_stat3:    .asciz " tok/s | generated "
s_stat4:    .asciz " tok @ "
s_stat5:    .asciz " tok/s]"
s_benchprompt: .asciz "ROMEO:"
c_help:     .asciz "/help"
c_greedy:   .asciz "/greedy"
c_temp:     .asciz "/t"
c_ntok:     .asciz "/n"
c_bench:    .asciz "/bench"

// ---------------------------------------------------------------------------
.bss
.balign 8
g_heap:     .skip 8
g_freq:     .skip 8
g_rng:      .skip 8
g_temp:     .skip 8
g_max_new:  .skip 8
g_last_cr:  .skip 8
g_dim:      .skip 8
g_hidden:   .skip 8
g_layers:   .skip 8
g_heads:    .skip 8
g_kvheads:  .skip 8
g_vocab:    .skip 8
g_seq:      .skip 8
g_hd:       .skip 8
g_kvdim:    .skip 8
g_kvmul:    .skip 8
g_itos:     .skip 8
g_emb:      .skip 8
g_rms_att:  .skip 8
g_wq:       .skip 8
g_wk:       .skip 8
g_wv:       .skip 8
g_wo:       .skip 8
g_rms_ffn:  .skip 8
g_w1:       .skip 8
g_w2:       .skip 8
g_w3:       .skip 8
g_rms_final:.skip 8
g_x:        .skip 8
g_xb:       .skip 8
g_xb2:      .skip 8
g_q:        .skip 8
g_hb:       .skip 8
g_hb2:      .skip 8
g_att:      .skip 8
g_logits:   .skip 8
g_kcache:   .skip 8
g_vcache:   .skip 8
g_rope_cos: .skip 8
g_rope_sin: .skip 8
g_ctoi:     .skip 256
g_tokens:   .skip 4096
g_line:     .skip 256
