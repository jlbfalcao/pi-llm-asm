// defs.inc — shared constants and macros (GNU as, AArch64)

// ---- IEEE-754 float32 constants (bit patterns) ----
.equ F_LOG2E,       0x3FB8AA3B
.equ F_C1,          0x3F318000      // ln2 hi (Cephes)
.equ F_C2,          0xB95E8083      // ln2 lo
.equ F_P0,          0x39506967
.equ F_P1,          0x3AB743CE
.equ F_P2,          0x3C088908
.equ F_P3,          0x3D2AA9C1
.equ F_P4,          0x3E2AAAAA
.equ F_P5,          0x3F000000
.equ F_XMAX,        0x42B0999A      //  88.3
.equ F_XMIN,        0xC2AE999A      // -87.3
.equ F_TWO_OVER_PI, 0x3F22F983
.equ F_DP1,         0x3FC90000      // pi/2 in 3 parts (Cody-Waite)
.equ F_DP2,         0x39FDA000
.equ F_DP3,         0x33A22169
.equ F_S1,          0xBE2AAAA3      // sin polynomial
.equ F_S2,          0x3C08839E
.equ F_S3,          0xB94CA1F9
.equ F_K1,          0x3D2AAAA5      // cos polynomial
.equ F_K2,          0xBAB6061A
.equ F_K3,          0x37CCF5CE
.equ F_EPS,         0x3727C5AC      // 1e-5 (RMSNorm)
.equ F_LN_THETA,    0x41135D8E      // ln(10000) (RoPE)
.equ F_INV24,       0x33800000      // 2^-24
.equ F_TEN,         0x41200000

// ---- peripherals ----
.equ PERIPH_PI3,    0x3F000000
.equ PERIPH_PI4,    0xFE000000
.equ GPIO_OFF,      0x200000
.equ UART0_OFF,     0x201000
.equ UART_DR,       0x00
.equ UART_FR,       0x18
.equ UART_IBRD,     0x24
.equ UART_FBRD,     0x28
.equ UART_LCRH,     0x2C
.equ UART_CR,       0x30
.equ UART_ICR,      0x44

// load a float32 constant (bits) into register sN using wTMP
.macro FCONST sreg, wtmp, bits
    movz \wtmp, #((\bits) & 0xFFFF)
    movk \wtmp, #(((\bits) >> 16) & 0xFFFF), lsl #16
    fmov \sreg, \wtmp
.endm

// load the (64-bit) value of a global variable
.macro LDG reg, sym
    ldr \reg, =\sym
    ldr \reg, [\reg]
.endm

// store reg into global variable sym (uses tmp)
.macro STG reg, sym, tmp
    ldr \tmp, =\sym
    str \reg, [\tmp]
.endm

// full AAPCS64 prologue/epilogue (x19-x30, d8-d15) — 160 bytes
.macro PROLOGUE
    stp x29, x30, [sp, #-160]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    stp d8,  d9,  [sp, #96]
    stp d10, d11, [sp, #112]
    stp d12, d13, [sp, #128]
    stp d14, d15, [sp, #144]
.endm

.macro EPILOGUE
    ldp d14, d15, [sp, #144]
    ldp d12, d13, [sp, #128]
    ldp d10, d11, [sp, #112]
    ldp d8,  d9,  [sp, #96]
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #160
    ret
.endm

// print a string literal
.macro PUTS str
    ldr x0, =\str
    bl uart_puts
.endm
