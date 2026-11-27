// boot.s — kernel8.img entry point (loaded at 0x80000 by the Pi firmware)
// Core 0 only; EL3/EL2 -> EL1; NEON enabled; .bss zeroed; MMU + caches enabled.
.include "defs.inc"

.section .text.boot, "ax"
.global _start
_start:
    mrs     x0, mpidr_el1
    and     x0, x0, #0xFF
    cbz     x0, 1f
0:  wfe                             // secondary cores stay parked
    b       0b
1:
    mrs     x20, CurrentEL
    lsr     x20, x20, #2            // x20 = boot EL (saved later)

    cmp     x20, #3
    b.ne    2f
    // ---- EL3 -> EL1 ----
    mrs     x1, s3_1_c15_c2_1       // CPUECTLR_EL1: SMPEN (A53/A72 cache coherency)
    orr     x1, x1, #(1 << 6)
    msr     s3_1_c15_c2_1, x1
    mov     x1, #0x5B1              // NS | RES1 | SMD | HCE | RW(AArch64)
    msr     scr_el3, x1
    mov     x1, #(1 << 31)          // HCR_EL2.RW = EL1 in AArch64
    msr     hcr_el2, x1
    ldr     x1, =0x30D00800
    msr     sctlr_el1, x1
    mov     x1, #0x3C5              // EL1h, DAIF masked
    msr     spsr_el3, x1
    adr     x1, el1_entry
    msr     elr_el3, x1
    eret
2:
    cmp     x20, #2
    b.ne    el1_entry
    // ---- EL2 -> EL1 ----
    mrs     x1, s3_1_c15_c2_1
    orr     x1, x1, #(1 << 6)
    msr     s3_1_c15_c2_1, x1
    mov     x1, #(1 << 31)
    msr     hcr_el2, x1
    mrs     x1, cnthctl_el2         // EL1 may read the physical counter
    orr     x1, x1, #3
    msr     cnthctl_el2, x1
    msr     cntvoff_el2, xzr
    mov     x1, #0x33FF             // no FP/SIMD traps
    msr     cptr_el2, x1
    msr     hstr_el2, xzr
    ldr     x1, =0x30D00800
    msr     sctlr_el1, x1
    mov     x1, #0x3C5
    msr     spsr_el2, x1
    adr     x1, el1_entry
    msr     elr_el2, x1
    eret

el1_entry:
    ldr     x1, =_start             // stack grows down from 0x80000
    mov     sp, x1
    mov     x1, #(3 << 20)          // CPACR_EL1.FPEN = 11 -> NEON/FP without traps
    msr     cpacr_el1, x1
    isb

    ldr     x1, =__bss_start
    ldr     x2, =__bss_end
3:  cmp     x1, x2
    b.hs    4f
    stp     xzr, xzr, [x1], #16
    b       3b
4:
    STG     x20, g_boot_el, x1
    bl      mmu_init
    bl      kmain
5:  wfe
    b       5b

// ---------------------------------------------------------------------------
// mmu_init: 4 GiB identity map, 4 KiB granule
//   L1[0] -> L2 table (512 x 2 MiB): normal WB RAM up to 0x3F000000, device after
//   L1[1], L1[2] -> 1 GiB normal RAM (Pi 4 with 2/4/8 GB)
//   L1[3] -> 1 GiB device (Pi 4 peripherals at 0xFE000000)
// ---------------------------------------------------------------------------
.equ BLK_NORMAL, 0x701                  // AF | SH=inner | AttrIdx=0 | block
.equ BLK_DEVICE, 0x0060000000000405     // UXN | PXN | AF | AttrIdx=1 | block

.text
mmu_init:
    ldr     x0, =pt_l1
    ldr     x1, =pt_l2
    orr     x2, x1, #3
    str     x2, [x0]
    ldr     x3, =BLK_NORMAL
    mov     x4, #0x40000000
    orr     x2, x4, x3
    str     x2, [x0, #8]
    mov     x4, #0x80000000
    orr     x2, x4, x3
    str     x2, [x0, #16]
    ldr     x3, =BLK_DEVICE
    mov     x4, #0xC0000000
    orr     x2, x4, x3
    str     x2, [x0, #24]

    ldr     x7, =BLK_NORMAL
    ldr     x8, =BLK_DEVICE
    mov     x6, #0x3F000000
    mov     x5, #0
1:  lsl     x4, x5, #21
    cmp     x4, x6
    csel    x3, x8, x7, hs
    orr     x2, x4, x3
    str     x2, [x1, x5, lsl #3]
    add     x5, x5, #1
    cmp     x5, #512
    b.lo    1b

    mov     x0, #0xFF                   // MAIR: attr0 = normal WB RW-alloc, attr1 = device nGnRnE
    msr     mair_el1, x0
    ldr     x0, =0x803520               // T0SZ=32 | IRGN0/ORGN0=WBWA | SH0=inner | EPD1
    msr     tcr_el1, x0
    ldr     x0, =pt_l1
    msr     ttbr0_el1, x0
    dsb     ish
    isb
    tlbi    vmalle1
    dsb     ish
    isb
    mrs     x0, sctlr_el1
    orr     x0, x0, #(1 << 0)           // M  (MMU)
    orr     x0, x0, #(1 << 2)           // C  (data cache)
    orr     x0, x0, #(1 << 12)          // I  (instruction cache)
    bic     x0, x0, #(1 << 1)           // A  (no alignment checking)
    msr     sctlr_el1, x0
    isb
    ret
.ltorg

.bss
.balign 4096
pt_l1:  .skip 4096
pt_l2:  .skip 4096
.global g_boot_el
.balign 8
g_boot_el: .skip 8
