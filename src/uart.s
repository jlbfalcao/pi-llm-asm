// uart.s — PL011 UART0 (GPIO14/15, 115200 8N1) for Pi 3 / Zero 2 W / Pi 4
.include "defs.inc"

.text
.global uart_init, uart_putc, uart_getc, uart_puts, print_u64, print_2dp, cpu_part

// cpu_part -> x0 = MIDR_EL1.PartNum (0xD03 = Cortex-A53, 0xD08 = Cortex-A72)
cpu_part:
    mrs     x0, midr_el1
    ubfx    x0, x0, #4, #12
    ret

uart_init:
    mov     x9, x30
    bl      cpu_part
    mov     x30, x9
    ldr     x1, =PERIPH_PI3
    ldr     x2, =PERIPH_PI4
    cmp     x0, #0xD08
    csel    x1, x2, x1, eq
    STG     x1, g_periph, x3
    add     x2, x1, #UART0_OFF
    STG     x2, g_uart, x3

    str     wzr, [x2, #UART_CR]         // disable UART
    // GPIO14/15 -> ALT0 (TXD0/RXD0): GPFSEL1 bits 12..17 = 100 100
    add     x3, x1, #GPIO_OFF
    ldr     w4, [x3, #4]
    mov     w5, #(0x3F << 12)
    bic     w4, w4, w5
    ldr     w5, =(0x24 << 12)
    orr     w4, w4, w5
    str     w4, [x3, #4]
    // baud: UARTCLK 48 MHz -> 115200 = IBRD 26, FBRD 3
    mov     w4, #0x7FF
    str     w4, [x2, #UART_ICR]
    mov     w4, #26
    str     w4, [x2, #UART_IBRD]
    mov     w4, #3
    str     w4, [x2, #UART_FBRD]
    mov     w4, #0x70                   // 8 bits, FIFO
    str     w4, [x2, #UART_LCRH]
    mov     w4, #0x301                  // UARTEN | TXE | RXE
    str     w4, [x2, #UART_CR]
    ret

// uart_putc(w0 = byte); '\n' -> "\r\n". Preserves x0..x8.
uart_putc:
    LDG     x10, g_uart
    cmp     w0, #10
    b.ne    2f
    mov     w11, #13
1:  ldr     w12, [x10, #UART_FR]
    tbnz    w12, #5, 1b                 // TXFF
    str     w11, [x10, #UART_DR]
2:  ldr     w12, [x10, #UART_FR]
    tbnz    w12, #5, 2b
    str     w0, [x10, #UART_DR]
    ret

// uart_getc -> w0 (blocking)
uart_getc:
    LDG     x10, g_uart
1:  ldr     w12, [x10, #UART_FR]
    tbnz    w12, #4, 1b                 // RXFE
    ldr     w0, [x10, #UART_DR]
    and     w0, w0, #0xFF
    ret

// uart_puts(x0 = NUL-terminated string)
uart_puts:
    stp     x29, x30, [sp, #-32]!
    str     x19, [sp, #16]
    mov     x19, x0
1:  ldrb    w0, [x19], #1
    cbz     w0, 2f
    bl      uart_putc
    b       1b
2:  ldr     x19, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret

// print_u64(x0) in decimal
print_u64:
    stp     x29, x30, [sp, #-64]!
    str     x19, [sp, #16]
    add     x1, sp, #56                 // buffer [sp+24 .. sp+56)
    strb    wzr, [x1]
    mov     x2, #10
1:  udiv    x3, x0, x2
    msub    x4, x3, x2, x0
    add     w4, w4, #'0'
    sub     x1, x1, #1
    strb    w4, [x1]
    mov     x0, x3
    cbnz    x0, 1b
    mov     x0, x1
    bl      uart_puts
    ldr     x19, [sp, #16]
    ldp     x29, x30, [sp], #64
    ret

// print_2dp(x0 = value * 100) -> "integer.dd"
print_2dp:
    stp     x29, x30, [sp, #-32]!
    str     x19, [sp, #16]
    mov     x2, #100
    udiv    x1, x0, x2
    msub    x19, x1, x2, x0
    mov     x0, x1
    bl      print_u64
    mov     w0, #'.'
    bl      uart_putc
    mov     x2, #10
    udiv    x0, x19, x2
    add     w0, w0, #'0'
    bl      uart_putc
    mov     x2, #10
    udiv    x1, x19, x2
    msub    x0, x1, x2, x19
    add     w0, w0, #'0'
    bl      uart_putc
    ldr     x19, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret
.ltorg

.bss
.balign 8
.global g_periph, g_uart
g_periph: .skip 8
g_uart:   .skip 8
