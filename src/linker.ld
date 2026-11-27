/* kernel8.img: loaded by the Raspberry Pi firmware at 0x80000 */
ENTRY(_start)
SECTIONS
{
    . = 0x80000;
    .text : { KEEP(*(.text.boot)) *(.text*) }
    .rodata : ALIGN(64) { *(.rodata*) }
    .data : ALIGN(16) { *(.data*) }
    .bss (NOLOAD) : ALIGN(4096) {
        __bss_start = .;
        *(.bss*) *(COMMON)
        . = ALIGN(16);
        __bss_end = .;
    }
    . = ALIGN(4096);
    __heap_start = .;
    /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
}
