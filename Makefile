# pi-llm.asm — bare-metal LLM for Raspberry Pi (AArch64), 100% assembly
CROSS  ?= aarch64-linux-gnu-
AS     := $(CROSS)as
LD     := $(CROSS)ld
OBJCOPY:= $(CROSS)objcopy
MODEL  ?= build/model.piml
BUILD  := build
SRCS   := boot uart math llm
OBJS   := $(addprefix $(BUILD)/,$(addsuffix .o,$(SRCS))) $(BUILD)/model.o

.PHONY: all run clean sdcard
all: $(BUILD)/kernel8.img

$(BUILD)/%.o: src/%.s src/defs.inc | $(BUILD)
	$(AS) -mcpu=cortex-a53 -I src -o $@ $<

$(BUILD)/model.o: src/model.s $(MODEL) | $(BUILD)
	$(AS) -I $(dir $(MODEL)) -o $@ $<

$(BUILD)/kernel8.elf: $(OBJS) src/linker.ld
	$(LD) -T src/linker.ld -nostdlib -o $@ $(OBJS)

$(BUILD)/kernel8.img: $(BUILD)/kernel8.elf
	$(OBJCOPY) -O binary $< $@
	@ls -l $@

$(BUILD):
	mkdir -p $(BUILD)

# Emulation: Raspberry Pi 3B in QEMU, console on UART0 (PL011)
run: $(BUILD)/kernel8.img
	qemu-system-aarch64 -M raspi3b -kernel $< -serial stdio -display none

sdcard: $(BUILD)/kernel8.img
	bash tools/mksdcard.sh

clean:
	rm -f $(BUILD)/*.o $(BUILD)/kernel8.*
