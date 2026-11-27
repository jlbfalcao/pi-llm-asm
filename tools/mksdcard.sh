#!/usr/bin/env bash
# Builds build/sdcard/ (copy everything to the root of a FAT32 SD card) and build/pi-llm-sdcard.zip
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=build/sdcard; FW=build/fw
mkdir -p "$OUT" "$FW"
BASE=https://raw.githubusercontent.com/raspberrypi/firmware/master/boot
for f in bootcode.bin start.elf fixup.dat start4.elf fixup4.dat LICENCE.broadcom; do
  [ -s "$FW/$f" ] || curl -sSfL -o "$FW/$f" "$BASE/$f"
  cp "$FW/$f" "$OUT/"
done
cp build/kernel8.img "$OUT/"
cat > "$OUT/config.txt" <<'CFG'
# pi-llm.asm — bare-metal AArch64 boot (no Linux)
arm_64bit=1
kernel=kernel8.img
kernel_address=0x80000
# UART0 (PL011) on GPIO14 (TX, pin 8) / GPIO15 (RX, pin 10), 115200 8N1
enable_uart=1
dtoverlay=disable-bt
init_uart_clock=48000000
CFG
rm -f build/pi-llm-sdcard.zip
(cd "$OUT" && zip -q -9 ../pi-llm-sdcard.zip *)
ls -l build/pi-llm-sdcard.zip
