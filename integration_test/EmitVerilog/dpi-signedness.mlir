// REQUIRES: verilator
// RUN: split-file %s %t
// RUN: circt-opt --export-verilog %t/input.mlir -o %t/lowered.mlir > %t/top.sv
// RUN: verilator --binary --top-module Top --Mdir %t/obj -Wno-fatal %t/top.sv %t/driver.cpp > %t/build.log 2>&1
// RUN: FileCheck %s --check-prefix=ABI < %t/obj/VTop__Dpi.h
// RUN: %t/obj/VTop

// ABI: void exchange(unsigned char* u8, unsigned short* u16, unsigned int* u32, unsigned long long* u64);
// ABI: void inspect(const svBitVecVal* narrow, const svBitVecVal* wide, char s8, short s16, int s32, long long s64, svBit flag);
// ABI: unsigned char result8();
// ABI: void update(unsigned int* state);

//--- input.mlir
sv.func private @exchange(out u8: ui8, out u16: ui16, out u32: ui32, out u64: ui64)
sv.func.dpi.import @exchange
sv.func private @inspect(in %narrow: si7, in %wide: ui65, in %s8: si8,
                        in %s16: si16, in %s32: si32, in %s64: si64,
                        in %flag: ui1)
sv.func.dpi.import @inspect
sv.func private @result8(out result: ui8 {sv.func.explicitly_returned})
sv.func.dpi.import @result8
sv.func private @update(in %state: !hw.inout<ui32>)
sv.func.dpi.import @update
hw.module @Top() {
  sv.verbatim "initial begin\0A  byte unsigned u8; shortint unsigned u16; int unsigned u32; longint unsigned u64;\0A  exchange(u8, u16, u32, u64);\0A  if (u8 != 8'hff || u16 != 16'hffff || u32 != 32'hffffffff || u64 != 64'hffffffffffffffff) $fatal;\0A  inspect(-7'sd1, 65'd1 << 64, -8'sd1, -16'sd1, -32'sd1, -64'sd1, 1'b1);\0A  if (result8() != 8'hff) $fatal;\0A  update(u32);\0A  if (u32 != 32'h80000000) $fatal;\0A  $finish;\0Aend"
  hw.output
}

//--- driver.cpp
#include "VTop__Dpi.h"
#include <cassert>
#include <limits>

extern "C" void exchange(unsigned char *u8, unsigned short *u16,
                         unsigned int *u32, unsigned long long *u64) {
  *u8 = std::numeric_limits<unsigned char>::max();
  *u16 = std::numeric_limits<unsigned short>::max();
  *u32 = std::numeric_limits<unsigned int>::max();
  *u64 = std::numeric_limits<unsigned long long>::max();
}
extern "C" void inspect(const svBitVecVal *narrow, const svBitVecVal *wide,
                        char s8, short s16, int s32, long long s64,
                        svBit flag) {
  assert(narrow[0] == 127 && wide[0] == 0 && wide[1] == 0 && wide[2] == 1);
  assert(static_cast<signed char>(s8) == -1 && s16 == -1 && s32 == -1 &&
         s64 == -1 && flag == 1);
}
extern "C" unsigned char result8() { return 255; }
extern "C" void update(unsigned int *state) {
  assert(*state == std::numeric_limits<unsigned int>::max());
  *state = 0x80000000u;
}
