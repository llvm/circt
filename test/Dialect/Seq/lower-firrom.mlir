// RUN: circt-opt --lower-seq-firrom %s --verify-diagnostics | FileCheck %s

// CHECK: hw.generator.schema @FIRRTLRom, "FIRRTL_Rom", ["depth", "numReadPorts", "readLatency", "width", "initFilename", "initIsBinary", "initIsInline"]

// CHECK: sv.macro.decl @SomeMacro
sv.macro.decl @SomeMacro

// CHECK: hw.module.generated @m0_rom1_12x42, @FIRRTLRom(%R0_addr: i4, %R0_en: i1, %R0_clk: i1) -> (R0_data: i42) attributes {depth = 12 : i64, initFilename = "", initIsBinary = false, initIsInline = false, numReadPorts = 1 : ui32, readLatency = 0 : ui32, width = 42 : ui32}

// CHECK-LABEL: hw.module @Foo
hw.module @Foo(%clk: i1, %en: i1, %addr: i4, %wdata: i42, %wmode: i1, %mask2: i2, %mask3: i3, %mask6: i6) {
  // CHECK-NEXT: [[TMP0:%.+]] = hw.instance "m0_rom1A_ext" @m0_rom1_12x42(R0_addr: %addr: i4, R0_en: %en: i1, R0_clk: %clk: i1) -> (R0_data: i42)
  // CHECK-NEXT: [[TMP1:%.+]] = hw.instance "m0_rom1B_ext" @m0_rom1_12x42(R0_addr: %addr: i4, R0_en: %en: i1, R0_clk: %clk: i1) -> (R0_data: i42)
  // CHECK-NEXT: comb.xor [[TMP0]], [[TMP1]]
  %m0_rom1A = seq.firrom 0 : <12 x 42>
  %m0_rom1B = seq.firrom 0 : <12 x 42>
  %0 = seq.firrom.read_port %m0_rom1A[%addr], clock %clk enable %en : <12 x 42>
  %1 = seq.firrom.read_port %m0_rom1B[%addr], clock %clk enable %en : <12 x 42>
  comb.xor %0, %1 : i42
}

// CHECK: hw.module.generated @m1_rom1_24x1337, @FIRRTLRom() attributes {depth = 24 : i64, initFilename = "", initIsBinary = false, initIsInline = false, numReadPorts = 0 : ui32, output_file = "foo", readLatency = 0 : ui32, width = 1337 : ui32}
// CHECK: hw.module.generated @m1_rom2_24x1337, @FIRRTLRom() attributes {depth = 24 : i64, initFilename = "", initIsBinary = false, initIsInline = false, numReadPorts = 0 : ui32, output_file = "bar", readLatency = 0 : ui32, width = 1337 : ui32}

// CHECK-LABEL: hw.module @SeparateOutputFiles
hw.module @SeparateOutputFiles() {
  // CHECK-NEXT: hw.instance "m1_rom1_ext" @m1_rom1_24x1337(
  // CHECK-NEXT: hw.instance "m1_rom2_ext" @m1_rom2_24x1337(
  %m1_rom1 = seq.firrom 0 {output_file = "foo"} : <24 x 1337>
  %m1_rom2 = seq.firrom 0 {output_file = "bar"} : <24 x 1337>
}

// CHECK: hw.module.generated @foo_m2_rom1_24x9001, @FIRRTLRom() attributes {depth = 24 : i64, initFilename = "", initIsBinary = false, initIsInline = false, numReadPorts = 0 : ui32, readLatency = 0 : ui32, width = 9001 : ui32}
// CHECK: hw.module.generated @bar_m2_rom2_24x9001, @FIRRTLRom() attributes {depth = 24 : i64, initFilename = "", initIsBinary = false, initIsInline = false, numReadPorts = 0 : ui32, readLatency = 0 : ui32, width = 9001 : ui32}
// CHECK: hw.module.generated @uwu_m2_rom_24x9001, @FIRRTLRom() attributes {depth = 24 : i64, initFilename = "", initIsBinary = false, initIsInline = false, numReadPorts = 0 : ui32, readLatency = 0 : ui32, width = 9001 : ui32}

// CHECK-LABEL: hw.module @SeparatePrefices
hw.module @SeparatePrefices() {
  // CHECK-NEXT: hw.instance "m2_rom1_ext" @foo_m2_rom1_24x9001(
  %m2_rom1 = seq.firrom 0 {prefix = "foo_"} : <24 x 9001>
  // CHECK-NEXT: hw.instance "m2_rom2_ext" @bar_m2_rom2_24x9001(
  %m2_rom2 = seq.firrom 0 {prefix = "bar_"} : <24 x 9001>
  // CHECK-NEXT: hw.instance "m2_rom3_ext" @uwu_m2_rom_24x9001(
  // CHECK-NEXT: hw.instance "m2_rom4_ext" @uwu_m2_rom_24x9001(
  %m2_rom3 = seq.firrom 0 {prefix = "uwu_"} : <24 x 9001>
  %m2_rom4 = seq.firrom 0 {prefix = "uwu_"} : <24 x 9001>
}
