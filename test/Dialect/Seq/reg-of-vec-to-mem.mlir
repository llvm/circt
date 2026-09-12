// RUN: circt-opt %s --seq-reg-of-vec-to-mem | FileCheck %s

// CHECK-LABEL: hw.module private @complex_mem
hw.module private @complex_mem(in %CLK : i1, in %D : i46, in %ADR : i13, in %WE : i1, in %ME : i1, out Q : i46) {
    %true = hw.constant true
    %c0_i46 = hw.constant 0 : i46
    %0 = comb.xor %WE, %true : i1
    %1 = comb.and %ME, %0 : i1
    %2 = hw.array_get %mem_core[%ADR] : !hw.array<8192xi46>, i13
    %3 = comb.xor %1, %true : i1
    %4 = comb.mux %3, %c0_i46, %2 : i46
    %5 = seq.to_clock %CLK
    %6 = comb.mux %1, %4, %Q_int : i46
    %Q_int = seq.firreg %6 clock %5 : i46
    // NOTE: The transformation cannot identify the memory read enable signal.
    %7 = comb.and %ME, %WE : i1
    %8 = hw.array_inject %mem_core[%ADR], %D : !hw.array<8192xi46>, i13
    %9 = comb.mux %7, %8, %mem_core : !hw.array<8192xi46>
    %mem_core = seq.firreg %9 clock %5 : !hw.array<8192xi46>
    hw.output %Q_int : i46
}

// CHECK: %[[clock:.+]] = seq.to_clock %CLK
// CHECK: %[[V6:.+]] = comb.and %ME, %WE : i1
// CHECK: %mem = seq.firmem 0, 1, undefined, undefined : <8192 x 46, mask 1>
// CHECK: %[[READ:.+]] = seq.firmem.read_port %mem[%ADR], clock %[[clock]] enable %true
// CHECK: seq.firmem.write_port %mem[%ADR] = %D, clock %[[clock]] enable %[[V6]]
// CHECK-NOT: seq.firreg %{{.*}} : !hw.array<8192xi46>
// CHECK-NOT: hw.array_get
// CHECK-NOT: hw.array_inject

// Simple test case
// CHECK-LABEL: hw.module @simple_mem
hw.module @simple_mem(in %clk : i1, in %addr : i2, in %data : i8, in %we : i1, out out : i8) {
    %clock = seq.to_clock %clk
    %true = hw.constant true
    %read = hw.array_get %mem[%addr] : !hw.array<4xi8>, i2
    %write = hw.array_inject %mem[%addr], %data : !hw.array<4xi8>, i2
    %next = comb.mux %we, %write, %mem : !hw.array<4xi8>
    %mem = seq.firreg %next clock %clock : !hw.array<4xi8>
    hw.output %read : i8
}

// CHECK: %[[clock:.+]] = seq.to_clock %clk
// CHECK: %mem = seq.firmem 0, 1, undefined, undefined : <4 x 8, mask 1>
// CHECK: %[[READ:.+]] = seq.firmem.read_port %mem[%addr], clock %[[clock]] enable %true
// CHECK: seq.firmem.write_port %mem[%addr] = %data, clock %[[clock]] enable %we
// CHECK: hw.output %[[READ]] : i8
// CHECK-NOT: seq.firreg %{{.*}} : !hw.array<8192xi46>
// CHECK-NOT: hw.array_get
// CHECK-NOT: hw.array_inject


hw.module @single_el_mem(in %clk : i1, in %addr : i0, in %data : i8, in %we : i1) {
    %clock = seq.to_clock %clk
    %read = hw.array_get %msingel[%addr] : !hw.array<1xi8>, i0
    %write = hw.array_inject %msingel[%addr], %data : !hw.array<1xi8>, i0
    %next = comb.mux %we, %write, %msingel : !hw.array<1xi8>
    %msingel = seq.firreg %next clock %clock : !hw.array<1xi8>
}

// CHECK: %[[clock:.+]] = seq.to_clock %clk
// CHECK: %mem = seq.firmem 0, 1, undefined, undefined : <1 x 8, mask 1>
// CHECK: %[[FALSE:.+]] = hw.constant false
// CHECK: %[[TRUE:.+]] = hw.constant true
// CHECK: %[[READ:.+]] = seq.firmem.read_port %mem[%[[FALSE]]], clock %[[clock]] enable %[[TRUE]]
// CHECK: %[[FALSE:.+]] = hw.constant false
// CHECK: seq.firmem.write_port %mem[%[[FALSE]]] = %data, clock %[[clock]] enable %we
// CHECK-NOT: seq.firreg %{{.*}} : !hw.array<8192xi46>
// CHECK-NOT: hw.array_get
// CHECK-NOT: hw.array_inject

// Test that transformation is skipped when mux has multiple uses
// CHECK-LABEL: hw.module @shared_mux_test(
hw.module @shared_mux_test(in %clk: i1, in %addr: i2, in %data: i8, in %we: i1, out other_out: !hw.array<4xi8>) {
  %clock = seq.to_clock %clk
  %write = hw.array_inject %mem[%addr], %data : !hw.array<4xi8>, i2
  %next = comb.mux %we, %write, %mem : !hw.array<4xi8>
  // CHECK: %mem = seq.firreg %{{.*}} clock %{{.*}} : !hw.array<4xi8>
  %mem = seq.firreg %next clock %clock : !hw.array<4xi8>
  
  // Mux result used elsewhere - should prevent transformation
  hw.output %next : !hw.array<4xi8>
}

// Test that transformation is skipped when register has multiple uses
// CHECK-LABEL: hw.module @shared_reg_test(
hw.module @shared_reg_test(in %clk: i1, in %addr: i2, in %data: i8, in %we: i1, out reg_out: !hw.array<4xi8>, out read_out: i8) {
  %clock = seq.to_clock %clk
  %read = hw.array_get %mem[%addr] : !hw.array<4xi8>, i2
  %write = hw.array_inject %mem[%addr], %data : !hw.array<4xi8>, i2
  %next = comb.mux %we, %write, %mem : !hw.array<4xi8>
  // CHECK: %mem = seq.firreg %{{.*}} clock %{{.*}} : !hw.array<4xi8>
  %mem = seq.firreg %next clock %clock : !hw.array<4xi8>
  
  // Register used elsewhere - should prevent transformation
  hw.output %mem, %read : !hw.array<4xi8>, i8
}

// Recognize a chain of equal-width masked writes, including an outer global
// enable mux. The lane updates are deliberately listed from low to high bits.
// CHECK-LABEL: hw.module @masked_mem(
hw.module @masked_mem(in %clk: i1, in %addr: i2, in %data: i32, in %mask: i4, in %enable: i1, out read: i32) {
  %clock = seq.to_clock %clk
  %m0 = comb.extract %mask from 0 : (i4) -> i1
  %m1 = comb.extract %mask from 1 : (i4) -> i1
  %m2 = comb.extract %mask from 2 : (i4) -> i1
  %m3 = comb.extract %mask from 3 : (i4) -> i1
  %d0 = comb.extract %data from 0 : (i32) -> i8
  %d1 = comb.extract %data from 8 : (i32) -> i8
  %d2 = comb.extract %data from 16 : (i32) -> i8
  %d3 = comb.extract %data from 24 : (i32) -> i8

  %old0 = hw.array_get %mem[%addr] : !hw.array<4xi32>, i2
  %high0 = comb.extract %old0 from 8 : (i32) -> i24
  %word0 = comb.concat %high0, %d0 : i24, i8
  %inject0 = hw.array_inject %mem[%addr], %word0 : !hw.array<4xi32>, i2
  %state0 = comb.mux %m0, %inject0, %mem : !hw.array<4xi32>

  %old1 = hw.array_get %state0[%addr] : !hw.array<4xi32>, i2
  %high1 = comb.extract %old1 from 16 : (i32) -> i16
  %low1 = comb.extract %old1 from 0 : (i32) -> i8
  %word1 = comb.concat %high1, %d1, %low1 : i16, i8, i8
  %inject1 = hw.array_inject %state0[%addr], %word1 : !hw.array<4xi32>, i2
  %state1 = comb.mux %m1, %inject1, %state0 : !hw.array<4xi32>

  %old2 = hw.array_get %state1[%addr] : !hw.array<4xi32>, i2
  %high2 = comb.extract %old2 from 24 : (i32) -> i8
  %low2 = comb.extract %old2 from 0 : (i32) -> i16
  %word2 = comb.concat %high2, %d2, %low2 : i8, i8, i16
  %inject2 = hw.array_inject %state1[%addr], %word2 : !hw.array<4xi32>, i2
  %state2 = comb.mux %m2, %inject2, %state1 : !hw.array<4xi32>

  %old3 = hw.array_get %state2[%addr] : !hw.array<4xi32>, i2
  %low3 = comb.extract %old3 from 0 : (i32) -> i24
  %word3 = comb.concat %d3, %low3 : i8, i24
  %inject3 = hw.array_inject %state2[%addr], %word3 : !hw.array<4xi32>, i2
  %state3 = comb.mux %m3, %inject3, %state2 : !hw.array<4xi32>

  %next = comb.mux %enable, %state3, %mem : !hw.array<4xi32>
  %mem = seq.firreg %next clock %clock : !hw.array<4xi32>
  %read = hw.array_get %mem[%addr] : !hw.array<4xi32>, i2
  hw.output %read : i32
}

// CHECK: %[[CLOCK:.+]] = seq.to_clock %clk
// CHECK: %[[M0:.+]] = comb.extract %mask from 0 : (i4) -> i1
// CHECK: %[[M1:.+]] = comb.extract %mask from 1 : (i4) -> i1
// CHECK: %[[M2:.+]] = comb.extract %mask from 2 : (i4) -> i1
// CHECK: %[[M3:.+]] = comb.extract %mask from 3 : (i4) -> i1
// CHECK: %[[D0:.+]] = comb.extract %data from 0 : (i32) -> i8
// CHECK: %[[D1:.+]] = comb.extract %data from 8 : (i32) -> i8
// CHECK: %[[D2:.+]] = comb.extract %data from 16 : (i32) -> i8
// CHECK: %[[D3:.+]] = comb.extract %data from 24 : (i32) -> i8
// CHECK: %mem = seq.firmem 0, 1, undefined, undefined : <4 x 32, mask 4>
// CHECK: %[[READ:.+]] = seq.firmem.read_port %mem[%addr], clock %[[CLOCK]]
// CHECK: %[[DATA:.+]] = comb.concat %[[D3]], %[[D2]], %[[D1]], %[[D0]] : i8, i8, i8, i8
// CHECK: %[[MASK:.+]] = comb.concat %[[M3]], %[[M2]], %[[M1]], %[[M0]] : i1, i1, i1, i1
// CHECK: seq.firmem.write_port %mem[%addr] = %[[DATA]], clock %[[CLOCK]] enable %enable mask %[[MASK]]
// CHECK: hw.output %[[READ]] : i32
// CHECK-NOT: seq.firreg %{{.*}} : !hw.array<4xi32>
// CHECK-NOT: hw.array_get
// CHECK-NOT: hw.array_inject

// A memory without an external read access is still a valid write-only memory.
// CHECK-LABEL: hw.module @masked_write_only(
hw.module @masked_write_only(in %clk: i1, in %addr: i2, in %lo: i8, in %hi: i8, in %lo_en: i1, in %hi_en: i1) {
  %clock = seq.to_clock %clk
  %old0 = hw.array_get %mem[%addr] : !hw.array<4xi16>, i2
  %old_hi = comb.extract %old0 from 8 : (i16) -> i8
  %word0 = comb.concat %old_hi, %lo : i8, i8
  %inject0 = hw.array_inject %mem[%addr], %word0 : !hw.array<4xi16>, i2
  %state0 = comb.mux %lo_en, %inject0, %mem : !hw.array<4xi16>
  %old1 = hw.array_get %state0[%addr] : !hw.array<4xi16>, i2
  %old_lo = comb.extract %old1 from 0 : (i16) -> i8
  %word1 = comb.concat %hi, %old_lo : i8, i8
  %inject1 = hw.array_inject %state0[%addr], %word1 : !hw.array<4xi16>, i2
  %next = comb.mux %hi_en, %inject1, %state0 : !hw.array<4xi16>
  %mem = seq.firreg %next clock %clock : !hw.array<4xi16>
}

// CHECK: %[[CLOCK:.+]] = seq.to_clock %clk
// CHECK: %mem = seq.firmem 0, 1, undefined, undefined : <4 x 16, mask 2>
// CHECK-NOT: seq.firmem.read_port %mem
// CHECK: %[[DATA:.+]] = comb.concat %hi, %lo : i8, i8
// CHECK: %[[MASK:.+]] = comb.concat %hi_en, %lo_en : i1, i1
// CHECK: %[[TRUE:.+]] = hw.constant true
// CHECK: seq.firmem.write_port %mem[%addr] = %[[DATA]], clock %[[CLOCK]] enable %[[TRUE]] mask %[[MASK]]
// CHECK-NOT: seq.firreg %{{.*}} : !hw.array<4xi16>

// Do not combine writes which target different addresses.
// CHECK-LABEL: hw.module @masked_different_addresses(
hw.module @masked_different_addresses(in %clk: i1, in %addr0: i2, in %addr1: i2, in %lo: i8, in %hi: i8, in %lo_en: i1, in %hi_en: i1, out read: i16) {
  %clock = seq.to_clock %clk
  %old0 = hw.array_get %mem[%addr0] : !hw.array<4xi16>, i2
  %old_hi = comb.extract %old0 from 8 : (i16) -> i8
  %word0 = comb.concat %old_hi, %lo : i8, i8
  %inject0 = hw.array_inject %mem[%addr0], %word0 : !hw.array<4xi16>, i2
  %state0 = comb.mux %lo_en, %inject0, %mem : !hw.array<4xi16>
  %old1 = hw.array_get %state0[%addr1] : !hw.array<4xi16>, i2
  %old_lo = comb.extract %old1 from 0 : (i16) -> i8
  %word1 = comb.concat %hi, %old_lo : i8, i8
  %inject1 = hw.array_inject %state0[%addr1], %word1 : !hw.array<4xi16>, i2
  %next = comb.mux %hi_en, %inject1, %state0 : !hw.array<4xi16>
  %mem = seq.firreg %next clock %clock : !hw.array<4xi16>
  %read = hw.array_get %mem[%addr0] : !hw.array<4xi16>, i2
  hw.output %read : i16
}

// CHECK: %[[CLOCK:.+]] = seq.to_clock %clk
// CHECK: seq.firreg %{{.*}} clock %[[CLOCK]] : !hw.array<4xi16>
// CHECK-NOT: seq.firmem
