// RUN: circt-opt %s --seq-reg-of-vec-to-mem | FileCheck %s

// CHECK-LABEL: hw.module private @sf_tagmem_external_mem_a
hw.module private @sf_tagmem_external_mem_a(in %CLK : i1, in %D : i46, in %ADR : i13, in %WE : i1, in %ME : i1, out Q : i46) {
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