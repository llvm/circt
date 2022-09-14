// REQUIRES: clang-tidy, systemc
// RUN: circt-translate %s --export-systemc > %t.cpp
// RUN: clang-tidy --extra-arg=-frtti %t.cpp

emitc.include <"systemc.h">

systemc.module @submodule (%in0: !systemc.in<!systemc.uint<32>>, %in1: !systemc.in<!systemc.uint<32>>, %out0: !systemc.out<!systemc.uint<32>>) {}

systemc.module @module (%port0: !systemc.in<i1>, %port1: !systemc.inout<!systemc.uint<64>>, %port2: !systemc.out<i64>, %port3: !systemc.out<!systemc.bv<1024>>, %port4: !systemc.out<i1>) {
  %submoduleInstance = systemc.instance.decl @submodule : !systemc.module<submodule(in0: !systemc.in<!systemc.uint<32>>, in1: !systemc.in<!systemc.uint<32>>, out0: !systemc.out<!systemc.uint<32>>)>
  %sig = systemc.signal : !systemc.signal<!systemc.uint<64>>
  %channel = systemc.signal : !systemc.signal<!systemc.uint<32>>
  %testvar = systemc.cpp.variable : i32
  %c42_i32 = hw.constant 42 : i32
  %testvarwithinit = systemc.cpp.variable %c42_i32 : i32
  systemc.ctor {
    systemc.instance.bind_port %submoduleInstance["in0"] to %channel : !systemc.module<submodule(in0: !systemc.in<!systemc.uint<32>>, in1: !systemc.in<!systemc.uint<32>>, out0: !systemc.out<!systemc.uint<32>>)>, !systemc.signal<!systemc.uint<32>>
    systemc.method %add
    systemc.thread %add
  }
  %add = systemc.func {
    %0 = systemc.signal.read %port1 : !systemc.inout<!systemc.uint<64>>
    systemc.signal.write %sig, %0 : !systemc.signal<!systemc.uint<64>>
    %1 = hw.constant 42 : i64
    systemc.signal.write %port2, %1 : !systemc.out<i64>
    %2 = hw.constant 1 : i1
    systemc.signal.write %port4, %2 : !systemc.out<i1>
    %3 = hw.constant 0 : i1
    systemc.signal.write %port4, %3 : !systemc.out<i1>
    systemc.cpp.assign %testvar = %c42_i32 : i32
    systemc.cpp.assign %testvarwithinit = %testvar : i32
  }
}

systemc.module @nativeCTypes (%port0: !systemc.in<i1>,
                              %port1: !systemc.in<i8>,
                              %port2: !systemc.in<i16>,
                              %port3: !systemc.in<i32>,
                              %port4: !systemc.in<i64>,
                              %port5: !systemc.in<si1>,
                              %port6: !systemc.in<si8>,
                              %port7: !systemc.in<si16>,
                              %port8: !systemc.in<si32>,
                              %port9: !systemc.in<si64>,
                              %port10: !systemc.in<ui1>,
                              %port11: !systemc.in<ui8>,
                              %port12: !systemc.in<ui16>,
                              %port13: !systemc.in<ui32>,
                              %port14: !systemc.in<ui64>) {}

systemc.module @systemCTypes (%p0: !systemc.in<!systemc.int_base>,
                              %p1: !systemc.in<!systemc.int<32>>,
                              %p2: !systemc.in<!systemc.uint_base>,
                              %p3: !systemc.in<!systemc.uint<32>>,
                              %p4: !systemc.in<!systemc.signed>,
                              %p5: !systemc.in<!systemc.bigint<256>>,
                              %p6: !systemc.in<!systemc.unsigned>,
                              %p7: !systemc.in<!systemc.biguint<256>>,
                              %p8: !systemc.in<!systemc.bv_base>,
                              %p9: !systemc.in<!systemc.bv<1024>>,
                              %p10: !systemc.in<!systemc.lv_base>,
                              %p11: !systemc.in<!systemc.lv<1024>>,
                              %p12: !systemc.in<!systemc.logic>) {}
