// REQUIRES: clang-tidy, systemc
// RUN: circt-translate %s --export-systemc > %t.cpp
// RUN: clang-tidy --extra-arg=-frtti %t.cpp

emitc.include <"systemc">

systemc.module @module (%port0: !systemc.in<i1>, %port1: !systemc.inout<i64>, %port2: !systemc.out<i512>, %port3: !systemc.out<i1024>, %port4: !systemc.out<i1>) {
  %sig = systemc.signal : !systemc.signal<i64>
  systemc.ctor {
    systemc.method %add
    systemc.thread %add
  }
  %add = systemc.func {
    %0 = systemc.signal.read %port1 : !systemc.inout<i64>
    systemc.signal.write %sig, %0 : !systemc.signal<i64>
    %1 = hw.constant 42 : i512
    systemc.signal.write %port2, %1 : !systemc.out<i512>
    %2 = hw.constant 1 : i1
    systemc.signal.write %port4, %2 : !systemc.out<i1>
    %3 = hw.constant 0 : i1
    systemc.signal.write %port4, %3 : !systemc.out<i1>
    %4 = comb.add %0, %0, %0 : i64
    %5 = comb.sub %4, %0 : i64
    %6 = comb.mul %0, %0, %5 : i64
    %7 = comb.and %6, %0, %0 : i64
    %8 = comb.or %0, %7, %0 : i64
    %9 = comb.xor %0, %0, %8 : i64
    %10 = comb.divu %9, %0 : i64
    %11 = comb.modu %0, %10 : i64
    %12 = comb.shl %0, %11 : i64
    %13 = comb.shru %12, %0 : i64
    systemc.signal.write %sig, %13 : !systemc.signal<i64>

    %14 = comb.concat %0, %0, %0, %0 : i64, i64, i64, i64
    %15 = comb.extract %14 from 16 : (i256) -> i64
    systemc.signal.write %sig, %15 : !systemc.signal<i64>
  }
}
