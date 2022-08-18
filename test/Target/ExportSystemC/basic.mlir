// RUN: circt-translate %s --export-systemc | FileCheck %s

// CHECK-LABEL: // stdout.h
// CHECK-NEXT: #ifndef STDOUT_H
// CHECK-NEXT: #define STDOUT_H

// CHECK: #include <systemc>
// CHECK: #include "nosystemheader"

emitc.include <"systemc">
emitc.include "nosystemheader"

// CHECK-EMPTY:
// CHECK-LABEL: SC_MODULE(basic) {
systemc.module @basic (%port0: !systemc.in<i1>, %port1: !systemc.inout<i64>, %port2: !systemc.out<i512>, %port3: !systemc.out<i1024>, %port4: !systemc.out<i1>) {
  // CHECK-NEXT: sc_core::sc_in<bool> port0;
  // CHECK-NEXT: sc_core::sc_inout<sc_dt::sc_uint<64>> port1;
  // CHECK-NEXT: sc_core::sc_out<sc_dt::sc_biguint<512>> port2;
  // CHECK-NEXT: sc_core::sc_out<sc_dt::sc_bv<1024>> port3;
  // CHECK-NEXT: sc_core::sc_out<bool> port4;
  // CHECK-NEXT: sc_core::sc_signal<sc_dt::sc_uint<64>> sig;
  %sig = systemc.signal : !systemc.signal<i64>
  // CHECK-EMPTY: 
  // CHECK-NEXT: SC_CTOR(basic) {
  systemc.ctor {
    // CHECK-NEXT: SC_METHOD(add);
    systemc.method %add
    // CHECK-NEXT: SC_THREAD(add);
    systemc.thread %add
  // CHECK-NEXT: }
  }
  // CHECK-EMPTY:
  // CHECK-NEXT: void add() {
  %add = systemc.func {
    // CHECK-NEXT: sig.write(port1.read());
    %0 = systemc.signal.read %port1 : !systemc.inout<i64>
    systemc.signal.write %sig, %0 : !systemc.signal<i64>
    // CHECK-NEXT: port2.write(42);
    %1 = hw.constant 42 : i512
    systemc.signal.write %port2, %1 : !systemc.out<i512>
    // CHECK-NEXT: port4.write(true);
    %2 = hw.constant 1 : i1
    systemc.signal.write %port4, %2 : !systemc.out<i1>
    // CHECK-NEXT: port4.write(false);
    %3 = hw.constant 0 : i1
    systemc.signal.write %port4, %3 : !systemc.out<i1>
  // CHECK-NEXT: }
  }
// CHECK-NEXT: };
}

// CHECK: #endif // STDOUT_H
