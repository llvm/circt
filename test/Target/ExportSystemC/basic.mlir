// RUN: circt-translate %s --export-systemc | FileCheck %s

// CHECK-LABEL: // stdout.h
// CHECK-NEXT: #ifndef STDOUT_H
// CHECK-NEXT: #define STDOUT_H

// CHECK: #include <systemc.h>
// CHECK: #include "nosystemheader"

emitc.include <"systemc.h">
emitc.include "nosystemheader"

// CHECK-EMPTY:
// CHECK-LABEL: SC_MODULE(submodule) {
systemc.module @submodule (%in0: !systemc.in<i32>, %in1: !systemc.in<i32>, %out0: !systemc.out<i32>) {
// CHECK-NEXT: sc_in<sc_uint<32>> in0;
// CHECK-NEXT: sc_in<sc_uint<32>> in1;
// CHECK-NEXT: sc_out<sc_uint<32>> out0;
// CHECK-NEXT: };
}

// CHECK-EMPTY:
// CHECK-LABEL: SC_MODULE(basic) {
systemc.module @basic (%port0: !systemc.in<i1>, %port1: !systemc.inout<i64>, %port2: !systemc.out<i512>, %port3: !systemc.out<i1024>, %port4: !systemc.out<i1>) {
  // CHECK-NEXT: sc_in<bool> port0;
  // CHECK-NEXT: sc_inout<sc_uint<64>> port1;
  // CHECK-NEXT: sc_out<sc_biguint<512>> port2;
  // CHECK-NEXT: sc_out<sc_bv<1024>> port3;
  // CHECK-NEXT: sc_out<bool> port4;
  // CHECK-NEXT: sc_signal<sc_uint<64>> sig;
  %sig = systemc.signal : !systemc.signal<i64>
  // CHECK-NEXT: sc_signal<sc_uint<32>> channel;
  %channel = systemc.signal : !systemc.signal<i32>
  // CHECK-NEXT: submodule submoduleInstance;
  %submoduleInstance = systemc.instance.decl @submodule : !systemc.module<submodule(in0: !systemc.in<i32>, in1: !systemc.in<i32>, out0: !systemc.out<i32>)>
  // CHECK-EMPTY: 
  // CHECK-NEXT: SC_CTOR(basic) {
  systemc.ctor {
    // CHECK-NEXT: submoduleInstance.in0(channel);
    systemc.instance.bind_port %submoduleInstance["in0"] to %channel : !systemc.module<submodule(in0: !systemc.in<i32>, in1: !systemc.in<i32>, out0: !systemc.out<i32>)>, !systemc.signal<i32>
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
