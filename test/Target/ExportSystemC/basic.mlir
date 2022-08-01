// RUN: circt-translate %s --export-systemc | FileCheck %s

// CHECK-LABEL: // stdout.h
// CHECK-NEXT: #ifndef STDOUT_H
// CHECK-NEXT: #define STDOUT_H

// CHECK: #include <systemc>

// CHECK-LABEL: SC_MODULE(basic) {
// CHECK-NEXT: sc_core::sc_in<bool> port0;
// CHECK-NEXT: sc_core::sc_inout<sc_dt::sc_uint<64>> port1;
// CHECK-NEXT: sc_core::sc_out<sc_dt::sc_biguint<512>> port2;
// CHECK-NEXT: sc_core::sc_out<sc_dt::sc_bv<1024>> port3;
// CHECK-NEXT: };
systemc.module @basic (%port0: !systemc.in<i1>, %port1: !systemc.inout<i64>, %port2: !systemc.out<i512>, %port3: !systemc.out<i1024>) { }

// CHECK: #endif // STDOUT_H
