// RUN: circt-opt -pass-pipeline="builtin.module(firrtl.circuit(firrtl.module(firrtl-group-sink)))" %s | FileCheck %s

// Test that simple things are sunk:
//   - nodes
//   - constants
//   - primitive operations
//
// CHECK-LABEL: firrtl.circuit "SimpleSink"
firrtl.circuit "SimpleSink" {
 firrtl.declgroup @A bind {
 }
 // CHECK: firrtl.module @SimpleSink
 firrtl.module @SimpleSink(in %a: !firrtl.uint<1>) {
   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
   %node = firrtl.node %a : !firrtl.uint<1>
   %0 = firrtl.not %a : (!firrtl.uint<1>) -> !firrtl.uint<1>
   // CHECK-NEXT: firrtl.group @A
   firrtl.group @A {
     // CHECK: %c0_ui1 = firrtl.constant
     %constant_group = firrtl.node %c0_ui1 : !firrtl.uint<1>
     // CHECK: %node = firrtl.node
     %node_group = firrtl.node %node : !firrtl.uint<1>
     // CHECK: %0 = firrtl.not
     %primop_group = firrtl.node %0 : !firrtl.uint<1>
   }
 }
}
