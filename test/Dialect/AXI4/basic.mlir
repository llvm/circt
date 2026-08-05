// RUN: circt-opt %s --allow-unregistered-dialect | circt-opt --allow-unregistered-dialect | FileCheck %s

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

// CHECK: #axi4.burst_spec<fixed, len = 1>
"test.attrs"() {a = #axi4.burst_spec<fixed, len = 1>} : () -> ()
// CHECK: #axi4.burst_spec<fixed, len = 16>
"test.attrs"() {a = #axi4.burst_spec<fixed, len = 16>} : () -> ()
// CHECK: #axi4.burst_spec<incr, len = 1>
"test.attrs"() {a = #axi4.burst_spec<incr, len = 1>} : () -> ()
// CHECK: #axi4.burst_spec<incr, len = 256>
"test.attrs"() {a = #axi4.burst_spec<incr, len = 256>} : () -> ()
// CHECK: #axi4.burst_spec<wrap, len = 2>
"test.attrs"() {a = #axi4.burst_spec<wrap, len = 2>} : () -> ()
// CHECK: #axi4.burst_spec<wrap, len = 16>
"test.attrs"() {a = #axi4.burst_spec<wrap, len = 16>} : () -> ()
