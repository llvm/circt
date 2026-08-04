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

// CHECK: #axi4.burst_set<<fixed, len = 4>>
"test.attrs"() {a = #axi4.burst_set<<fixed, len = 4>>} : () -> ()

// Check burst_sets are correctly canonicalized after parsing
// CHECK: #axi4.burst_set<<fixed, len = 4>, <incr, len = 8>>
"test.attrs"() {a = #axi4.burst_set<<incr, len = 8>, <fixed, len = 4>>} : () -> ()
// CHECK: #axi4.burst_set<<incr, len = 8>>
"test.attrs"() {a = #axi4.burst_set<<incr, len = 8>, <incr, len = 8>>} : () -> ()
// CHECK: #axi4.burst_set<<fixed, len = 16>, <incr, len = 1>, <incr, len = 256>, <wrap, len = 2>>
"test.attrs"() {a = #axi4.burst_set<<wrap, len = 2>, <incr, len = 256>, <incr, len = 1>, <fixed, len = 16>>} : () -> ()

// CHECK: #axi4.window<base = 0x4000, last = 0x40ff, burst_specs = <<fixed, len = 4>>>
"test.attrs"() {a = #axi4.window<base = 0x4000, last = 0x40ff, burst_specs = <<fixed, len = 4>>>} : () -> ()

// Check a window may cover the whole address space
// CHECK: #axi4.window<base = 0x0, last = 0xffffffffffffffff, burst_specs = <<fixed, len = 4>>>
"test.attrs"() {a = #axi4.window<base = 0x0, last = 0xffffffffffffffff, burst_specs = <<fixed, len = 4>>>} : () -> ()
