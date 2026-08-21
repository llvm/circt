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

// CHECK: #axi4.window_set<<base = 0x0, last = 0xff, burst_specs = <<fixed, len = 4>>>>
"test.attrs"() {a = #axi4.window_set<<base = 0x0, last = 0xff, burst_specs = <<fixed, len = 4>>>>} : () -> ()

// Check window_sets are correctly normalized after parsing
// CHECK: #axi4.window_set<<base = 0x0, last = 0xff, burst_specs = <<fixed, len = 4>>>, <base = 0x1000, last = 0x10ff, burst_specs = <<incr, len = 8>>>>
"test.attrs"() {a = #axi4.window_set<<base = 0x1000, last = 0x10ff, burst_specs = <<incr, len = 8>>>, <base = 0x0, last = 0xff, burst_specs = <<fixed, len = 4>>>>} : () -> ()
// CHECK: #axi4.window_set<<base = 0x0, last = 0xff, burst_specs = <<fixed, len = 4>>>>
"test.attrs"() {a = #axi4.window_set<<base = 0x0, last = 0xff, burst_specs = <<fixed, len = 4>>>, <base = 0x0, last = 0xff, burst_specs = <<fixed, len = 4>>>>} : () -> ()

// Check overlapping windows are split into disjoint windows unioning their
// capabilities
// CHECK: #axi4.window_set<<base = 0x0, last = 0xff, burst_specs = <<fixed, len = 4>, <incr, len = 8>>>, <base = 0x100, last = 0xfff, burst_specs = <<fixed, len = 4>>>>
"test.attrs"() {a = #axi4.window_set<<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>, <base = 0x0, last = 0xff, burst_specs = <<incr, len = 8>>>>} : () -> ()

// Check contiguous windows are merged only if their capabilities match
// CHECK: #axi4.window_set<<base = 0x0, last = 0x1ff, burst_specs = <<fixed, len = 4>>>>
"test.attrs"() {a = #axi4.window_set<<base = 0x0, last = 0xff, burst_specs = <<fixed, len = 4>>>, <base = 0x100, last = 0x1ff, burst_specs = <<fixed, len = 4>>>>} : () -> ()
// CHECK: #axi4.window_set<<base = 0x0, last = 0xff, burst_specs = <<fixed, len = 4>>>, <base = 0x100, last = 0x1ff, burst_specs = <<incr, len = 8>>>>
"test.attrs"() {a = #axi4.window_set<<base = 0x0, last = 0xff, burst_specs = <<fixed, len = 4>>>, <base = 0x100, last = 0x1ff, burst_specs = <<incr, len = 8>>>>} : () -> ()

// Check a window at the top of the address space is normalized correctly
// CHECK: #axi4.window_set<<base = 0xfffffffffffff000, last = 0xffffffffffffffff, burst_specs = <<fixed, len = 4>>>>
"test.attrs"() {a = #axi4.window_set<<base = 0xfffffffffffff000, last = 0xffffffffffffffff, burst_specs = <<fixed, len = 4>>>>} : () -> ()

// Check windows covering the whole address space are merged
// CHECK: #axi4.window_set<<base = 0x0, last = 0xffffffffffffffff, burst_specs = <<fixed, len = 4>>>>
"test.attrs"() {a = #axi4.window_set<<base = 0x0, last = 0x7fffffffffffffff, burst_specs = <<fixed, len = 4>>>, <base = 0x8000000000000000, last = 0xffffffffffffffff, burst_specs = <<fixed, len = 4>>>>} : () -> ()

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// CHECK: !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
"test.port"() : () -> !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// Check with widest widths (and make sure ID width check doesn't overflow)
// CHECK: !axi4.port<addr_width = 64, data_width = 1024, write_id_width = 32, read_id_width = 32, user_width = 8, windows = <<base = 0x0, last = 0xffffffffffffffff, burst_specs = <<incr, len = 256>>>>, outstanding_writes = 4294967295, outstanding_reads = 4294967295>
"test.port"() : () -> !axi4.port<addr_width = 64, data_width = 1024, write_id_width = 32, read_id_width = 32, user_width = 8, windows = <<base = 0x0, last = 0xffffffffffffffff, burst_specs = <<incr, len = 256>>>>, outstanding_writes = 4294967295, outstanding_reads = 4294967295>

// Check fields may be given in any order, and are printed in a canonical one
// CHECK: !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
"test.port"() : () -> !axi4.port<outstanding_reads = 4, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, data_width = 64, user_width = 0, addr_width = 32, outstanding_writes = 4, read_id_width = 4, write_id_width = 4>
