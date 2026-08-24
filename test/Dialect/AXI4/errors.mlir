// RUN: circt-opt %s --allow-unregistered-dialect --split-input-file --verify-diagnostics

// expected-error @below {{expected ','}}
"test.attrs"() {a = #axi4.burst_spec<fixed>} : () -> ()

// -----

// expected-error @below {{'fixed' burst 'len' must be between 1 and 16, got 0}}
"test.attrs"() {a = #axi4.burst_spec<fixed, len = 0>} : () -> ()

// -----

// expected-error @below {{'fixed' burst 'len' must be between 1 and 16, got 17}}
"test.attrs"() {a = #axi4.burst_spec<fixed, len = 17>} : () -> ()

// -----

// expected-error @below {{'incr' burst 'len' must be between 1 and 256, got 0}}
"test.attrs"() {a = #axi4.burst_spec<incr, len = 0>} : () -> ()

// -----

// expected-error @below {{'incr' burst 'len' must be between 1 and 256, got 257}}
"test.attrs"() {a = #axi4.burst_spec<incr, len = 257>} : () -> ()

// -----

// expected-error @below {{'wrap' burst 'len' must be 2, 4, 8, or 16, got 1}}
"test.attrs"() {a = #axi4.burst_spec<wrap, len = 1>} : () -> ()

// -----

// expected-error @below {{'wrap' burst 'len' must be 2, 4, 8, or 16, got 7}}
"test.attrs"() {a = #axi4.burst_spec<wrap, len = 7>} : () -> ()

// -----

// expected-error @below {{'wrap' burst 'len' must be 2, 4, 8, or 16, got 32}}
"test.attrs"() {a = #axi4.burst_spec<wrap, len = 32>} : () -> ()

// -----

// expected-error @below {{'burst_set' must be non-empty}}
"test.attrs"() {a = #axi4.burst_set<>} : () -> ()

// -----

// expected-error @below {{window 'last' address 0x3fff must not be less than 'base' address 0x4000}}
"test.attrs"() {a = #axi4.window<base = 0x4000, last = 0x3fff, burst_specs = <<fixed, len = 4>>>} : () -> ()

// -----

// expected-error @below {{'window_set' must be non-empty}}
"test.attrs"() {a = #axi4.window_set<>} : () -> ()

// -----

// expected-error @below {{port 'addr_width' must be at most 64, got 65}}
"test.port"() : () -> !axi4.port<addr_width = 65, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// -----

// expected-error @below {{port 'data_width' must be a power of two between 8 and 1024, got 24}}
"test.port"() : () -> !axi4.port<addr_width = 32, data_width = 24, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// -----

// expected-error @below {{port 'data_width' must be a power of two between 8 and 1024, got 4}}
"test.port"() : () -> !axi4.port<addr_width = 32, data_width = 4, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// -----

// expected-error @below {{port 'data_width' must be a power of two between 8 and 1024, got 2048}}
"test.port"() : () -> !axi4.port<addr_width = 32, data_width = 2048, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// -----

// expected-error @below {{port 'write_id_width' must be at most 32, got 33}}
"test.port"() : () -> !axi4.port<addr_width = 32, data_width = 64, write_id_width = 33, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// -----

// expected-error @below {{port 'read_id_width' must be at most 32, got 33}}
"test.port"() : () -> !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 33, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// -----

// expected-error @below {{port 'outstanding_writes' must be at most 4 for a 'write_id_width' of 2, got 5}}
"test.port"() : () -> !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 5, outstanding_reads = 4>

// -----

// expected-error @below {{port 'outstanding_reads' must be at most 4 for a 'read_id_width' of 2, got 5}}
"test.port"() : () -> !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 5>

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @Fanout(in %clk : !seq.clock, in %rst_ni : i1) {
  // expected-error @below {{'axi4.abstract_manager' op port result must have at most one use; route through an 'axi4.xbar' to fan out to multiple endpoints}}
  %mgr = axi4.abstract_manager %clk, %rst_ni : !port
  axi4.abstract_subordinate %clk, %rst_ni, %mgr : !port
  axi4.abstract_subordinate %clk, %rst_ni, %mgr : !port
}

// -----

hw.module @NotAPort(in %clk : !seq.clock, in %rst_ni : i1,
                    in %s : !hw.struct<a: i4>, in %v : i1) {
  // expected-error @below {{'port' must be an AXI4 port interface, but got 'i32'}}
  %port, %aw_ready, %w_ready, %b, %b_valid, %ar_ready, %r, %r_valid = axi4.channel_structs_to_port %clk, %rst_ni aw %s, %v w %s, %v b %v ar %s, %v r %v : i32
  hw.output
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!bad_aw = !hw.struct<id: i4, addr: i16, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4, user: i0>
!w = !hw.struct<data: i64, strb: i8, last: i1, user: i0>
!b = !hw.struct<id: i4, resp: i2, user: i0>
!r = !hw.struct<id: i4, data: i64, resp: i2, last: i1, user: i0>

hw.module @BadPayload(in %clk : !seq.clock, in %rst_ni : i1,
                      in %aw : !bad_aw, in %w : !w, in %v : i1) {
  // expected-error @below {{'axi4.channel_structs_to_port' op failed to verify that AW payload matches the port type}}
  %port, %aw_ready, %w_ready, %b, %b_valid, %ar_ready, %r, %r_valid = "axi4.channel_structs_to_port"(%clk, %rst_ni, %aw, %v, %w, %v, %v, %aw, %v, %v) : (!seq.clock, i1, !bad_aw, i1, !w, i1, i1, !bad_aw, i1, i1) -> (!port, i1, i1, !b, i1, i1, !r, i1)
  hw.output
}
