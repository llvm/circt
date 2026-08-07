// RUN: circt-opt %s --verify-axi4-networks --split-input-file --verify-diagnostics

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// A well-formed network produces no diagnostics
hw.module @Clean(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr = axi4.abstract_manager %clk, %rst_ni : !port
  axi4.abstract_subordinate %clk, %rst_ni, %mgr concurrent_writes 4 concurrent_reads 4 : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// expected-error @below {{AXI4 port must have at most one use; route through an 'axi4.xbar' to fan out to multiple endpoints}}
hw.module @BlockArgFanout(in %clk : !seq.clock, in %rst_ni : i1, in %port : !port) {
  axi4.abstract_subordinate %clk, %rst_ni, %port concurrent_writes 4 concurrent_reads 4 : !port
  axi4.abstract_subordinate %clk, %rst_ni, %port concurrent_writes 4 concurrent_reads 4 : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(in %clk : !seq.clock, in %rst_ni : i1, out axi : !port)

hw.module @InstanceFanout(in %clk : !seq.clock, in %rst_ni : i1) {
  // expected-error @below {{AXI4 port must have at most one use; route through an 'axi4.xbar' to fan out to multiple endpoints}}
  %axi = hw.instance "mgr" @Manager(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1) -> (axi: !port)
  axi4.abstract_subordinate %clk, %rst_ni, %axi concurrent_writes 4 concurrent_reads 4 : !port
  axi4.abstract_subordinate %clk, %rst_ni, %axi concurrent_writes 4 concurrent_reads 4 : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @Dangling(in %clk : !seq.clock, in %rst_ni : i1) {
  // expected-warning @below {{AXI4 port has no uses, so takes no part in a network}}
  %mgr = axi4.abstract_manager %clk, %rst_ni : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @ClockCrossing(in %clk : !seq.clock, in %other_clk : !seq.clock, in %rst_ni : i1) {
  // expected-note @below {{connected operation here}}
  %mgr = axi4.abstract_manager %clk, %rst_ni : !port
  // expected-error @below {{'axi4.abstract_subordinate' op is in a different clock domain to the 'axi4.abstract_manager' connected to it}}
  axi4.abstract_subordinate %other_clk, %rst_ni, %mgr concurrent_writes 4 concurrent_reads 4 : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @ResetCrossing(in %clk : !seq.clock, in %rst_ni : i1, in %other_rst_ni : i1) {
  // expected-note @below {{connected operation here}}
  %mgr = axi4.abstract_manager %clk, %rst_ni : !port
  // expected-error @below {{'axi4.abstract_subordinate' op is in a different reset domain to the 'axi4.abstract_manager' connected to it}}
  axi4.abstract_subordinate %clk, %other_rst_ni, %mgr concurrent_writes 4 concurrent_reads 4 : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @UndersizedSubordinate(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr = axi4.abstract_manager %clk, %rst_ni : !port
  // expected-warning @below {{endpoint can handle fewer writes than the port reaching it can have concurrently outstanding (2 < 4)}}
  axi4.abstract_subordinate %clk, %rst_ni, %mgr concurrent_writes 2 concurrent_reads 4 : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 3, user_width = 4, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!b = !hw.struct<id: i5, resp: i2, user: i4>
!r = !hw.struct<id: i3, data: i64, resp: i2, last: i1, user: i4>

hw.module @UndersizedBridge(in %clk : !seq.clock, in %rst_ni : i1,
                            in %port : !port, in %aw_ready : i1,
                            in %w_ready : i1, in %b : !b, in %b_valid : i1,
                            in %ar_ready : i1, in %r : !r, in %r_valid : i1) {
  // expected-warning @+2 {{endpoint can handle fewer reads than the port reaching it can have concurrently outstanding (1 < 4)}}
  %aw, %aw_valid, %w, %w_valid, %b_ready, %ar, %ar_valid, %r_ready =
    axi4.port_to_channel_structs %clk, %rst_ni, %port
      aw %aw_ready w %w_ready b %b, %b_valid
      ar %ar_ready r %r, %r_valid
      concurrent_writes 4 concurrent_reads 1 : !port
}
