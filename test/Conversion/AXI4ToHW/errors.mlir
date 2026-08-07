// RUN: circt-opt %s --lower-axi4-to-hw --split-input-file --verify-diagnostics

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// Abstract endpoints model an endpoint with no RTL behind it, so there is
// nothing to lower them to
hw.module @AbstractNetwork(in %clk : !seq.clock, in %rst_ni : i1) {
  // expected-error @below {{'axi4.abstract_manager' op models an endpoint with no RTL, so cannot be lowered}}
  %mgr = axi4.abstract_manager %clk, %rst_ni : !port
  // expected-error @below {{'axi4.abstract_subordinate' op models an endpoint with no RTL, so cannot be lowered}}
  axi4.abstract_subordinate %clk, %rst_ni, %mgr concurrent_writes 4 concurrent_reads 4 : !port
}

// -----

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(out axi : !mgr)
hw.module.extern @Subordinate(in %axi : !sub)

hw.module @Crossbar(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !mgr)
  // expected-error @below {{'axi4.xbar' op lowering is not yet implemented}}
  %s = axi4.xbar %clk, %rst_ni mgrs %m : (!mgr) -> (!sub)
  hw.instance "sub" @Subordinate(axi: %s: !sub) -> ()
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(out axi : !port)

// A port with no consumer has no signals at the far end to wire to
hw.module @DanglingResult() {
  // expected-error @below {{AXI4 port has no uses, so cannot be lowered}}
  %p = hw.instance "mgr" @Manager() -> (axi: !port)
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// The same for a port arriving as a module argument
// expected-error @below {{AXI4 port has no uses, so cannot be lowered}}
hw.module @DanglingArgument(in %p : !port) {
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// Make sure we're vocal about modifying module signatures when we can't
// change the specification too
// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Manager(out axi : !port)

// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Subordinate(in %axi : !port)

// A module with several ports is warned about once
// expected-warning @below {{lowering AXI4 ports 'axi_sub', 'axi_mgr' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Bridge(in %axi_sub : !port, out axi_mgr : !port)

hw.module @Externs() {
  %p = hw.instance "mgr" @Manager() -> (axi: !port)
  %q = hw.instance "bridge" @Bridge(axi_sub: %p: !port) -> (axi_mgr: !port)
  hw.instance "sub" @Subordinate(axi: %q: !port) -> ()
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// Make sure we complain about being asked to lower an implicit clock domain
// crossing
hw.module @ClockCrossing(in %clk_a : !seq.clock, in %clk_b : !seq.clock, in %rst_ni : i1) {
  // expected-note @below {{connected operation here}}
  %port, %aw_ready, %w_ready, %b, %b_valid, %ar_ready, %r, %r_valid = axi4.channel_structs_to_port %clk_a, %rst_ni
      aw %aw, %aw_valid w %w, %w_valid b %b_ready
      ar %ar, %ar_valid r %r_ready : !port
  // expected-error @below {{'axi4.port_to_channel_structs' op is in a different clock domain to the 'axi4.channel_structs_to_port' connected to it}}
  %aw, %aw_valid, %w, %w_valid, %b_ready, %ar, %ar_valid, %r_ready = axi4.port_to_channel_structs %clk_b, %rst_ni, %port
      aw %aw_ready w %w_ready b %b, %b_valid
      ar %ar_ready r %r, %r_valid concurrent_writes 4 concurrent_reads 4 : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// Make sure we complain about being asked to lower an implicit reset domain
// crossing
hw.module @ResetCrossing(in %clk : !seq.clock, in %rst_a : i1, in %rst_b : i1) {
  // expected-note @below {{connected operation here}}
  %port, %aw_ready, %w_ready, %b, %b_valid, %ar_ready, %r, %r_valid = axi4.channel_structs_to_port %clk, %rst_a
      aw %aw, %aw_valid w %w, %w_valid b %b_ready
      ar %ar, %ar_valid r %r_ready : !port
  // expected-error @below {{'axi4.port_to_channel_structs' op is in a different reset domain to the 'axi4.channel_structs_to_port' connected to it}}
  %aw, %aw_valid, %w, %w_valid, %b_ready, %ar, %ar_valid, %r_ready = axi4.port_to_channel_structs %clk, %rst_b, %port
      aw %aw_ready w %w_ready b %b, %b_valid
      ar %ar_ready r %r, %r_valid concurrent_writes 4 concurrent_reads 4 : !port
}
