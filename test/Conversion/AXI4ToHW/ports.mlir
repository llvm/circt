// RUN: circt-opt %s --lower-axi4-to-hw | FileCheck %s --implicit-check-not=axi4. --implicit-check-not=seq.const_clock

// The implicit check-nots assert that no AXI4 op and no filler clock survive
// anywhere in the output.

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 3, user_width = 4, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!narrow = !axi4.port<addr_width = 16, data_width = 8, write_id_width = 1, read_id_width = 1, user_width = 0, windows = <<base = 0x0, last = 0xff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 2, outstanding_reads = 2>

!aw = !hw.struct<id: i5, addr: i32, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4, user: i4>
!w = !hw.struct<data: i64, strb: i8, last: i1, user: i4>
!ar = !hw.struct<id: i3, addr: i32, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4, user: i4>
!b = !hw.struct<id: i5, resp: i2, user: i4>
!r = !hw.struct<id: i3, data: i64, resp: i2, last: i1, user: i4>

// An output port makes the module a manager, so it drives the request payloads
// and valids and receives their readys
// CHECK-LABEL: hw.module.extern @ExternManager(
// CHECK-SAME:    in %clk : !seq.clock, in %rst_ni : i1,
// CHECK-SAME:    in %axi_awready : i1,
// CHECK-SAME:    in %axi_wready : i1,
// CHECK-SAME:    in %axi_b : !hw.struct<id: i5, resp: i2, user: i4>, in %axi_bvalid : i1,
// CHECK-SAME:    in %axi_arready : i1,
// CHECK-SAME:    in %axi_r : !hw.struct<id: i3, data: i64, resp: i2, last: i1, user: i4>, in %axi_rvalid : i1,
// CHECK-SAME:    out axi_aw : !hw.struct<id: i5, addr: i32, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4, user: i4>, out axi_awvalid : i1,
// CHECK-SAME:    out axi_w : !hw.struct<data: i64, strb: i8, last: i1, user: i4>, out axi_wvalid : i1,
// CHECK-SAME:    out axi_bready : i1,
// CHECK-SAME:    out axi_ar : !hw.struct<id: i3, addr: i32, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4, user: i4>, out axi_arvalid : i1,
// CHECK-SAME:    out axi_rready : i1)
hw.module.extern @ExternManager(in %clk : !seq.clock, in %rst_ni : i1, out axi : !port)

// An input port is the exact mirror: every signal takes the other direction.
// CHECK-LABEL: hw.module.extern @ExternSubordinate(
// CHECK-SAME:    in %clk : !seq.clock, in %rst_ni : i1,
// CHECK-SAME:    in %axi_aw : !hw.struct<id: i5, addr: i32,
// CHECK-SAME:    in %axi_awvalid : i1,
// CHECK-SAME:    in %axi_w : !hw.struct<data: i64, strb: i8, last: i1, user: i4>, in %axi_wvalid : i1,
// CHECK-SAME:    in %axi_bready : i1,
// CHECK-SAME:    in %axi_ar : !hw.struct<id: i3, addr: i32,
// CHECK-SAME:    in %axi_arvalid : i1,
// CHECK-SAME:    in %axi_rready : i1,
// CHECK-SAME:    out axi_awready : i1, out axi_wready : i1,
// CHECK-SAME:    out axi_b : !hw.struct<id: i5, resp: i2, user: i4>, out axi_bvalid : i1,
// CHECK-SAME:    out axi_arready : i1,
// CHECK-SAME:    out axi_r : !hw.struct<id: i3, data: i64, resp: i2, last: i1, user: i4>, out axi_rvalid : i1)
hw.module.extern @ExternSubordinate(in %clk : !seq.clock, in %rst_ni : i1, in %axi : !port)

// Make sure orders are preserved when multiple ports (in both directions)
// are present
// Also confirm that a zero user width still yields a `user` field, and that
// `strb` follows the data width.
// CHECK-LABEL: hw.module.extern @MultiPort(
// CHECK-SAME:    in %up_aw : !hw.struct<id: i5, addr: i32,
// CHECK-SAME:    in %up_rready : i1,
// CHECK-SAME:    in %narrow_up_aw : !hw.struct<id: i1, addr: i16, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4, user: i0>, in %narrow_up_awvalid : i1,
// CHECK-SAME:    in %narrow_up_w : !hw.struct<data: i8, strb: i1, last: i1, user: i0>, in %narrow_up_wvalid : i1,
// CHECK-SAME:    in %narrow_up_bready : i1,
// CHECK-SAME:    in %narrow_up_ar : !hw.struct<id: i1, addr: i16,
// CHECK-SAME:    in %narrow_up_rready : i1,
// CHECK-SAME:    in %down_awready : i1,
// CHECK-SAME:    in %narrow_down_awready : i1,
// CHECK-SAME:    in %narrow_down_r : !hw.struct<id: i1, data: i8, resp: i2, last: i1, user: i0>, in %narrow_down_rvalid : i1,
// CHECK-SAME:    out up_awready : i1,
// CHECK-SAME:    out narrow_up_b : !hw.struct<id: i1, resp: i2, user: i0>, out narrow_up_bvalid : i1,
// CHECK-SAME:    out down_aw : !hw.struct<id: i5, addr: i32,
// CHECK-SAME:    out narrow_down_rready : i1)
hw.module.extern @MultiPort(in %up : !port, in %narrow_up : !narrow, out down : !port, out narrow_down : !narrow)

// Module currently contains only a conversion op from structs to an
// !axi4.port - in lowering it should just become a passthrough
// CHECK-LABEL: hw.module @StructsToPort(
// CHECK-SAME:    out axi_aw : !hw.struct<id: i5, addr: i32,
// CHECK:         hw.output %aw, %aw_valid, %w, %w_valid, %b_ready, %ar, %ar_valid, %r_ready,
// CHECK-SAME:              %axi_awready, %axi_wready, %axi_b, %axi_bvalid, %axi_arready, %axi_r, %axi_rvalid :
hw.module @StructsToPort(in %clk : !seq.clock, in %rst_ni : i1,
                         in %aw : !aw, in %aw_valid : i1,
                         in %w : !w, in %w_valid : i1, in %b_ready : i1,
                         in %ar : !ar, in %ar_valid : i1, in %r_ready : i1,
                         out axi : !port, out aw_ready : i1, out w_ready : i1,
                         out b : !b, out b_valid : i1, out ar_ready : i1,
                         out r : !r, out r_valid : i1) {
  %p, %awr, %wr, %bb, %bv, %arr, %rb, %rv = axi4.channel_structs_to_port %clk, %rst_ni
    aw %aw, %aw_valid w %w, %w_valid b %b_ready ar %ar, %ar_valid r %r_ready : !port
  hw.output %p, %awr, %wr, %bb, %bv, %arr, %rb, %rv : !port, i1, i1, !b, i1, i1, !r, i1
}

// Check the same as above, but for conversion from !axi4.port to structs
// CHECK-LABEL: hw.module @PortToStructs(
// CHECK:         hw.output %aw_ready, %w_ready, %b, %b_valid, %ar_ready, %r, %r_valid,
// CHECK-SAME:              %axi_aw, %axi_awvalid, %axi_w, %axi_wvalid, %axi_bready, %axi_ar, %axi_arvalid, %axi_rready :
hw.module @PortToStructs(in %clk : !seq.clock, in %rst_ni : i1, in %axi : !port,
                         in %aw_ready : i1, in %w_ready : i1,
                         in %b : !b, in %b_valid : i1, in %ar_ready : i1,
                         in %r : !r, in %r_valid : i1,
                         out aw : !aw, out aw_valid : i1,
                         out w : !w, out w_valid : i1, out b_ready : i1,
                         out ar : !ar, out ar_valid : i1, out r_ready : i1) {
  %aw2, %awv, %w2, %wv, %br, %ar2, %arv, %rr = axi4.port_to_channel_structs %clk, %rst_ni, %axi
    aw %aw_ready w %w_ready b %b, %b_valid ar %ar_ready r %r, %r_valid concurrent_writes 4 concurrent_reads 4 : !port
  hw.output %aw2, %awv, %w2, %wv, %br, %ar2, %arv, %rr : !aw, i1, !w, i1, i1, !ar, i1, i1
}

// Check that a module that passes through an !axi4.port becomes a passthrough
// for all signals
// CHECK-LABEL: hw.module @Passthrough(
// CHECK:         hw.output %q_awready, %q_wready, %q_b, %q_bvalid, %q_arready, %q_r, %q_rvalid,
// CHECK-SAME:              %p_aw, %p_awvalid, %p_w, %p_wvalid, %p_bready, %p_ar, %p_arvalid, %p_rready :
hw.module @Passthrough(in %p : !port, out q : !port) {
  hw.output %p : !port
}

// Two instances joined by a port, with no AXI4 op in the parent at all. Each
// instance's forward signals feed the other's, and the readys and responses
// come back the other way.
// CHECK-LABEL: hw.module @PointToPoint(
// CHECK:         %mgr.axi_aw, %mgr.axi_awvalid, %mgr.axi_w, %mgr.axi_wvalid, %mgr.axi_bready, %mgr.axi_ar, %mgr.axi_arvalid, %mgr.axi_rready = hw.instance "mgr" @ExternManager(
// CHECK-SAME:      axi_awready: %sub.axi_awready: i1
// CHECK-SAME:      axi_b: %sub.axi_b: !hw.struct<id: i5, resp: i2, user: i4>
// CHECK-SAME:      axi_rvalid: %sub.axi_rvalid: i1
// CHECK:         %sub.axi_awready, %sub.axi_wready, %sub.axi_b, %sub.axi_bvalid, %sub.axi_arready, %sub.axi_r, %sub.axi_rvalid = hw.instance "sub" @ExternSubordinate(
// CHECK-SAME:      axi_aw: %mgr.axi_aw: !hw.struct<id: i5, addr: i32,
// CHECK-SAME:      axi_rready: %mgr.axi_rready: i1
hw.module @PointToPoint(in %clk : !seq.clock, in %rst_ni : i1) {
  %p = hw.instance "mgr" @ExternManager(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1) -> (axi: !port)
  hw.instance "sub" @ExternSubordinate(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1, axi: %p: !port) -> ()
}

// A port crossing two levels of hierarchy, through the passthrough above.
// CHECK-LABEL: hw.module @Hierarchy(
// CHECK:         hw.instance "mgr" @ExternManager(
// CHECK-SAME:      axi_awready: %mid.p_awready: i1
// CHECK:         hw.instance "mid" @Passthrough(
// CHECK-SAME:      p_aw: %mgr.axi_aw:
// CHECK-SAME:      q_awready: %sub.axi_awready: i1
// CHECK:         hw.instance "sub" @ExternSubordinate(
// CHECK-SAME:      axi_aw: %mid.q_aw:
hw.module @Hierarchy(in %clk : !seq.clock, in %rst_ni : i1) {
  %p = hw.instance "mgr" @ExternManager(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1) -> (axi: !port)
  %q = hw.instance "mid" @Passthrough(p: %p: !port) -> (q: !port)
  hw.instance "sub" @ExternSubordinate(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1, axi: %q: !port) -> ()
}
