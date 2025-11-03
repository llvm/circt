firrtl.circuit "MemBundle" {
  // CHECK-SAME: out %out: !firrtl.bundle<a: uint<7>>
  firrtl.module @MemBundle(out %out: !firrtl.bundle<a: uint>) {
    // CHECK: firrtl.mem
    // CHECK-SAME: data flip: bundle<a: uint<7>>
    // CHECK-SAME: data: bundle<a: uint<7>>
    // CHECK-SAME: data: bundle<a: uint<7>>
    %m_p0, %m_p1, %m_p2 = firrtl.mem Undefined {
      depth = 8 : i64,
      name = "m",
      portNames = ["p0", "p1", "p2"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32} :
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: bundle<a: uint>>,
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<a: uint>, mask: bundle<a: uint<1>>>,
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint>, wmode: uint<1>, wdata: bundle<a: uint>, wmask: bundle<a: uint<1>>>
    %m_p0_data = firrtl.subfield %m_p0[data] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: bundle<a: uint>>
    %m_p1_data = firrtl.subfield %m_p1[data] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<a: uint>, mask: bundle<a: uint<1>>>
    %m_p2_wdata = firrtl.subfield %m_p2[wdata] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint>, wmode: uint<1>, wdata: bundle<a: uint>, wmask: bundle<a: uint<1>>>
    %m_p1_data_a = firrtl.subfield %m_p1_data[a] : !firrtl.bundle<a: uint>
    %m_p2_wdata_a = firrtl.subfield %m_p2_wdata[a] : !firrtl.bundle<a: uint>
    %c0_ui5 = firrtl.constant 0 : !firrtl.uint<5>
    %c0_ui7 = firrtl.constant 0 : !firrtl.uint<7>
    firrtl.connect %m_p1_data_a, %c0_ui5 : !firrtl.uint, !firrtl.uint<5>
    firrtl.connect %m_p2_wdata_a, %c0_ui7 : !firrtl.uint, !firrtl.uint<7>
    firrtl.connect %out, %m_p0_data : !firrtl.bundle<a: uint>, !firrtl.bundle<a: uint>
  }
}