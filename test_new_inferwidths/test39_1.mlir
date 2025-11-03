firrtl.circuit "MemScalar" {
  firrtl.module @MemScalar(out %out: !firrtl.uint, out %dbg: !firrtl.probe<vector<uint, 8>>) {
    // CHECK: firrtl.mem
    // CHECK-SAME: !firrtl.probe<vector<uint<7>, 8>>
    // CHECK-SAME: data flip: uint<7>
    // CHECK-SAME: data: uint<7>
    // CHECK-SAME: data: uint<7>
    %m_dbg, %m_p0, %m_p1, %m_p2 = firrtl.mem Undefined {
      depth = 8 : i64,
      name = "m",
      portNames = ["dbg", "p0", "p1", "p2"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32} :
      !firrtl.probe<vector<uint, 8>>,
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint>,
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint, mask: uint<1>>,
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, rdata flip: uint, wmode: uint<1>, wdata: uint, wmask: uint<1>>
    %m_p0_data = firrtl.subfield %m_p0[data] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint>
    %m_p1_data = firrtl.subfield %m_p1[data] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint, mask: uint<1>>
    %m_p2_wdata = firrtl.subfield %m_p2[wdata] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, rdata flip: uint, wmode: uint<1>, wdata: uint, wmask: uint<1>>
    %c0_ui5 = firrtl.constant 0 : !firrtl.uint<5>
    %c0_ui7 = firrtl.constant 0 : !firrtl.uint<7>
    firrtl.connect %m_p1_data, %c0_ui5 : !firrtl.uint, !firrtl.uint<5>
    firrtl.connect %m_p2_wdata, %c0_ui7 : !firrtl.uint, !firrtl.uint<7>
    firrtl.connect %out, %m_p0_data : !firrtl.uint, !firrtl.uint
    firrtl.ref.define %dbg, %m_dbg : !firrtl.probe<vector<uint, 8>>
    // CHECK:  firrtl.ref.define %dbg, %m_dbg : !firrtl.probe<vector<uint<7>, 8>>
  }
}