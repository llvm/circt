// RUN: circt-opt --lower-firrtl-to-hw --verify-diagnostics %s | FileCheck %s --check-prefix CHECK

firrtl.circuit "Foo" {
  // CHECK-LABEL: hw.module @Foo
  firrtl.module @Foo(
    in %clk: !firrtl.clock,
    in %en: !firrtl.uint<1>,
    in %addr: !firrtl.uint<4>,
    in %wdata: !firrtl.uint<42>,
    in %wmode: !firrtl.uint<1>,
    in %mask2: !firrtl.uint<2>,
    in %mask3: !firrtl.uint<3>,
    in %mask6: !firrtl.uint<6>
  ) {
    // CHECK-NEXT: %mem1 = seq.firmem 0, 1, undefined, port_order : <12 x 42>
    // CHECK-NEXT: [[RDATA:%.+]] = seq.firmem.read_port %mem1[%addr], clock %clk enable %en :
    // CHECK-NEXT: hw.wire [[RDATA]] sym @mem1_data
    %mem1_r = firrtl.mem Undefined {depth = 12 : i64, name = "mem1", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem1_r.clk = firrtl.subfield %mem1_r[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem1_r.en = firrtl.subfield %mem1_r[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem1_r.addr = firrtl.subfield %mem1_r[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem1_r.data = firrtl.subfield %mem1_r[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.matchingconnect %mem1_r.clk, %clk : !firrtl.clock
    firrtl.matchingconnect %mem1_r.en, %en : !firrtl.uint<1>
    firrtl.matchingconnect %mem1_r.addr, %addr : !firrtl.uint<4>
    %mem1_data = firrtl.node sym @mem1_data %mem1_r.data : !firrtl.uint<42>

    // CHECK-NEXT: %mem2 = seq.firmem 1, 2, old, port_order : <13 x 42, mask 2>
    // CHECK-NEXT: seq.firmem.write_port %mem2[%addr] = %wdata, clock %clk enable %en mask %mask2 :
    %mem2_w = firrtl.mem Old {depth = 13 : i64, name = "mem2", portNames = ["w"], readLatency = 1 : i32, writeLatency = 2 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<2>>
    %mem2_w.clk = firrtl.subfield %mem2_w[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<2>>
    %mem2_w.en = firrtl.subfield %mem2_w[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<2>>
    %mem2_w.addr = firrtl.subfield %mem2_w[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<2>>
    %mem2_w.data = firrtl.subfield %mem2_w[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<2>>
    %mem2_w.mask = firrtl.subfield %mem2_w[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<2>>
    firrtl.matchingconnect %mem2_w.clk, %clk : !firrtl.clock
    firrtl.matchingconnect %mem2_w.en, %en : !firrtl.uint<1>
    firrtl.matchingconnect %mem2_w.addr, %addr : !firrtl.uint<4>
    firrtl.matchingconnect %mem2_w.data, %wdata : !firrtl.uint<42>
    firrtl.matchingconnect %mem2_w.mask, %mask2 : !firrtl.uint<2>

    // CHECK-NEXT: %mem3 = seq.firmem 3, 2, new, port_order : <14 x 42, mask 3>
    // CHECK-NEXT: [[RDATA:%.+]] = seq.firmem.read_write_port %mem3[%addr] = %wdata if %wmode, clock %clk enable %en mask %mask3 :
    // CHECK-NEXT: hw.wire [[RDATA]] sym @mem3_data
    %mem3_rw = firrtl.mem New {depth = 14 : i64, name = "mem3", portNames = ["rw"], readLatency = 3 : i32, writeLatency = 2 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<3>>
    %mem3_rw.clk = firrtl.subfield %mem3_rw[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<3>>
    %mem3_rw.en = firrtl.subfield %mem3_rw[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<3>>
    %mem3_rw.addr = firrtl.subfield %mem3_rw[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<3>>
    %mem3_rw.wdata = firrtl.subfield %mem3_rw[wdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<3>>
    %mem3_rw.rdata = firrtl.subfield %mem3_rw[rdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<3>>
    %mem3_rw.wmask = firrtl.subfield %mem3_rw[wmask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<3>>
    %mem3_rw.wmode = firrtl.subfield %mem3_rw[wmode] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<3>>
    firrtl.matchingconnect %mem3_rw.clk, %clk : !firrtl.clock
    firrtl.matchingconnect %mem3_rw.en, %en : !firrtl.uint<1>
    firrtl.matchingconnect %mem3_rw.addr, %addr : !firrtl.uint<4>
    firrtl.matchingconnect %mem3_rw.wdata, %wdata : !firrtl.uint<42>
    firrtl.matchingconnect %mem3_rw.wmask, %mask3 : !firrtl.uint<3>
    firrtl.matchingconnect %mem3_rw.wmode, %wmode : !firrtl.uint<1>
    %mem3_data = firrtl.node sym @mem3_data %mem3_rw.rdata : !firrtl.uint<42>

    // CHECK-NEXT: %mem4 = seq.firmem 4, 5, undefined, port_order : <15 x 42, mask 6>
    // CHECK-NEXT: [[RDATA1:%.+]] = seq.firmem.read_port %mem4[%addr], clock %clk enable %en :
    // CHECK-NEXT: seq.firmem.write_port %mem4[%addr] = %wdata, clock %clk enable %en mask %mask6 :
    // CHECK-NEXT: [[RDATA2:%.+]] = seq.firmem.read_write_port %mem4[%addr] = %wdata if %wmode, clock %clk enable %en mask %mask6 :
    // CHECK-NEXT: hw.wire [[RDATA1]] sym @mem4_data0
    // CHECK-NEXT: hw.wire [[RDATA2]] sym @mem4_data1
    %mem4_r, %mem4_w, %mem4_rw = firrtl.mem Undefined {depth = 15 : i64, name = "mem4", portNames = ["r", "w", "rw"], readLatency = 4 : i32, writeLatency = 5 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<6>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<6>>
    %mem4_r.clk = firrtl.subfield %mem4_r[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem4_w.clk = firrtl.subfield %mem4_w[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<6>>
    %mem4_rw.clk = firrtl.subfield %mem4_rw[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<6>>
    %mem4_r.en = firrtl.subfield %mem4_r[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem4_w.en = firrtl.subfield %mem4_w[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<6>>
    %mem4_rw.en = firrtl.subfield %mem4_rw[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<6>>
    %mem4_r.addr = firrtl.subfield %mem4_r[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem4_w.addr = firrtl.subfield %mem4_w[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<6>>
    %mem4_rw.addr = firrtl.subfield %mem4_rw[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<6>>
    %mem4_w.data = firrtl.subfield %mem4_w[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<6>>
    %mem4_rw.wdata = firrtl.subfield %mem4_rw[wdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<6>>
    %mem4_r.data = firrtl.subfield %mem4_r[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem4_rw.rdata = firrtl.subfield %mem4_rw[rdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<6>>
    %mem4_w.mask = firrtl.subfield %mem4_w[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<6>>
    %mem4_rw.wmask = firrtl.subfield %mem4_rw[wmask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<6>>
    %mem4_rw.wmode = firrtl.subfield %mem4_rw[wmode] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<6>>
    firrtl.matchingconnect %mem4_r.clk, %clk : !firrtl.clock
    firrtl.matchingconnect %mem4_w.clk, %clk : !firrtl.clock
    firrtl.matchingconnect %mem4_rw.clk, %clk : !firrtl.clock
    firrtl.matchingconnect %mem4_r.en, %en : !firrtl.uint<1>
    firrtl.matchingconnect %mem4_w.en, %en : !firrtl.uint<1>
    firrtl.matchingconnect %mem4_rw.en, %en : !firrtl.uint<1>
    firrtl.matchingconnect %mem4_r.addr, %addr : !firrtl.uint<4>
    firrtl.matchingconnect %mem4_w.addr, %addr : !firrtl.uint<4>
    firrtl.matchingconnect %mem4_rw.addr, %addr : !firrtl.uint<4>
    firrtl.matchingconnect %mem4_w.data, %wdata : !firrtl.uint<42>
    firrtl.matchingconnect %mem4_rw.wdata, %wdata : !firrtl.uint<42>
    firrtl.matchingconnect %mem4_w.mask, %mask6 : !firrtl.uint<6>
    firrtl.matchingconnect %mem4_rw.wmask, %mask6 : !firrtl.uint<6>
    firrtl.matchingconnect %mem4_rw.wmode, %wmode : !firrtl.uint<1>
    %mem4_data0 = firrtl.node sym @mem4_data0 %mem4_r.data : !firrtl.uint<42>
    %mem4_data1 = firrtl.node sym @mem4_data1 %mem4_rw.rdata : !firrtl.uint<42>
  }

  // CHECK-LABEL: hw.module @ZeroDataWidth
  firrtl.module @ZeroDataWidth(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>, in %addr: !firrtl.uint<4>, in %data: !firrtl.uint<0>, in %mask: !firrtl.uint<1>) {
    // CHECK: [[CONST_X:%.+]] = sv.constantX : i1
    // CHECK: %mem = seq.firmem 0, 1, undefined, port_order : <12 x 0>
    // CHECK: seq.firmem.write_port %mem[{{%.+}}] = [[CONST_X]], clock {{.+}} enable {{%.+}} : <12 x 0>
    %mem_w = firrtl.mem Undefined {depth = 12 : i64, name = "mem", portNames = ["w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>

    %mem_w.clk = firrtl.subfield %mem_w[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>
    %mem_w.en = firrtl.subfield %mem_w[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>
    %mem_w.addr = firrtl.subfield %mem_w[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>
    %mem_w.data = firrtl.subfield %mem_w[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>
    %mem_w.mask = firrtl.subfield %mem_w[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>
    firrtl.matchingconnect %mem_w.clk, %clk : !firrtl.clock
    firrtl.matchingconnect %mem_w.en, %en : !firrtl.uint<1>
    firrtl.matchingconnect %mem_w.addr, %addr : !firrtl.uint<4>
    firrtl.matchingconnect %mem_w.data, %data : !firrtl.uint<0>
    firrtl.matchingconnect %mem_w.mask, %mask : !firrtl.uint<1>
  }

  // FIRRTL memories with a single mask bit for the entire word should lower to
  // a memory without any mask, and instead have that single mask bit be merged
  // into the enable condition on the write port.
  //
  // CHECK-LABEL: hw.module @FoldSingleMaskBitIntoEnable
  firrtl.module @FoldSingleMaskBitIntoEnable(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>, in %addr: !firrtl.uint<4>, in %data: !firrtl.uint<42>, in %mask: !firrtl.uint<1>) {
    // CHECK: %mem = seq.firmem 0, 1, undefined, port_order : <12 x 42>
    // CHECK: [[TMP:%.+]] = comb.and %en, %mask :
    // CHECK: seq.firmem.write_port %mem[{{%.+}}] = {{%.+}}, clock {{%.+}} enable [[TMP]] :
    %mem_w = firrtl.mem Undefined {depth = 12 : i64, name = "mem", portNames = ["w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    %mem_w.clk = firrtl.subfield %mem_w[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    %mem_w.en = firrtl.subfield %mem_w[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    %mem_w.addr = firrtl.subfield %mem_w[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    %mem_w.data = firrtl.subfield %mem_w[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    %mem_w.mask = firrtl.subfield %mem_w[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>

    firrtl.matchingconnect %mem_w.clk, %clk : !firrtl.clock
    firrtl.matchingconnect %mem_w.en, %en : !firrtl.uint<1>
    firrtl.matchingconnect %mem_w.addr, %addr : !firrtl.uint<4>
    firrtl.matchingconnect %mem_w.data, %data : !firrtl.uint<42>
    firrtl.matchingconnect %mem_w.mask, %mask : !firrtl.uint<1>
  }

  // FIRRTL memories with a single mask bit for the entire word should lower to
  // a memory without any mask, and instead have that single mask bit be merged
  // into the mode operand on the read-write port.
  //
  // CHECK-LABEL: hw.module @FoldSingleMaskBitIntoMode
  firrtl.module @FoldSingleMaskBitIntoMode(in %clk : !firrtl.clock, in %en : !firrtl.uint<1>, in %addr : !firrtl.uint<4>, in %wdata : !firrtl.uint<42>, in %wmode: !firrtl.uint<1>, in %wmask: !firrtl.uint<1>) {
    // CHECK: %mem = seq.firmem 0, 1, undefined, port_order : <12 x 42>
    // CHECK: [[TMP:%.+]] = comb.and %wmode, %wmask :
    // CHECK: seq.firmem.read_write_port %mem[{{%.+}}] = {{%.+}} if [[TMP]], clock {{%.+}} enable {{%.+}} :
    %mem_rw = firrtl.mem Undefined {depth = 12 : i64, name = "mem", portNames = ["rw"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    %mem_rw.clk = firrtl.subfield %mem_rw[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    %mem_rw.en = firrtl.subfield %mem_rw[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    %mem_rw.addr = firrtl.subfield %mem_rw[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    %mem_rw.wdata = firrtl.subfield %mem_rw[wdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    %mem_rw.rdata = firrtl.subfield %mem_rw[rdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    %mem_rw.wmask = firrtl.subfield %mem_rw[wmask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    %mem_rw.wmode = firrtl.subfield %mem_rw[wmode] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.matchingconnect %mem_rw.clk, %clk : !firrtl.clock
    firrtl.matchingconnect %mem_rw.en, %en : !firrtl.uint<1>
    firrtl.matchingconnect %mem_rw.addr, %addr : !firrtl.uint<4>
    firrtl.matchingconnect %mem_rw.wdata, %wdata : !firrtl.uint<42>
    firrtl.matchingconnect %mem_rw.wmask, %wmask : !firrtl.uint<1>
    firrtl.matchingconnect %mem_rw.wmode, %wmode : !firrtl.uint<1>
  }

  // CHECK-LABEL: hw.module @MemInit
  firrtl.module @MemInit(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>, in %addr: !firrtl.uint<4>, in %data: !firrtl.uint<42>) {
    // CHECK: %mem1 = seq.firmem
    // CHECK-SAME: init = #seq.firmem.init<"mem.txt", false, false>
    %mem1_r = firrtl.mem Undefined {depth = 12 : i64, name = "mem1", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32, init = #firrtl.meminit<"mem.txt", false, false>} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem1_r.clk = firrtl.subfield %mem1_r[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem1_r.en = firrtl.subfield %mem1_r[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem1_r.addr = firrtl.subfield %mem1_r[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.matchingconnect %mem1_r.clk, %clk : !firrtl.clock
    firrtl.matchingconnect %mem1_r.en, %en : !firrtl.uint<1>
    firrtl.matchingconnect %mem1_r.addr, %addr : !firrtl.uint<4>

    // CHECK: %mem2 = seq.firmem
    // CHECK-SAME: init = #seq.firmem.init<"mem.txt", false, true>
    %mem2_r = firrtl.mem Undefined {depth = 12 : i64, name = "mem2", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32, init = #firrtl.meminit<"mem.txt", false, true>} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem2_r.clk = firrtl.subfield %mem2_r[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem2_r.en = firrtl.subfield %mem2_r[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem2_r.addr = firrtl.subfield %mem2_r[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.matchingconnect %mem2_r.clk, %clk : !firrtl.clock
    firrtl.matchingconnect %mem2_r.en, %en : !firrtl.uint<1>
    firrtl.matchingconnect %mem2_r.addr, %addr : !firrtl.uint<4>

    // CHECK: %mem3 = seq.firmem
    // CHECK-SAME: init = #seq.firmem.init<"mem.txt", true, false>
    %mem3_r = firrtl.mem Undefined {depth = 12 : i64, name = "mem3", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32, init = #firrtl.meminit<"mem.txt", true, false>} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem3_r.clk = firrtl.subfield %mem3_r[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem3_r.en = firrtl.subfield %mem3_r[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem3_r.addr = firrtl.subfield %mem3_r[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.matchingconnect %mem3_r.clk, %clk : !firrtl.clock
    firrtl.matchingconnect %mem3_r.en, %en : !firrtl.uint<1>
    firrtl.matchingconnect %mem3_r.addr, %addr : !firrtl.uint<4>
  }

  // CHECK-LABEL: hw.module @IncompleteRead
  firrtl.module @IncompleteRead(
    in %clock: !firrtl.clock,
    in %addr: !firrtl.uint<4>,
    in %en: !firrtl.uint<1>
  ) {
    // The read port has no use of the data field.
    //
    // CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, port_order : <12 x 42>
    // CHECK-NEXT: seq.firmem.read_port %mem[%addr], clock %clock enable %en :
    %mem_r = firrtl.mem Undefined {depth = 12 : i64, name = "mem", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem_r.clk = firrtl.subfield %mem_r[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem_r.en = firrtl.subfield %mem_r[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem_r.addr = firrtl.subfield %mem_r[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.matchingconnect %mem_r.clk, %clock : !firrtl.clock
    firrtl.matchingconnect %mem_r.en, %en : !firrtl.uint<1>
    firrtl.matchingconnect %mem_r.addr, %addr : !firrtl.uint<4>
  }

  // CHECK-LABEL: hw.module @Depth1
  firrtl.module @Depth1(
    in %clock: !firrtl.clock,
    in %addr: !firrtl.uint<1>,
    in %en: !firrtl.uint<1>,
    out %data: !firrtl.uint<42>
  ) {
    // CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, port_order : <1 x 42>
    // CHECK-NEXT: [[RDATA:%.+]] = seq.firmem.read_port %mem[%addr], clock %clock enable %en :
    // CHECK-NEXT: hw.output [[RDATA]]
    %mem_r = firrtl.mem Undefined {depth = 1 : i64, name = "mem", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem_r.clk = firrtl.subfield %mem_r[clk] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem_r.addr = firrtl.subfield %mem_r[addr] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem_r.en = firrtl.subfield %mem_r[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem_r.data = firrtl.subfield %mem_r[data] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.connect %mem_r.clk, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %mem_r.addr, %addr : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %mem_r.en, %en : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %data, %mem_r.data : !firrtl.uint<42>, !firrtl.uint<42>
  }

  // CHECK-LABEL: hw.module @ExcessiveDepth
  firrtl.module @ExcessiveDepth(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>, in %addr31: !firrtl.uint<31>, in %addr33: !firrtl.uint<33>, in %data: !firrtl.uint<42>) {
    // CHECK: seq.firmem 0, 1, undefined, port_order : <2147483648 x 42>
    // CHECK: seq.firmem 0, 1, undefined, port_order : <8589934592 x 42>
    %mem_0 = firrtl.mem Undefined {depth = 2147483648 : i64, name = "mem31", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<31>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem_0.clk = firrtl.subfield %mem_0[clk] : !firrtl.bundle<addr: uint<31>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem_0.en = firrtl.subfield %mem_0[en] : !firrtl.bundle<addr: uint<31>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem_0.addr = firrtl.subfield %mem_0[addr] : !firrtl.bundle<addr: uint<31>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.matchingconnect %mem_0.clk, %clk : !firrtl.clock
    firrtl.matchingconnect %mem_0.en, %en : !firrtl.uint<1>
    firrtl.matchingconnect %mem_0.addr, %addr31 : !firrtl.uint<31>

    %mem_1 = firrtl.mem Undefined {depth = 8589934592 : i64, name = "mem33", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<33>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem_1.clk = firrtl.subfield %mem_1[clk] : !firrtl.bundle<addr: uint<33>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem_1.en = firrtl.subfield %mem_1[en] : !firrtl.bundle<addr: uint<33>, en: uint<1>, clk: clock, data flip: uint<42>>
    %mem_1.addr = firrtl.subfield %mem_1[addr] : !firrtl.bundle<addr: uint<33>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.matchingconnect %mem_1.clk, %clk : !firrtl.clock
    firrtl.matchingconnect %mem_1.en, %en : !firrtl.uint<1>
    firrtl.matchingconnect %mem_1.addr, %addr33 : !firrtl.uint<33>
  }
}
