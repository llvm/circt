// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-flatten-memory)))' %s | FileCheck  %s


firrtl.circuit "Mem" {
  firrtl.module public  @Mem(in %clock: !firrtl.clock, in %rAddr: !firrtl.uint<4>, in %rEn: !firrtl.uint<1>, out %rData: !firrtl.bundle<a: uint<8>, b: uint<8>>, in %wAddr: !firrtl.uint<4>, in %wEn: !firrtl.uint<1>, in %wMask: !firrtl.bundle<a: uint<1>, b: uint<1>>, in %wData: !firrtl.bundle<a: uint<8>, b: uint<8>>) {
    %memory_r, %memory_w = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r", "w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    %0 = firrtl.subfield %memory_r(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>) -> !firrtl.clock
    firrtl.strictconnect %0, %clock : !firrtl.clock
    %1 = firrtl.subfield %memory_r(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>) -> !firrtl.uint<1>
    firrtl.strictconnect %1, %rEn : !firrtl.uint<1>
    %2 = firrtl.subfield %memory_r(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>) -> !firrtl.uint<4>
    firrtl.strictconnect %2, %rAddr : !firrtl.uint<4>
    %3 = firrtl.subfield %memory_r(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>) -> !firrtl.bundle<a: uint<8>, b: uint<8>>
    firrtl.strictconnect %rData, %3 : !firrtl.bundle<a: uint<8>, b: uint<8>>
    %4 = firrtl.subfield %memory_w(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.clock
    firrtl.strictconnect %4, %clock : !firrtl.clock
    %5 = firrtl.subfield %memory_w(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<1>
    firrtl.strictconnect %5, %wEn : !firrtl.uint<1>
    %6 = firrtl.subfield %memory_w(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<4>
    firrtl.strictconnect %6, %wAddr : !firrtl.uint<4>
    %7 = firrtl.subfield %memory_w(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.strictconnect %7, %wMask : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %8 = firrtl.subfield %memory_w(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<8>, b: uint<8>>
    firrtl.strictconnect %8, %wData : !firrtl.bundle<a: uint<8>, b: uint<8>>
    // ---------------------------------------------------------------------------------
    // After flattenning the memory data
    // CHECK: %[[memory_r:.+]], %[[memory_w:.+]] = firrtl.mem Undefined  {depth = 16 : i64, name = "memory", portNames = ["r", "w"], readLatency = 0 : i32, writeLatency = 1 : i32}
    // CHECK-SAME: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<16>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<16>, mask: uint<2>>
    // CHECK: %[[memory_r_0:.+]] = firrtl.wire  {name = "memory_r"} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>
    // CHECK: %[[v0:.+]] = firrtl.subfield %[[memory_r]](0)
    // CHECK: firrtl.strictconnect %[[v0]], %[[memory_r_addr:.+]] :
    // CHECK: %[[v1:.+]] = firrtl.subfield %[[memory_r]](1)
    // CHECK: firrtl.strictconnect %[[v1]], %[[memory_r_en:.+]] :
    // CHECK: %[[v2:.+]] = firrtl.subfield %[[memory_r]](2)
    // CHECK: firrtl.strictconnect %[[v2]], %[[memory_r_clk:.+]] :
    // CHECK: %[[v3:.+]] = firrtl.subfield %[[memory_r]](3)
    //
    // ---------------------------------------------------------------------------------
    // Read ports
    // CHECK:  %[[v4:.+]] = firrtl.bitcast %[[v3]] : (!firrtl.uint<16>) -> !firrtl.bundle<a: uint<8>, b: uint<8>>
    // CHECK:  firrtl.strictconnect %[[memory_r_data:.+]], %[[v4]] :
    // --------------------------------------------------------------------------------
    // Write Ports
    // CHECK:  %[[memory_w_1:.+]] = firrtl.wire  {name = "memory_w"} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    // CHECK:  %[[v9:.+]] = firrtl.subfield %[[memory_w]](3)
    // CHECK:  %[[v17:.+]] = firrtl.bitcast %[[v15:.+]] : (!firrtl.bundle<a: uint<8>, b: uint<8>>) -> !firrtl.uint<16>
    // CHECK:  firrtl.strictconnect %[[v9]], %[[v17]]
    //
    // --------------------------------------------------------------------------------
    // Mask Ports
    //  CHECK: %[[v11:.+]] = firrtl.subfield %[[memory_w]](4)
    //  CHECK: %[[v12:.+]] = firrtl.bitcast %[[v18:.+]] : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<2>
    //  CHECK: firrtl.strictconnect %[[v11]], %[[v12]]
    // --------------------------------------------------------------------------------
    // Connections to module ports
    // CHECK:  %[[v21:.+]] = firrtl.subfield %[[memory_r_0]](2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>) -> !firrtl.clock
    // CHECK:  firrtl.strictconnect %[[v21]], %clock :
    // CHECK:  %[[v22:.+]]  = firrtl.subfield %[[memory_r_0]](1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>) -> !firrtl.uint<1>
    // CHECK:  firrtl.strictconnect %[[v22]], %rEn : !firrtl.uint<1>
    // CHECK:  %[[v23:.+]]  = firrtl.subfield %[[memory_r_0]](0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>) -> !firrtl.uint<4>
    // CHECK:  firrtl.strictconnect %[[v23]], %rAddr : !firrtl.uint<4>
    // CHECK:  %[[v24:.+]]  = firrtl.subfield %[[memory_r_0]](3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>) -> !firrtl.bundle<a: uint<8>, b: uint<8>>
    // CHECK:  firrtl.strictconnect %rData, %[[v24]] : !firrtl.bundle<a: uint<8>, b: uint<8>>
    // CHECK:  %[[v25:.+]]  = firrtl.subfield %[[memory_w_1]](2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.clock
    // CHECK:  firrtl.strictconnect %[[v25]], %clock : !firrtl.clock
    // CHECK:  %[[v26:.+]]  = firrtl.subfield %[[memory_w_1]](1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<1>
    // CHECK:  firrtl.strictconnect %[[v26]], %wEn : !firrtl.uint<1>
    // CHECK:  %[[v27:.+]]  = firrtl.subfield %[[memory_w_1]](0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<4>
    // CHECK:  firrtl.strictconnect %[[v27]], %wAddr : !firrtl.uint<4>
    // CHECK:  %[[v28:.+]]  = firrtl.subfield %[[memory_w_1]](4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
    // CHECK:  firrtl.strictconnect %[[v28]], %wMask : !firrtl.bundle<a: uint<1>, b: uint<1>>
    // CHECK:  %[[v29:.+]]  = firrtl.subfield %[[memory_w_1]](3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<8>, b: uint<8>>
    // CHECK:  firrtl.strictconnect %[[v29]], %wData : !firrtl.bundle<a: uint<8>, b: uint<8>>
  }

firrtl.module @MemoryRWSplit(in %clock: !firrtl.clock, in %rwEn: !firrtl.uint<1>, in %rwMode: !firrtl.uint<1>, in %rwAddr: !firrtl.uint<4>, in %rwMask: !firrtl.bundle<a: uint<1>, b: uint<1>>, in %rwDataIn: !firrtl.bundle<a: uint<8>, b: uint<9>>, out %rwDataOut: !firrtl.bundle<a: uint<8>, b: uint<9>>) {
  %memory_rw = firrtl.mem Undefined  {depth = 16 : i64, groupID = 1 : ui32, name = "memory", portNames = ["rw"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint<8>, b: uint<9>>, wmode: uint<1>, wdata: bundle<a: uint<8>, b: uint<9>>, wmask: bundle<a: uint<1>, b: uint<1>>>
  // CHECK:  %memory_rw = firrtl.mem Undefined  {depth = 16 : i64, groupID = 1 : ui32, name = "memory", portNames = ["rw"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<17>, wmode: uint<1>, wdata: uint<17>, wmask: uint<17>>
  // CHECK:  %[[memory_rw_0:.+]] = firrtl.wire  {name = "memory_rw"} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint<8>, b: uint<9>>, wmode: uint<1>, wdata: bundle<a: uint<8>, b: uint<9>>, wmask: bundle<a: uint<1>, b: uint<1>>>
  %0 = firrtl.subfield %memory_rw(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint<8>, b: uint<9>>, wmode: uint<1>, wdata: bundle<a: uint<8>, b: uint<9>>, wmask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<8>, b: uint<9>>
  %1 = firrtl.subfield %memory_rw(5) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint<8>, b: uint<9>>, wmode: uint<1>, wdata: bundle<a: uint<8>, b: uint<9>>, wmask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<8>, b: uint<9>>
  %2 = firrtl.subfield %memory_rw(6) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint<8>, b: uint<9>>, wmode: uint<1>, wdata: bundle<a: uint<8>, b: uint<9>>, wmask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
  %3 = firrtl.subfield %memory_rw(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint<8>, b: uint<9>>, wmode: uint<1>, wdata: bundle<a: uint<8>, b: uint<9>>, wmask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<1>
  %4 = firrtl.subfield %memory_rw(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint<8>, b: uint<9>>, wmode: uint<1>, wdata: bundle<a: uint<8>, b: uint<9>>, wmask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<4>
  %5 = firrtl.subfield %memory_rw(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint<8>, b: uint<9>>, wmode: uint<1>, wdata: bundle<a: uint<8>, b: uint<9>>, wmask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<1>
  %6 = firrtl.subfield %memory_rw(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint<8>, b: uint<9>>, wmode: uint<1>, wdata: bundle<a: uint<8>, b: uint<9>>, wmask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.clock
  firrtl.connect %6, %clock : !firrtl.clock, !firrtl.clock
  firrtl.connect %5, %rwEn : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %4, %rwAddr : !firrtl.uint<4>, !firrtl.uint<4>
  firrtl.connect %3, %rwMode : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %2, %rwMask : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
  firrtl.connect %1, %rwDataIn : !firrtl.bundle<a: uint<8>, b: uint<9>>, !firrtl.bundle<a: uint<8>, b: uint<9>>
  firrtl.connect %rwDataOut, %0 : !firrtl.bundle<a: uint<8>, b: uint<9>>, !firrtl.bundle<a: uint<8>, b: uint<9>>
  // CHECK:  %[[v6:.+]] = firrtl.subfield %[[memory_rw_0]](3) :
  // CHECK:  %[[v7:.+]] = firrtl.subfield %memory_rw(3) :
  // CHECK:  %[[v8:.+]] = firrtl.bitcast %[[v7]] :
  // CHECK:  firrtl.strictconnect %[[v6]], %[[v8]] :
  // CHECK:  %[[v9:.+]] = firrtl.subfield %[[memory_rw_0]](4) :
  // CHECK:  %[[v10:.+]] = firrtl.subfield %memory_rw(4) :
  // CHECK:  firrtl.strictconnect %[[v10]], %[[v9]] : !firrtl.uint<1>
  // CHECK:  %[[v11:.+]] = firrtl.subfield %[[memory_rw_0]](5) :
  // CHECK:  %[[v12:.+]] = firrtl.subfield %memory_rw(5) :
  // CHECK:  %[[v13:.+]] = firrtl.bitcast %[[v11]] : (!firrtl.bundle<a: uint<8>, b: uint<9>>) -> !firrtl.uint<17>
  // CHECK:  firrtl.strictconnect %[[v12]], %[[v13]] :
  // CHECK:  %[[v14:.+]] = firrtl.subfield %[[memory_rw_0]](6) :
  // CHECK:  %[[v15:.+]] = firrtl.subfield %memory_rw(6) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<17>, wmode: uint<1>, wdata: uint<17>, wmask: uint<17>>) -> !firrtl.uint<17>
  // CHECK:  %[[v16:.+]] = firrtl.bitcast %14 : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<2>
  // CHECK:  %[[v17:.+]] = firrtl.bits %16 0 to 0 : (!firrtl.uint<2>) -> !firrtl.uint<1>
  // CHECK:  %[[v18:.+]] = firrtl.cat %[[v17]], %[[v17]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  // CHECK:  %[[v19:.+]] = firrtl.cat %[[v17]], %[[v18]] : (!firrtl.uint<1>, !firrtl.uint<2>) -> !firrtl.uint<3>
  // CHECK:  %[[v24:.+]] = firrtl.cat %[[v17]], %[[v23:.+]] : (!firrtl.uint<1>, !firrtl.uint<7>) -> !firrtl.uint<8>
  // CHECK:  %[[v25:.+]] = firrtl.bits %16 1 to 1 : (!firrtl.uint<2>) -> !firrtl.uint<1>
  // CHECK:  %[[v26:.+]] = firrtl.cat %[[v25]], %[[v24]] : (!firrtl.uint<1>, !firrtl.uint<8>) -> !firrtl.uint<9>
  // CHECK:  %[[v27:.+]] = firrtl.cat %[[v25]], %[[v26]] : (!firrtl.uint<1>, !firrtl.uint<9>) -> !firrtl.uint<10>
  // CHECK:  %[[v28:.+]] = firrtl.cat %[[v25]], %[[v27]] : (!firrtl.uint<1>, !firrtl.uint<10>) -> !firrtl.uint<11>
  // CHECK:  %[[v34:.+]] = firrtl.cat %[[v25]], %[[v33:.+]] : (!firrtl.uint<1>, !firrtl.uint<16>) -> !firrtl.uint<17>
  // CHECK:  firrtl.strictconnect %[[v15]], %[[v34]] :
  // Ensure 0 bit fields are handled properly.
  %ram_MPORT = firrtl.mem Undefined  {depth = 4 : i64, groupID = 1 : ui32, name = "ram", portNames = ["MPORT"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data: bundle<entry: bundle<a: uint<0>, b: uint<1>, c: uint<2>>>, mask: bundle<entry: bundle<a: uint<1>, b: uint<1>, c: uint<1>>>>
  // CHECK: %ram_MPORT = firrtl.mem Undefined  {depth = 4 : i64, groupID = 1 : ui32, name = "ram", portNames = ["MPORT"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data: uint<3>, mask: uint<3>>

}


  firrtl.module @ZeroBitMasks(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io: !firrtl.bundle<a: uint<0>, b: uint<20>>) {
    %invalid = firrtl.invalidvalue : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %invalid_0 = firrtl.invalidvalue : !firrtl.bundle<a: uint<0>, b: uint<20>>
    %ram_MPORT = firrtl.mem Undefined  {depth = 1 : i64, groupID = 1 : ui32, name = "ram", portNames = ["MPORT"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<a: uint<0>, b: uint<20>>, mask: bundle<a: uint<1>, b: uint<1>>>
    %3 = firrtl.subfield %ram_MPORT(3) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<a: uint<0>, b: uint<20>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<0>, b: uint<20>>
    firrtl.strictconnect %3, %invalid_0 : !firrtl.bundle<a: uint<0>, b: uint<20>>
    %4 = firrtl.subfield %ram_MPORT(4) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<a: uint<0>, b: uint<20>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.strictconnect %4, %invalid : !firrtl.bundle<a: uint<1>, b: uint<1>>
    // CHECK:  %ram_MPORT = firrtl.mem Undefined  {depth = 1 : i64, groupID = 1 : ui32, name = "ram", portNames = ["MPORT"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<20>, mask: uint<1>>
    // CHECK:  %ram_MPORT_1 = firrtl.wire  {name = "ram_MPORT"} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<a: uint<0>, b: uint<20>>, mask: bundle<a: uint<1>, b: uint<1>>>
    // CHECK:  %[[v6:.+]] = firrtl.subfield %ram_MPORT_1(3) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<a: uint<0>, b: uint<20>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<0>, b: uint<20>>
    // CHECK:  %[[v7:.+]] = firrtl.subfield %ram_MPORT(3) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<20>, mask: uint<1>>) -> !firrtl.uint<20>
    // CHECK:  %[[v8:.+]] = firrtl.bitcast %6 : (!firrtl.bundle<a: uint<0>, b: uint<20>>) -> !firrtl.uint<20>
    // CHECK:  firrtl.strictconnect %7, %8 : !firrtl.uint<20>
    // CHECK:  %[[v9:.+]] = firrtl.subfield %ram_MPORT_1(4) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<a: uint<0>, b: uint<20>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
    // CHECK:  %[[v10:.+]] = firrtl.subfield %ram_MPORT(4) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<20>, mask: uint<1>>) -> !firrtl.uint<1>
    // CHECK:  %[[v11:.+]] = firrtl.bitcast %9 : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<2>
    // CHECK:  %[[v12:.+]] = firrtl.bits %11 0 to 0 : (!firrtl.uint<2>) -> !firrtl.uint<1>
    // CHECK:  %[[v13:.+]] = firrtl.bits %11 1 to 1 : (!firrtl.uint<2>) -> !firrtl.uint<1>
    // CHECK:  firrtl.strictconnect %[[v10]], %[[v13]] : !firrtl.uint<1>
    // CHECK:  %[[v14:.+]] = firrtl.subfield %ram_MPORT_1(3) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<a: uint<0>, b: uint<20>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<0>, b: uint<20>>
    // CHECK:  firrtl.strictconnect %[[v14]], %invalid_0 : !firrtl.bundle<a: uint<0>, b: uint<20>>
    // CHECK:  %[[v15:.+]] = firrtl.subfield %ram_MPORT_1(4) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<a: uint<0>, b: uint<20>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.connect %3, %io : !firrtl.bundle<a: uint<0>, b: uint<20>>, !firrtl.bundle<a: uint<0>, b: uint<20>>
  }

  // Tests all the cases when the memory is ignored and not flattened.
  firrtl.module @ZeroBitMem(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io: !firrtl.bundle<a: uint<0>, b: uint<20>>) {
    // Case 1: No widths.
    %ram_MPORT = firrtl.mem Undefined  {depth = 1 : i64, groupID = 1 : ui32, name = "ram", portNames = ["MPORT"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<>, mask: bundle<>>
    // CHECK: %ram_MPORT = firrtl.mem Undefined  {depth = 1 : i64, groupID = 1 : ui32, name = "ram", portNames = ["MPORT"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<>, mask: bundle<>>
    // Case 2: All widths of the data add up to zero.
    %ram_MPORT1 = firrtl.mem Undefined  {depth = 1 : i64, groupID = 1 : ui32, name = "ram", portNames = ["MPORT"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<a: uint<0>, b: uint<0>>, mask: bundle<a: uint<1>, b: uint<1>>>
    // CHECK: = firrtl.mem Undefined  {depth = 1 : i64, groupID = 1 : ui32, name = "ram", portNames = ["MPORT"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<a: uint<0>, b: uint<0>>, mask: bundle<a: uint<1>, b: uint<1>>>
    // Case 3: Aggregate contains only a single element.
    %single     = firrtl.mem Undefined  {depth = 1 : i64, groupID = 1 : ui32, name = "ram", portNames = ["MPORT"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<b: uint<10>>, mask: bundle<b: uint<1>>>
    // CHECK: = firrtl.mem Undefined  {depth = 1 : i64, groupID = 1 : ui32, name = "ram", portNames = ["MPORT"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<b: uint<10>>, mask: bundle<b: uint<1>>>
    // Case 4: Ground Type with zero width.
    %ram_MPORT2, %ram_io_deq_bits_MPORT = firrtl.mem  Undefined  {depth = 2 : i64, name = "ram", portNames = ["MPORT", "io_deq_bits_MPORT"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>, !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<0>>
    // CHECK:  = firrtl.mem  Undefined  {depth = 2 : i64, name = "ram", portNames = ["MPORT", "io_deq_bits_MPORT"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>, !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<0>>
    // Case 5: Any Ground Type.
    %ram_MPORT3, %ram_io_deq_bits_MPORT2 = firrtl.mem  Undefined  {depth = 2 : i64, name = "ram", portNames = ["MPORT", "io_deq_bits_MPORT"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<1>, mask: uint<1>>, !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    // CHECK:  = firrtl.mem  Undefined  {depth = 2 : i64, name = "ram", portNames = ["MPORT", "io_deq_bits_MPORT"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<1>, mask: uint<1>>, !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
  }
}
