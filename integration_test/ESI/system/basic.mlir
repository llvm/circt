// REQUIRES: rtl-sim
// RUN: circt-opt %s --lower-esi-to-physical --lower-esi-ports --lower-esi-to-rtl -verify-diagnostics > %t1.mlir
// RUN: circt-translate %t1.mlir -emit-verilog -verify-diagnostics > %t2.sv
// RUN: circt-rtl-sim.py %t2.sv %INC%/circt/Dialect/ESI/ESIPrimitives.sv %S/supplement.sv --cycles 25 | FileCheck %s

module {
  rtl.externmodule @IntCountProd(%clk: i1, %rstn: i1) -> (%ints: !esi.channel<i32>)
  rtl.externmodule @IntAcc(%clk: i1, %rstn: i1, %ints: !esi.channel<i32>) -> ()
  rtl.module @top(%clk: i1, %rstn: i1) -> () {
    %intStream = rtl.instance "prod" @IntCountProd(%clk, %rstn) : (i1, i1) -> (!esi.channel<i32>)
    %intStreamBuffered = esi.buffer %clk, %rstn, %intStream {stages=2, name="intChan"} : i32
    rtl.instance "acc" @IntAcc(%clk, %rstn, %intStreamBuffered) : (i1, i1, !esi.channel<i32>) -> ()
  }
  // CHECK:      [driver] Starting simulation
  // CHECK-NEXT: Total:          0
  // CHECK-NEXT: Data:     0
  // CHECK-NEXT: Total:          0
  // CHECK-NEXT: Data:     0
  // CHECK-NEXT: Total:          0
  // CHECK-NEXT: Data:     0
  // CHECK-NEXT: Total:          0
  // CHECK-NEXT: Data:     1
  // CHECK-NEXT: Total:          1
  // CHECK-NEXT: Data:     2
  // CHECK-NEXT: Total:          3
  // CHECK-NEXT: Data:     3
  // CHECK-NEXT: Total:          6
  // CHECK-NEXT: Data:     4
  // CHECK-NEXT: Total:         10
  // CHECK-NEXT: Data:     5
  // CHECK-NEXT: Total:         15
  // CHECK-NEXT: Data:     6
  // CHECK-NEXT: Total:         21
  // CHECK-NEXT: Data:     7
  // CHECK-NEXT: Total:         28
  // CHECK-NEXT: Data:     8
  // CHECK-NEXT: Total:         36
  // CHECK-NEXT: Data:     9
  // CHECK-NEXT: Total:         45
  // CHECK-NEXT: Data:    10
  // CHECK-NEXT: Total:         55
  // CHECK-NEXT: Data:    11
  // CHECK-NEXT: Total:         66
  // CHECK-NEXT: Data:    12
  // CHECK-NEXT: Total:         78
  // CHECK-NEXT: Data:    13
  // CHECK-NEXT: Total:         91
  // CHECK-NEXT: Data:    14
  // CHECK-NEXT: Total:        105
  // CHECK-NEXT: Data:    15
  // CHECK-NEXT: Total:        120
  // CHECK-NEXT: Data:    16
  // CHECK-NEXT: Total:        136
  // CHECK-NEXT: Data:    17
  // CHECK-NEXT: Total:        153
  // CHECK-NEXT: Data:    18
  // CHECK-NEXT: Total:        171
  // CHECK-NEXT: Data:    19
  // CHECK-NEXT: Total:        190
  // CHECK-NEXT: Data:    20
  // CHECK-NEXT: Total:        210
  // CHECK-NEXT: Data:    21
  // CHECK-NEXT: Total:        231
  // CHECK-NEXT: Data:    22
  // CHECK-NEXT: [driver] Ending simulation at tick #59
}
