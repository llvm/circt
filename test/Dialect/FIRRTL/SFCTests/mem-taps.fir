; RUN: firtool --verilog %s | FileCheck %s
; RUN: firtool --verilog -preserve-aggregate=1d-vec %s | FileCheck %s --check-prefix=AGGGREGATE
; RUN: firtool --verilog -lower-annotations-no-ref-type-ports %s | FileCheck %s --check-prefix=NOREFS
; RUN: firtool --parse-only %s | circt-opt --firrtl-probes-to-signals | firtool --verilog --format=mlir | FileCheck %s --check-prefix=PROBESTOSIGNALS

FIRRTL version 4.0.0
circuit Top : %[[
  {
    "class":"firrtl.transforms.DontTouchAnnotation",
    "target":"~Top|Top>memTap"
  },
  {
    "class":"sifive.enterprise.grandcentral.MemTapAnnotation",
    "source":"~Top|DUTModule>rf",
    "sink":[
      "~Top|Top>memTap[0]",
      "~Top|Top>memTap[1]",
      "~Top|Top>memTap[2]",
      "~Top|Top>memTap[3]",
      "~Top|Top>memTap[4]",
      "~Top|Top>memTap[5]",
      "~Top|Top>memTap[6]",
      "~Top|Top>memTap[7]"
    ]
  }
]]
  module DUTModule :
    input clock : Clock
    input reset : Reset
    output io : { flip addr : UInt<3>, flip dataIn : UInt<8>, flip wen : UInt<1>, dataOut : UInt<8>}

    cmem rf : UInt<8> [8]
    infer mport read = rf[io.addr], clock
    connect io.dataOut, read
    when io.wen :
      infer mport write = rf[io.addr], clock
      connect write, io.dataIn

  public module Top :
    input clock : Clock
    input reset : UInt<1>
    output io : { flip addr : UInt<3>, flip dataIn : UInt<8>, flip wen : UInt<1>, dataOut : UInt<8>}

    inst dut of DUTModule
    connect dut.clock, clock
    connect dut.reset, reset
    wire memTap : UInt<8>[8]
    invalidate memTap
    connect io.dataOut, dut.io.dataOut
    connect dut.io.wen, io.wen
    connect dut.io.dataIn, io.dataIn
    connect dut.io.addr, io.addr

; CHECK:      module Top(
; CHECK-NOT:  module
; CHECK:        wire [7:0] memTap_0 = Top.dut.rf_ext.Memory[0];
; CHECK-NEXT:   wire [7:0] memTap_1 = Top.dut.rf_ext.Memory[1];
; CHECK-NEXT:   wire [7:0] memTap_2 = Top.dut.rf_ext.Memory[2];
; CHECK-NEXT:   wire [7:0] memTap_3 = Top.dut.rf_ext.Memory[3];
; CHECK-NEXT:   wire [7:0] memTap_4 = Top.dut.rf_ext.Memory[4];
; CHECK-NEXT:   wire [7:0] memTap_5 = Top.dut.rf_ext.Memory[5];
; CHECK-NEXT:   wire [7:0] memTap_6 = Top.dut.rf_ext.Memory[6];
; CHECK-NEXT:   wire [7:0] memTap_7 = Top.dut.rf_ext.Memory[7];
; AGGGREGATE:       wire [7:0][7:0] memTap =
; AGGGREGATE-NEXT{LITERAL}: {{Top.dut.rf_ext.Memory[7]},
; AGGGREGATE-NEXT:           {Top.dut.rf_ext.Memory[6]},
; AGGGREGATE-NEXT:           {Top.dut.rf_ext.Memory[5]},
; AGGGREGATE-NEXT:           {Top.dut.rf_ext.Memory[4]},
; AGGGREGATE-NEXT:           {Top.dut.rf_ext.Memory[3]},
; AGGGREGATE-NEXT:           {Top.dut.rf_ext.Memory[2]},
; AGGGREGATE-NEXT:           {Top.dut.rf_ext.Memory[1]},
; AGGGREGATE-NEXT:           {Top.dut.rf_ext.Memory[0]}};
; CHECK:      endmodule

; NOREFS:      module DUTModule(
; NOREFS-NOT:  endmodule
; NOREFS:        rf_8x8 rf_ext (
; NOREFS:          .R1_data (memTap_7),
; NOREFS:          .R2_data (memTap_6),
; NOREFS:          .R3_data (memTap_5),
; NOREFS:          .R4_data (memTap_4),
; NOREFS:          .R5_data (memTap_3),
; NOREFS:          .R6_data (memTap_2),
; NOREFS:          .R7_data (memTap_1),
; NOREFS:          .R8_data (memTap_0)
; NOREFS:   )
; NOREFS:      endmodule
; NOREFS:      DUTModule dut (
; NOREFS-NOT:  endmodule
; NOREFS:        .memTap_0 (memTap_0),
; NOREFS-NEXT:   .memTap_1 (memTap_1),
; NOREFS-NEXT:   .memTap_2 (memTap_2),
; NOREFS-NEXT:   .memTap_3 (memTap_3),
; NOREFS-NEXT:   .memTap_4 (memTap_4),
; NOREFS-NEXT:   .memTap_5 (memTap_5),
; NOREFS-NEXT:   .memTap_6 (memTap_6),
; NOREFS-NEXT:   .memTap_7 (memTap_7)
; NOREFS-NEXT: )

; PROBESTOSIGNALS:      module DUTModule(
; PROBESTOSIGNALS-NOT:  endmodule
; PROBESTOSIGNALS:        rf_8x8 rf_ext (
; PROBESTOSIGNALS:          .R1_data (memTap_0_7),
; PROBESTOSIGNALS:          .R2_data (memTap_0_6),
; PROBESTOSIGNALS:          .R3_data (memTap_0_5),
; PROBESTOSIGNALS:          .R4_data (memTap_0_4),
; PROBESTOSIGNALS:          .R5_data (memTap_0_3),
; PROBESTOSIGNALS:          .R6_data (memTap_0_2),
; PROBESTOSIGNALS:          .R7_data (memTap_0_1),
; PROBESTOSIGNALS:          .R8_data (memTap_0_0)
; PROBESTOSIGNALS:   )
; PROBESTOSIGNALS:      endmodule
; PROBESTOSIGNALS:      DUTModule dut (
; PROBESTOSIGNALS-NOT:  endmodule
; PROBESTOSIGNALS:        .memTap_0_0 (memTap_0),
; PROBESTOSIGNALS-NEXT:   .memTap_0_1 (memTap_1),
; PROBESTOSIGNALS-NEXT:   .memTap_0_2 (memTap_2),
; PROBESTOSIGNALS-NEXT:   .memTap_0_3 (memTap_3),
; PROBESTOSIGNALS-NEXT:   .memTap_0_4 (memTap_4),
; PROBESTOSIGNALS-NEXT:   .memTap_0_5 (memTap_5),
; PROBESTOSIGNALS-NEXT:   .memTap_0_6 (memTap_6),
; PROBESTOSIGNALS-NEXT:   .memTap_0_7 (memTap_7)
; PROBESTOSIGNALS-NEXT: )
