; RUN: circt-translate -import-firrtl %s | FileCheck %s --check-prefix=MLIR
; RUN: circt-translate -import-firrtl %s | circt-translate -export-firrtl | FileCheck %s --check-prefix=FIR

; Round-trip of the required inline register clock-edge / reset-kind /
; reset-polarity qualifiers through .fir import and ExportFIRRTL. Every clock
; edge x reset kind x reset polarity combination must import to the matching
; attributes and export back to the identical .fir syntax.

FIRRTL version 4.0.0
circuit Test :
  public module Test :
    input clock : Clock
    input reset : Reset
    input d : UInt<8>
    output q : UInt<8>

    ; MLIR: firrtl.reg interesting_name %clock {clockEdge = 0 : i32}
    ; FIR: reg posedgeReg : UInt<8>, posedge clock
    reg posedgeReg : UInt<8>, posedge clock

    ; MLIR: firrtl.reg interesting_name %clock {clockEdge = 1 : i32}
    ; FIR: reg negedgeReg : UInt<8>, negedge clock
    reg negedgeReg : UInt<8>, negedge clock

    ; MLIR: firrtl.regreset interesting_name %clock, %reset, %{{.+}} {clockEdge = 0 : i32, resetPolarity = 0 : i32, resetType = 0 : i32}
    ; FIR: regreset syncHigh : UInt<8>, posedge clock, sync activehigh reset,
    regreset syncHigh : UInt<8>, posedge clock, sync activehigh reset, UInt<8>(0)

    ; MLIR: firrtl.regreset interesting_name %clock, %reset, %{{.+}} {clockEdge = 1 : i32, resetPolarity = 1 : i32, resetType = 1 : i32}
    ; FIR: regreset asyncLow : UInt<8>, negedge clock, async activelow reset,
    regreset asyncLow : UInt<8>, negedge clock, async activelow reset, UInt<8>(0)

    ; MLIR: firrtl.regreset interesting_name %clock, %reset, %{{.+}} {clockEdge = 0 : i32, resetPolarity = 0 : i32, resetType = 1 : i32}
    ; FIR: regreset asyncHigh : UInt<8>, posedge clock, async activehigh reset,
    regreset asyncHigh : UInt<8>, posedge clock, async activehigh reset, UInt<8>(0)

    ; MLIR: firrtl.regreset interesting_name %clock, %reset, %{{.+}} {clockEdge = 1 : i32, resetPolarity = 1 : i32, resetType = 0 : i32}
    ; FIR: regreset syncLow : UInt<8>, negedge clock, sync activelow reset,
    regreset syncLow : UInt<8>, negedge clock, sync activelow reset, UInt<8>(0)

    connect posedgeReg, d
    connect negedgeReg, d
    connect syncHigh, d
    connect asyncLow, d
    connect asyncHigh, d
    connect syncLow, d
    connect q, posedgeReg
