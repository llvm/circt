; RUN: firtool %s --split-input-file | FileCheck %s

; CHECK: Generated by

; CHECK-LABEL: module TestHarness(
FIRRTL version 4.0.0
circuit TestHarness:
  extmodule DUT:
    input clock: Clock
    output read: Probe<UInt<32>>
    output write: RWProbe<UInt<32>>

  public module TestHarness:
    input clock: Clock

    inst dut of DUT
    connect dut.clock, clock

    ; CHECK: fwrite
    ; CHECK-SAME: "%x", TestHarness.dut.`ref_DUT_read)
    printf(clock, UInt<1>(1), "%x", read(dut.read))
    ; CHECK: initial
    ; CHECK: force TestHarness.dut.`ref_DUT_write = 32'hDEADBEEF;
    force_initial(dut.write, UInt<32>(0hdeadbeef))

; CHECK: endmodule

; // -----

; CHECK: Generated by

; CHECK-LABEL: module DUT(

FIRRTL version 4.0.0
circuit DUT:
  extmodule NotTheRealName:
    input clock: Clock
    output read: Probe<UInt<32>>
    output write: RWProbe<UInt<32>>
    defname = Inner

  public module DUT:
    input clock: Clock
    output read: Probe<UInt<32>>
    output write: RWProbe<UInt<32>>

    inst i of NotTheRealName
    connect i.clock, clock
    define read = i.read
    define write = i.write

; CHECK-LABEL: FILE "ref_DUT.sv"
; CHECK-EMPTY:
; CHECK-NEXT: Generated by
; CHECK-NEXT: `define ref_DUT_read i.`ref_Inner_read
; CHECK-NEXT: `define ref_DUT_write i.`ref_Inner_write

; // -----

; CHECK: Generated by

; CHECK-LABEL: module Inner(
FIRRTL version 4.0.0
circuit Inner:
  public module Inner:
    input clock: Clock
    output read: Probe<UInt<32>>
    output write: RWProbe<UInt<32>>

    reg r : UInt<32>, clock
    define read = probe(r)
    define write = rwprobe(r)
    ; CHECK: reg  [31:0] [[REG:.+]];
    ; CHECK: wire [31:0] [[REG_READ:.+]] = [[REG]];

; CHECK-LABEL: FILE "ref_Inner.sv"
; CHECK-EMPTY:
; CHECK-NEXT: Generated by
; CHECK-NEXT: `define ref_Inner_read [[REG_READ]]
; CHECK-NEXT: `define ref_Inner_write [[REG]]
