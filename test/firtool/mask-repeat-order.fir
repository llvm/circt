; RUN: firtool --disable-all-randomization %s | FileCheck %s

; This test checks the end-to-end firtool lowering for the aggregate memory
; mask repeat ordering bug reported in issue #10105.
;
; See: https://github.com/llvm/circt/issues/10105
FIRRTL version 6.0.0

circuit MaskNested:
  ; CHECK-LABEL: module MaskNested(
  ; CHECK{LITERAL}: .W0_mask ({{3{wen}}, mem_wr_mask_b_1, mem_wr_mask_b_0})
  public module MaskNested:
    input clock: Clock
    input wen: UInt<1>
    input addr: UInt<2>
    input sel: UInt<1>
    input in_a: UInt<3>
    input in_b: UInt<1>
    output out_a: UInt<3>
    output out_b: UInt<1>

    cmem mem: { a: UInt<3>, b: UInt<1>[2] }[4]
    infer mport rd = mem[addr], clock
    infer mport wr = mem[addr], clock

    when wen:
      connect wr.a, in_a
      connect wr.b[sel], in_b

    connect out_a, rd.a
    connect out_b, rd.b[sel]
