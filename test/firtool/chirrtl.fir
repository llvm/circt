; RUN: firtool -blackbox-memory -remove-unused-ports=false -verilog %s | FileCheck %s

; This is testing that CHIRRTL enable inference is working as intended.  If the
; no-op wires and nodes are not optimized away, then both ports should always
; be enabled.  If they are accidentally removed before the lower-chirrtl pass,
; then they won't be enabled.

circuit test:
  module test:
    input p: UInt<1>
    input addr: UInt<4>
    input clock: Clock
    output out0: UInt<8>
    output out1: UInt<8>

    ; CHECK: testmem testmem (
    smem testmem : UInt<8>[16]

    ; CHECK: .testport0_en   (1'h1),
    node _T_0 = addr
    when p:
      read mport testport0 = testmem[_T_0], clock
    out0 <= testport0

    ; CHECK: .testport1_en   (1'h1),
    wire _T_1: UInt<4>
    _T_1 <= addr
    when p:
      read mport testport1 = testmem[_T_1], clock
    out1 <= testport1

