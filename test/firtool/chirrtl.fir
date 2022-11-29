; RUN: firtool --repl-seq-mem --repl-seq-mem-file="dummy" -disable-imdce %s | FileCheck %s

; This is testing that CHIRRTL enable inference is working as intended.  If the
; no-op wires and nodes are not optimized away, then both ports should always
; be enabled.  If they are accidentally removed before the lower-chirrtl pass,
; then they won't be enabled.

circuit test: %[[{
    "class": "sifive.enterprise.firrtl.MarkDUTAnnotation",
    "target":"~test|test"
  }, 
  {
    "class": "sifive.enterprise.firrtl.ConvertMemToRegOfVecAnnotation$"
  }]]
  module test:
    input p: UInt<1>
    input addr: UInt<4>
    input clock: Clock
    output out0: UInt<8>
    output out1: UInt<8>

    smem testmem : UInt<8>[16]

    node _T_0 = addr
    when p:
      read mport testport0 = testmem[_T_0], clock
    out0 <= testport0

    wire _T_1: UInt<4>
    _T_1 <= addr
    when p:
      read mport testport1 = testmem[_T_1], clock
    out1 <= testport1
  ; No write to the memory, initialized with invalid, hence optimized away.
  ; CHECK: assign [[G1:.*]] = 8'h0;
  ; CHECK: assign [[G2:.*]] = 8'h0;
  ; CHECK: assign out0 = [[G1]];
  ; CHECK: assign out1 = [[G2]];

