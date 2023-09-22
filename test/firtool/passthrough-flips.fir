; Check firtool pipeline hoisting passthroughs with flips (run hoist pass after LowerTypes, presently).
; RUN: firtool -disable-hoisting-hw-passthrough=false %s | FileCheck %s --implicit-check-not module
; CHECK: module PassthroughFlips
; CHECK: endmodule
circuit PassthroughFlips:
  module Child:
    input x : {a: UInt<5>, flip b: UInt<3>}
    output y : {a: UInt<5>, flip b: UInt<3>}
    y <= x
  module PassthroughFlips:
    input x : {a: UInt<5>, flip b: UInt<3>}
    output y : {a: UInt<5>, flip b: UInt<3>}

    inst c1 of Child
    inst c2 of Child

    c1.x <= x
    c2.x <= c1.y
    y <= c2.y

