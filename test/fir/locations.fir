; RUN: spt-translate -import-firrtl --mlir-print-debuginfo -verify-diagnostics %s | FileCheck %s

circuit basic :  @[CIRCUIT.scala 127]

  ; CHECK-LABEL: firrtl.module @MyModule(%in: !firrtl.uint, %out: !firrtl.flip<uint<8>>) {
  module MyModule :   @[FooBar.scala 369:27]
    input in: UInt
    output out: UInt<8>

    ; CHECK: firrtl.connect {{.*}}loc{{.*}}test/fir/locations.fir"
    out <= in

  ; MODULE: CHECK: } loc("FooBar.scala":369:27)


  ; CHECK-LABEL: firrtl.module @M2
  module M2 :   @[Filename With Space.scala 1:2]
    input in: UInt
  ; CHECK: } loc("Filename With Space.scala":1:2)

  ; CHECK-LABEL: firrtl.module @M3
  module M3 :   @[NoSpaceInLocator]   ; expected-warning {{ignoring unknown @ info record format}}
    input in: UInt
  ; CHECK: } loc({{.*}}test/fir/locations.fir"

; CIRCUIT: CHECK: } loc("CIRCUIT.scala":127:0)
