; RUN: circt-translate -import-firrtl --mlir-print-debuginfo -verify-diagnostics -mlir-print-local-scope %s | FileCheck %s

circuit MyModule :  @[CIRCUIT.scala 127]

  ; CHECK-LABEL: firrtl.module @MyModule(in %in: !firrtl.uint, out %out: !firrtl.uint<8>) {
  module MyModule :   @[FooBar.scala 369:27]
    input in: UInt
    output out: UInt<8>

    ; CHECK: firrtl.connect {{.*}}loc("Somewhere.galaxy":42:1)
    out <= in   @[Somewhere.galaxy 42:1]

    ; CHECK: firrtl.skip loc("SKipLoc.scala":42:24)
    skip  @[SKipLoc.scala 42:24]

  ; MODULE: CHECK: } loc("FooBar.scala":369:27)


  ; CHECK-LABEL: firrtl.module @M2
  module M2 :   @[Filename With Space.scala 1:2]
    input in: UInt
  ; CHECK: } loc("Filename With Space.scala":1:2)

  ; CHECK-LABEL: firrtl.module @M3
  module M3 :   @[NoSpaceInLocator]   ; expected-warning {{ignoring unknown @ info record format}}
    input in: UInt
  ; CHECK: } loc({{.*}}test{{.}}Dialect{{.}}FIRRTL{{.}}parse-locations.fir"


  ; CHECK-LABEL: firrtl.module @Subexpressions
  module Subexpressions :
    input in: UInt
    output auto : { out_0 : UInt<1> }

    ; CHECK: %out_0 = firrtl.wire
    wire out_0 : { member : { 0 : { reset : UInt<1>}}}

    ; CHECK: %0 = firrtl.subfield %auto("out_0") {{.*}} loc("Field":173:49)
    ; CHECK: %1 = firrtl.subfield %out_0("member") {{.*}} loc("Field":173:49)
    ; CHECK: %2 = firrtl.subfield %1("0") {{.*}} loc("Field":173:49)
    ; CHECK: %3 = firrtl.subfield %2("reset") : {{.*}} loc("Field":173:49)
    ; CHECK: firrtl.partialconnect %0, %3 : {{.*}} loc("Field":173:49)
    auto.out_0 <- out_0.member.0.reset @[Field 173:49]

    ; Fused locators: https://github.com/llvm/circt/issues/224

    ; CHECK: %thing = firrtl.wire{{.*}} loc(fused["XX.scala":123:19, "YY.haskell":309:14, "ZZ.swift":3:4])
    wire thing : SInt<4> @[XX.scala 123:19 YY.haskell 309:14 ZZ.swift 3:4]

    ; CHECK: %other_thing = firrtl.wire{{.*}} loc("File with space.perl":1:23)
    wire other_thing : SInt<4> @[File with space.perl 1:23]

; CIRCUIT: CHECK: } loc("CIRCUIT.scala":127:0)
