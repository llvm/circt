; RUN: spt-translate -import-firrtl --mlir-print-debuginfo %s | FileCheck %s

circuit basic :  

  ; CHECK-LABEL: firrtl.module @MyModule(%in: !firrtl.uint, %out: !firrtl.flip<uint<8>>) {
  module MyModule :   @[FooBar.scala 369:27]
    input in: UInt
    output out: UInt<8>

    ; CHECK: firrtl.connect {{.*}}loc{{.*}}test/fir/locations.fir"
    out <= in

  ; MODULE: CHECK: } loc(unknown)
  ; CIRCUIT: CHECK: } loc{{.*}}test/fir/locations.fir":3:1
