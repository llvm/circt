// RUN: spt-opt %s | FileCheck %s

//module MyModule :
//  input in: UInt<8>
//  output out: UInt<8>
//  out <= in

"firrtl.module"() ( {
  %0 = "firrtl.input"() {name = "in"} : () -> ui8
  %1 = "firrtl.output"() {name = "out"} : () -> ui8
  "firrtl.connect"(%1, %0) : (ui8, ui8) -> ()
  "firrtl.done"() : () -> ()
}) {name = "MyModule"} : () -> ()

// CHECK-LABEL: "firrtl.module"() ( {
// CHECK-NEXT:    %0 = "firrtl.input"() {name = "in"} : () -> ui8
// CHECK-NEXT:    %1 = "firrtl.output"() {name = "out"} : () -> ui8
// CHECK-NEXT:    "firrtl.connect"(%1, %0) : (ui8, ui8) -> ()
// CHECK-NEXT:    "firrtl.done"() : () -> ()
// CHECK-NEXT:  }) {name = "MyModule"} : () -> ()


//circuit Top :
//  module Top :
//    output out:UInt
//    input b:UInt<32>
//    input d:UInt<16>
//    out <= add(b,d)

"firrtl.circuit"() ( {
  "firrtl.module"() ( {
    %0 = "firrtl.output"() {name = "out"} : () -> !firrtl.uint
    %1 = "firrtl.input"() {name = "b"} : () -> ui32
    %2 = "firrtl.input"() {name = "d"} : () -> ui16
    %3 = "firrtl.add"(%1, %2) : (ui32, ui16) -> ui32
    "firrtl.connect"(%0, %3) : (!firrtl.uint, ui32) -> ()
    "firrtl.done"() : () -> ()
  }) {name = "Top"} : () -> ()
  "firrtl.done"() : () -> ()
}) {name = "Top"} : () -> ()

// CHECK-LABEL: "firrtl.circuit"() ( {
// CHECK-NEXT:    "firrtl.module"() ( {
// CHECK-NEXT:      %0 = "firrtl.output"() {name = "out"} : () -> !firrtl.uint
// CHECK-NEXT:      %1 = "firrtl.input"() {name = "b"} : () -> ui32
// CHECK-NEXT:      %2 = "firrtl.input"() {name = "d"} : () -> ui16
// CHECK-NEXT:      %3 = "firrtl.add"(%1, %2) : (ui32, ui16) -> ui32
// CHECK-NEXT:      "firrtl.connect"(%0, %3) : (!firrtl.uint, ui32) -> ()
// CHECK-NEXT:      "firrtl.done"() : () -> ()
// CHECK-NEXT:    }) {name = "Top"} : () -> ()
// CHECK-NEXT:    "firrtl.done"() : () -> ()
// CHECK-NEXT:  }) {name = "Top"} : () -> ()

