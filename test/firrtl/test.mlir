// RUN: spt-opt %s | FileCheck %s

//module MyModule :
//  input in: UInt<8>
//  output out: UInt<8>
//  out <= in

firrtl.module "MyModule" {
  %0 = "firrtl.input"() {name = "in"} : () -> ui8
  %1 = "firrtl.output"() {name = "out"} : () -> ui8
  firrtl.connect %1, %0 : ui8, ui8
}

// CHECK-LABEL: firrtl.module "MyModule" {
// CHECK-NEXT:    %0 = "firrtl.input"() {name = "in"} : () -> ui8
// CHECK-NEXT:    %1 = "firrtl.output"() {name = "out"} : () -> ui8
// CHECK-NEXT:    firrtl.connect %1, %0 : ui8, ui8
// CHECK-NEXT:  }


//circuit Top :
//  module Top :
//    output out:UInt
//    input b:UInt<32>
//    input d:UInt<16>
//    out <= add(b,d)

firrtl.circuit "Top" {
  firrtl.module "Top" {
    %0 = "firrtl.output"() {name = "out"} : () -> !firrtl.uint
    %1 = "firrtl.input"() {name = "b"} : () -> ui32
    %2 = "firrtl.input"() {name = "d"} : () -> ui16
    %3 = firrtl.add %1, %2 : (ui32, ui16) -> ui32
    
    %4 = firrtl.invalid {name = "Name"} : ui16
    %5 = firrtl.add %3, %4 : (ui32, ui16) -> ui32
    
    firrtl.connect %0, %5 : !firrtl.uint, ui32
  }
}

// CHECK-LABEL: firrtl.circuit "Top" {
// CHECK-NEXT:    firrtl.module "Top" {
// CHECK-NEXT:      %0 = "firrtl.output"() {name = "out"} : () -> !firrtl.uint
// CHECK-NEXT:      %1 = "firrtl.input"() {name = "b"} : () -> ui32
// CHECK-NEXT:      %2 = "firrtl.input"() {name = "d"} : () -> ui16
// CHECK-NEXT:      %3 = firrtl.add %1, %2 : (ui32, ui16) -> ui32
// CHECK-NEXT:      %4 = firrtl.invalid {name = "Name"} : ui16
// CHECK-NEXT:      %5 = firrtl.add %3, %4 : (ui32, ui16) -> ui32
// CHECK-NEXT:      firrtl.connect %0, %5 : !firrtl.uint, ui32
// CHECK-NEXT:    }
// CHECK-NEXT:  }

