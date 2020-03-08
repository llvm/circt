// RUN: spt-opt %s | FileCheck %s

//module MyModule :
//  input in: UInt<8>
//  output out: UInt<8>
//  out <= in
firrtl.module @MyModule(%in : ui8,
                        %out : ui8 { firrtl.output }) {
  firrtl.connect %out, %in : ui8, ui8
}

// CHECK-LABEL: firrtl.module @MyModule(%arg0: ui8 {firrtl.name = "in"}, %arg1: ui8 {firrtl.name = "out", firrtl.output}) {
// CHECK-NEXT:    firrtl.connect %arg1, %arg0 : ui8, ui8
// CHECK-NEXT:  }


//circuit Top :
//  module Top :
//    output out:UInt
//    input b:UInt<32>
//    input d:UInt<16>
//    out <= add(b,d)

firrtl.circuit "Top" {
  firrtl.module @Top(%out : !firrtl.uint {firrtl.output},
                     %b : ui32,
                     %d : ui16) {
    //%0 = "firrtl.output"() {name = "out"} : () -> !firrtl.uint
    //%1 = "firrtl.input"() {name = "b"} : () -> ui32
    //%2 = "firrtl.input"() {name = "d"} : () -> ui16
    %3 = firrtl.add %b, %d : (ui32, ui16) -> ui32
    
    %4 = firrtl.invalid {name = "Name"} : ui16
    %5 = firrtl.add %3, %4 : (ui32, ui16) -> ui32
    
    firrtl.connect %out, %5 : !firrtl.uint, ui32
  }
}

// CHECK-LABEL: firrtl.circuit "Top" {
// CHECK-NEXT:    firrtl.module @Top(%arg0: !firrtl.uint {firrtl.name = "out", firrtl.output}, %arg1: ui32 {firrtl.name = "b"}, %arg2: ui16 {firrtl.name = "d"}) {
// CHECK-NEXT:      %0 = firrtl.add %arg1, %arg2 : (ui32, ui16) -> ui32
// CHECK-NEXT:      %1 = firrtl.invalid {name = "Name"} : ui16
// CHECK-NEXT:      %2 = firrtl.add %0, %1 : (ui32, ui16) -> ui32
// CHECK-NEXT:      firrtl.connect %arg0, %2 : !firrtl.uint, ui32
// CHECK-NEXT:    }
// CHECK-NEXT:  }

