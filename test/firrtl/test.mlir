// RUN: spt-opt %s | FileCheck %s

//module MyModule :
//  input in: UInt<8>
//  output out: UInt<8>
//  out <= in
firrtl.module @MyModule(%in : ui8,
                        %out : ui8 { firrtl.output }) {
  firrtl.connect %out, %in : ui8, ui8
}

// CHECK-LABEL: firrtl.module @MyModule(%in: ui8 {firrtl.name = "in"}, %out: ui8 {firrtl.name = "out", firrtl.output}) {
// CHECK-NEXT:    firrtl.connect %out, %in : ui8, ui8
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
    %3 = firrtl.add %b, %d : (ui32, ui16) -> ui32
    
    %4 = firrtl.invalid {firrtl.name = "Name"} : ui16
    %5 = firrtl.add %3, %4 : (ui32, ui16) -> ui32
    
    firrtl.connect %out, %5 : !firrtl.uint, ui32
  }
}

// CHECK-LABEL: firrtl.circuit "Top" {
// CHECK-NEXT:    firrtl.module @Top(%out: !firrtl.uint {firrtl.name = "out", firrtl.output}, %b: ui32 {firrtl.name = "b"}, %d: ui16 {firrtl.name = "d"}) {
// CHECK-NEXT:      %0 = firrtl.add %b, %d : (ui32, ui16) -> ui32
// CHECK-NEXT:      %Name = firrtl.invalid {firrtl.name = "Name"} : ui16
// CHECK-NEXT:      %1 = firrtl.add %0, %Name : (ui32, ui16) -> ui32
// CHECK-NEXT:      firrtl.connect %out, %1 : !firrtl.uint, ui32
// CHECK-NEXT:    }
// CHECK-NEXT:  }

