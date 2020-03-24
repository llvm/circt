// RUN: spt-opt %s | FileCheck %s

//module MyModule :
//  input in: UInt<8>
//  output out: UInt<8>
//  out <= in
firrtl.module @MyModule(%in : !firrtl.uint<8>,
                        %out : !firrtl.uint<8> { firrtl.output }) {
  firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
}

// CHECK-LABEL: firrtl.module @MyModule(%in: !firrtl.uint<8>, %out: !firrtl.uint<8> {firrtl.output}) {
// CHECK-NEXT:    firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
// CHECK-NEXT:  }


//circuit Top :
//  module Top :
//    output out:UInt
//    input b:UInt<32>
//    input c:Analog<13>
//    input d:UInt<16>
//    out <= add(b,d)

firrtl.circuit "Top" {
  firrtl.module @Top(%out: !firrtl.uint {firrtl.output},
                     %b: !firrtl.uint<32>,
                     %c: !firrtl.analog<13>,
                     %d: !firrtl.uint<16>) {
    %3 = firrtl.add %b, %d : (!firrtl.uint<32>, !firrtl.uint<16>) -> !firrtl.uint<32>
    
    %4 = firrtl.invalid {firrtl.name = "Name"} : !firrtl.uint<16>
    %5 = firrtl.add %3, %4 : (!firrtl.uint<32>, !firrtl.uint<16>) -> !firrtl.uint<32>
    
    firrtl.connect %out, %5 : !firrtl.uint, !firrtl.uint<32>
  }
}

// CHECK-LABEL: firrtl.circuit "Top" {
// CHECK-NEXT:    firrtl.module @Top(%out: !firrtl.uint {firrtl.output},
// CHECK:                            %b: !firrtl.uint<32>, %c: !firrtl.analog<13>, %d: !firrtl.uint<16>) {
// CHECK-NEXT:      %0 = firrtl.add %b, %d : (!firrtl.uint<32>, !firrtl.uint<16>) -> !firrtl.uint<32>
// CHECK-NEXT:      %Name = firrtl.invalid {firrtl.name = "Name"} : !firrtl.uint<16>
// CHECK-NEXT:      %1 = firrtl.add %0, %Name : (!firrtl.uint<32>, !firrtl.uint<16>) -> !firrtl.uint<32>
// CHECK-NEXT:      firrtl.connect %out, %1 : !firrtl.uint, !firrtl.uint<32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }


// Test some hard cases of name handling.
firrtl.module @Mod2(%in : !firrtl.uint<8> { firrtl.name = "some name"},
                    %out : !firrtl.uint<8> { firrtl.output }) {
  firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
}

// CHECK-LABEL: firrtl.module @Mod2(%some_name: !firrtl.uint<8> {firrtl.name = "some name"},
// CHECK:                           %out: !firrtl.uint<8> {firrtl.output}) {
// CHECK-NEXT:    firrtl.connect %out, %some_name : !firrtl.uint<8>, !firrtl.uint<8>
// CHECK-NEXT:  }
