// RUN: circt-opt %s | FileCheck %s

//module MyModule :
//  input in: UInt<8>
//  output out: UInt<8>
//  out <= in
firrtl.module @MyModule(%in : !firrtl.uint<8>,
                        %out : !firrtl.flip<uint<8>>) {
  firrtl.connect %out, %in : !firrtl.flip<uint<8>>, !firrtl.uint<8>
}

// CHECK-LABEL: firrtl.module @MyModule(%in: !firrtl.uint<8>, %out: !firrtl.flip<uint<8>>) {
// CHECK-NEXT:    firrtl.connect %out, %in : !firrtl.flip<uint<8>>, !firrtl.uint<8>
// CHECK-NEXT:  }


//circuit Top :
//  module Top :
//    output out:UInt
//    input b:UInt<32>
//    input c:Analog<13>
//    input d:UInt<16>
//    out <= add(b,d)

firrtl.circuit "Top" {
  firrtl.module @Top(%out: !firrtl.flip<uint>,
                     %b: !firrtl.uint<32>,
                     %c: !firrtl.analog<13>,
                     %d: !firrtl.uint<16>) {
    %3 = firrtl.add %b, %d : (!firrtl.uint<32>, !firrtl.uint<16>) -> !firrtl.uint<32>
    
    firrtl.invalid %c : !firrtl.analog<13>
    %5 = firrtl.add %3, %d : (!firrtl.uint<32>, !firrtl.uint<16>) -> !firrtl.uint<32>
    
    firrtl.connect %out, %5 : !firrtl.flip<uint>, !firrtl.uint<32>
  }
}

// CHECK-LABEL: firrtl.circuit "Top" {
// CHECK-NEXT:    firrtl.module @Top(%out: !firrtl.flip<uint>,
// CHECK:                            %b: !firrtl.uint<32>, %c: !firrtl.analog<13>, %d: !firrtl.uint<16>) {
// CHECK-NEXT:      %0 = firrtl.add %b, %d : (!firrtl.uint<32>, !firrtl.uint<16>) -> !firrtl.uint<32>
// CHECK-NEXT:      firrtl.invalid %c : !firrtl.analog<13>
// CHECK-NEXT:      %1 = firrtl.add %0, %d : (!firrtl.uint<32>, !firrtl.uint<16>) -> !firrtl.uint<32>
// CHECK-NEXT:      firrtl.connect %out, %1 : !firrtl.flip<uint>, !firrtl.uint<32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }


// Test some hard cases of name handling.
firrtl.module @Mod2(%in : !firrtl.uint<8> { firrtl.name = "some name"},
                    %out : !firrtl.flip<uint<8>>) {
  firrtl.connect %out, %in : !firrtl.flip<uint<8>>, !firrtl.uint<8>
}

// CHECK-LABEL: firrtl.module @Mod2(%some_name: !firrtl.uint<8> {firrtl.name = "some name"},
// CHECK:                           %out: !firrtl.flip<uint<8>>) {
// CHECK-NEXT:    firrtl.connect %out, %some_name : !firrtl.flip<uint<8>>, !firrtl.uint<8>
// CHECK-NEXT:  }


// Modules may be completely empty.
// CHECK-LABEL: firrtl.module @no_ports() {
firrtl.module @no_ports() {
}

// stdIntCast can work with clock inputs/outputs too.
// CHECK-LABEL: @ClockCast
firrtl.module @ClockCast(%clock: !firrtl.clock, %in1 : i1) {
  // CHECK: %0 = firrtl.stdIntCast %clock : (!firrtl.clock) -> i1
  %0 = firrtl.stdIntCast %clock : (!firrtl.clock) -> i1

  // CHECK: %1 = firrtl.stdIntCast %in1 : (i1) -> !firrtl.clock
  %1 = firrtl.stdIntCast %in1 : (i1) -> !firrtl.clock
}
