// RUN: cirt-opt -cse %s | FileCheck %s

// CHECK-LABEL: firrtl.module @And
firrtl.module @And(%in1: !firrtl.uint<4>, %in2: !firrtl.uint<4>,
                   %out1: !firrtl.flip<uint<4>>,
                   %out2: !firrtl.flip<uint<4>>) {
  // And operations should get CSE'd.

  // CHECK: %0 = firrtl.and %in1, %in2
  %0 = firrtl.and %in1, %in2 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  // CHECK-NEXT: firrtl.connect %out1, %0
  firrtl.connect %out1, %0 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

  // CHECK-NEXT: firrtl.connect %out2, %0
  %1 = firrtl.and %in1, %in2 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out2, %1 : !firrtl.flip<uint<4>>, !firrtl.uint<4>
}

// CHECK-LABEL: firrtl.module @Wire
firrtl.module @Wire() {

   // CHECK: %_t = firrtl.wire
   // CHECK-NEXT: %_t_0 = firrtl.wire
   %w1 = firrtl.wire {name = "_t"} : !firrtl.uint<1>
   %w2 = firrtl.wire {name = "_t"} : !firrtl.uint<1>

  // CHECK-NEXT: firrtl.connect %_t, %_t_0
  firrtl.connect %w1, %w2 : !firrtl.uint<1>, !firrtl.uint<1>
}
