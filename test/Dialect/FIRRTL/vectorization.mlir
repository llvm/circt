// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(vectorization)))' %s | FileCheck %s

firrtl.circuit "ElementWise" {
// CHECK-LABEL: @ElementWise
firrtl.module @ElementWise(in %a: !firrtl.vector<uint<1>, 2>, in %b: !firrtl.vector<uint<1>, 2>, out %c_0: !firrtl.vector<uint<1>, 2>, out %c_1: !firrtl.vector<uint<1>, 2>, out %c_2: !firrtl.vector<uint<1>, 2>) {
  // CHECK-NEXT: %0 = firrtl.elementwise_or %a, %b : (!firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>) -> !firrtl.vector<uint<1>, 2>
  // CHECK: firrtl.strictconnect %c_0_write, %0 : !firrtl.vector<uint<1>, 2>
  // CHECK-NEXT: %1 = firrtl.elementwise_and %a, %b : (!firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>) -> !firrtl.vector<uint<1>, 2>
  // CHECK: firrtl.strictconnect %c_1_write, %1 : !firrtl.vector<uint<1>, 2>
  // CHECK-NEXT: %2 = firrtl.elementwise_xor %a, %b : (!firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>) -> !firrtl.vector<uint<1>, 2>
  // CHECK: firrtl.strictconnect %c_2_write, %2 : !firrtl.vector<uint<1>, 2>
  %0 = firrtl.subindex %b[1] : !firrtl.vector<uint<1>, 2>
  %1 = firrtl.subindex %a[1] : !firrtl.vector<uint<1>, 2>
  %2 = firrtl.subindex %b[0] : !firrtl.vector<uint<1>, 2>
  %3 = firrtl.subindex %a[0] : !firrtl.vector<uint<1>, 2>
  %4 = firrtl.or %3, %2 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  %5 = firrtl.or %1, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  %6 = firrtl.vectorcreate %4, %5 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.vector<uint<1>, 2>
  %c_0_write = firrtl.wrapSink %c_0 : !firrtl.vector<uint<1>, 2>
  firrtl.strictconnect %c_0_write, %6 : !firrtl.vector<uint<1>, 2>
  %7 = firrtl.and %3, %2 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  %8 = firrtl.and %1, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  %9 = firrtl.vectorcreate %7, %8 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.vector<uint<1>, 2>
  %c_1_write = firrtl.wrapSink %c_1 : !firrtl.vector<uint<1>, 2>
  firrtl.strictconnect %c_1_write, %9 : !firrtl.vector<uint<1>, 2>
  %10 = firrtl.xor %3, %2 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  %11 = firrtl.xor %1, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  %12 = firrtl.vectorcreate %10, %11 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.vector<uint<1>, 2>
  %c_2_write = firrtl.wrapSink %c_2 : !firrtl.vector<uint<1>, 2>
  firrtl.strictconnect %c_2_write, %12 : !firrtl.vector<uint<1>, 2>
}
}

