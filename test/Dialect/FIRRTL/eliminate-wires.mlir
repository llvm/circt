// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-eliminate-wires)))' %s | FileCheck %s

firrtl.circuit "TopLevel" {

  // CHECK-LABEL: @TopLevel
  firrtl.module @TopLevel(in %source: !firrtl.uint<1>, 
                             out %sink: !firrtl.uint<1>) {
    // CHECK-NOT: firrtl.wire
    %w = firrtl.wire : !firrtl.uint<1>
    %w_read, %w_write = firrtl.deduplex %w : !firrtl.uint<1>
    firrtl.strictconnect %w_write, %source : !firrtl.uint<1>
    %wn = firrtl.not %w : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %x = firrtl.wire : !firrtl.uint<1>
    %x_read, %x_write = firrtl.deduplex %x : !firrtl.uint<1>
    firrtl.strictconnect %x_write, %wn : !firrtl.uint<1>
    %sink_read, %sink_write = firrtl.deduplex %sink : !firrtl.uint<1>
    firrtl.strictconnect %sink_write, %x : !firrtl.uint<1>
    firrtl.strictconnect %sink_write, %w : !firrtl.uint<1>
  }

  // CHECK-LABEL: @Foo
  firrtl.module private @Foo() {
    %a = firrtl.wire : !firrtl.uint<3>
    %b = firrtl.wire : !firrtl.uint<3>
    %invalid_ui3 = firrtl.invalidvalue : !firrtl.uint<3>
    %b_read, %b_write = firrtl.deduplex %b : !firrtl.uint<3>
    firrtl.strictconnect %b_write, %invalid_ui3 : !firrtl.uint<3>
    %a_read, %a_write = firrtl.deduplex %a : !firrtl.uint<3>
    firrtl.strictconnect %a_write, %b : !firrtl.uint<3>
    // CHECK: %[[inv:.*]] = firrtl.invalidvalue : !firrtl.uint<3>
    // CHECK-NEXT:  %b = firrtl.node %[[inv]] : !firrtl.uint<3>
    // CHECK-NEXT:  %a = firrtl.node %b : !firrtl.uint<3>
  }
}