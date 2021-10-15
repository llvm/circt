// RUN: circt-reduce %s --test %S/trivial.sh --test-arg firtool | FileCheck %s

firrtl.circuit "Foo" {
  // CHECK: firrtl.extmodule @FooFooFoo
  firrtl.module @FooFooFoo(in %x: !firrtl.uint<1>, out %y: !firrtl.uint<1>) {
    firrtl.connect %y, %x : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK: firrtl.extmodule @FooFooBar
  firrtl.module @FooFooBar(in %x: !firrtl.uint<1>, out %y: !firrtl.uint<1>) {
    firrtl.connect %y, %x : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK: firrtl.module @FooFoo
  firrtl.module @FooFoo(in %x: !firrtl.uint<1>, out %y: !firrtl.uint<1>) {
    %x0_x, %x0_y = firrtl.instance x0 @FooFooFoo(in x: !firrtl.uint<1>, out y: !firrtl.uint<1>)
    %x1_x, %x1_y = firrtl.instance x1 @FooFooBar(in x: !firrtl.uint<1>, out y: !firrtl.uint<1>)
    firrtl.connect %x0_x, %x : !firrtl.uint<1>, !firrtl.uint<1>
    // Skip %x1_x to trigger a "sink not fully initialized" warning
    firrtl.connect %y, %x0_y : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK: firrtl.extmodule @FooBar
  firrtl.module @FooBar(in %x: !firrtl.uint<1>, out %y: !firrtl.uint<1>) {
    firrtl.connect %y, %x : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK: firrtl.extmodule @Foo
  firrtl.module @Foo(in %x: !firrtl.uint<1>, out %y: !firrtl.uint<1>) {
    %x0_x, %x0_y = firrtl.instance x0 @FooFoo(in x: !firrtl.uint<1>, out y: !firrtl.uint<1>)
    %x1_x, %x1_y = firrtl.instance x1 @FooBar(in x: !firrtl.uint<1>, out y: !firrtl.uint<1>)
    firrtl.connect %x0_x, %x : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %x1_x, %x : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %y, %x0_y : !firrtl.uint<1>, !firrtl.uint<1>
  }
}
