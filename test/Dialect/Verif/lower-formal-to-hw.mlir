// RUN: circt-opt --lower-formal-to-hw %s | FileCheck %s

hw.module @Foo(in %bar : i32, in %baz : i16) {}

// CHECK-LABEL: hw.module @FormalTop(in %symbolic_value_0 : i32, in %symbolic_value_1 : i16)
verif.formal @FormalTop {
  %0 = verif.symbolic_value : i32
  %1 = verif.symbolic_value : i16
  %2 = comb.extract %0 from 16 :  (i32) -> i16
  // CHECK: [[RES:%[0-9]+]] = comb.xor
  %3 = comb.xor %1, %2 : i16
  // CHECK-NEXT: hw.instance "foo" @Foo(bar: %symbolic_value_0: i32, baz: [[RES]]: i16)
  hw.instance "foo" @Foo(bar: %0: i32, baz: %3: i16) -> ()
}
