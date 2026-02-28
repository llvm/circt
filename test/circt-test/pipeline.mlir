// RUN: circt-test --ir %s | FileCheck %s
// RUN: circt-test --ir --ignore-contracts %s | FileCheck %s --check-prefix=NOCONTRACT

// CHECK: hw.module @Foo
// CHECK-NOT: verif.contract
// CHECK: verif.formal @Foo_CheckContract

// NOCONTRACT: hw.module @Foo
// NOCONTRACT-NOT: verif.contract
// NOCONTRACT-NOT: verif.formal @Foo

hw.module @Foo(in %a: i1, in %b: i1, out z: i1) {
  %0 = comb.xor %a, %b : i1
  %1 = verif.contract %0 : i1 {
    %2 = comb.add %a, %b : i1
    %3 = comb.icmp eq %1, %2 : i1
    verif.ensure %3 : i1
  }
  hw.output %1 : i1
}
