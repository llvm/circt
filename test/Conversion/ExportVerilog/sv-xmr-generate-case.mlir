// RUN: circt-opt %s -export-verilog -verify-diagnostics | FileCheck %s

// Verifies hierarchical XMR paths include `generate` / `generate.case` block
// names (issue #9972).

hw.hierpath @hier [@Case1::@sym, @Foo::@sym]

hw.module @Foo(in %x: i1 {hw.exportPort = #hw<innerSym@sym>}) {
  hw.output
}

hw.module @Bar() {
  %true = hw.constant true
  sv.initial {
    %0 = sv.xmr.ref @hier : !hw.inout<i1>
    sv.force %0, %true : i1
  }
  hw.output
}

hw.module @Case1<NUM: i64>(in %x: i1) {
  sv.generate "foo_case": {
    sv.generate.case #hw.param.decl.ref<"NUM"> : i64 [
      case (0 : i64, "case0") {
        hw.instance "test" sym @sym @Foo(x: %x: i1) -> ()
        hw.instance "bar" @Bar() -> ()
      }
      case (unit, "dflt") {
      }
    ]
  }
  hw.output
}

// CHECK-LABEL: module Foo
// CHECK-LABEL: module Bar
// CHECK: force Case1.foo_case.case0.test.x
