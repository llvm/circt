// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-translate %s --export-quartus-tcl | FileCheck %s --check-prefix=TCL

hw.module.extern @Foo()

hw.module @top() {
  // CHECK: hw.instance "foo1" @Foo() {"loc:memBank1" = #msft.physloc<M20K, 4, 10, 1>}
  // TCL:   set_location_assignment M20K_X4_Y10_N1 -to $parent|foo1|memBank1
  hw.instance "foo1" @Foo() {"loc:memBank1" = #msft.physloc<M20K, 4, 10, 1> } : () -> ()

  // CHECK: hw.instance "foo2" @Foo() {"loc:memBank2" = #msft.physloc<M20K, 5, 10, 1>}
  // TCL:   set_location_assignment M20K_X5_Y10_N1 -to $parent|foo2|memBank2
  hw.instance "foo2" @Foo() {"loc:memBank2" = #msft.physloc<M20K, 5, 10, 1> } : () -> ()
}
