//  RUN: not circt-bmc %s -b 10 --module ModuleAsserts --shared-libs=%libz3 2>&1 | FileCheck %s
//  CHECK: error: bounded model checking problems with multiple assertions are not yet correctly handled - instead, you can assert the conjunction of your assertions

hw.module @OneAssert(in %in: i1) {
  verif.assert %in : i1
}

hw.module @ModuleAsserts(in %i0: i1, in %i1: i1) {
  hw.instance "a" @OneAssert(in: %i0: i1) -> ()
  hw.instance "b" @OneAssert(in: %i1: i1) -> ()
}

