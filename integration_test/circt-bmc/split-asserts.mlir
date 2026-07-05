// Without flattening, the asserts stay inside instantiated modules, where
// they cannot be scoped correctly; they are rejected rather than mis-scoped.
//  RUN: not circt-bmc %s -b 10 --module ModuleAsserts --shared-libs=%libz3 --flatten-modules=false 2>&1 | FileCheck %s
//  CHECK: error: assertions inside instantiated modules or called functions are not supported - inline them into the top module first (e.g. with --flatten-modules)

hw.module @OneAssert(in %in: i1) {
  verif.assert %in : i1
}

hw.module @ModuleAsserts(in %i0: i1, in %i1: i1) {
  hw.instance "a" @OneAssert(in: %i0: i1) -> ()
  hw.instance "b" @OneAssert(in: %i1: i1) -> ()
}

