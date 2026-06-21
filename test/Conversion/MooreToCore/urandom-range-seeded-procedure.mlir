// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK: func.func private @__circt_urandom_range(i32, i32, !llvm.ptr) -> i32
// CHECK-LABEL: hw.module @URandomSeedModule
moore.module @URandomSeedModule(in %lo: !moore.i32, in %hi: !moore.i32) {
  %seed = moore.variable : <!moore.i32>
  moore.procedure initial {
    // CHECK: llhd.process
    // CHECK: llvm.alloca
    // CHECK: llhd.prb
    // CHECK: llvm.store
    // CHECK: func.call @__circt_urandom_range
    // CHECK: llvm.load
    // CHECK: llhd.drv
    %0 = moore.builtin.urandom_range %lo, %hi, %seed
    moore.return
  }
}
