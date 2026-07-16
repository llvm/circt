// RUN: circt-opt %s --arc-lower-state | FileCheck %s

func.func private @malloc(i64) -> !llvm.ptr

// Side-effecting module-level ops consuming or producing a signal's
// initializer value (the malloc from a module-scope class `new`, the object
// memset, and the typeinfo store) are one-time initialization actions and
// must all lower into the initial phase against the SAME allocation the
// signal captures. Lowering them per-eval re-executed the initializer chain
// on a fresh malloc every evaluation (leaked each eval) while the signal
// kept the pointer from its own, separate initial-phase clone -- so the
// object the design reads was never memset and had a null typeinfo.
// CHECK-LABEL: arc.model @ModuleScopeObjectInit
hw.module @ModuleScopeObjectInit() {
  %ti = llvm.mlir.addressof @typeinfo : !llvm.ptr
  %c0_i8 = llvm.mlir.constant(0 : i8) : i8
  %c24_i64 = llvm.mlir.constant(24 : i64) : i64
  %obj = llhd.sig %ptr : !llvm.ptr
  %ptr = arc.execute -> (!llvm.ptr) {
    %sz = llvm.mlir.constant(24 : i64) : i64
    %m = func.call @malloc(%sz) : (i64) -> !llvm.ptr
    arc.output %m : !llvm.ptr
  }
  "llvm.intr.memset"(%ptr, %c0_i8, %c24_i64) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  llvm.store %ti, %ptr : !llvm.ptr, !llvm.ptr
  hw.output
}
// CHECK:      arc.initial {
// CHECK:        [[PTR:%.+]] = arc.execute -> (!llvm.ptr) {
// CHECK:          func.call @malloc
// CHECK:        }
// CHECK:        arc.state_write %{{.+}} = [[PTR]] : <!llvm.ptr>
// CHECK:        "llvm.intr.memset"([[PTR]],
// CHECK:        llvm.store %{{.+}}, [[PTR]] : !llvm.ptr, !llvm.ptr
// CHECK:      }
// The initializer chain must not be duplicated into the per-eval body:
// exactly one malloc in the whole model.
// CHECK-NOT: func.call @malloc

llvm.mlir.global internal constant @typeinfo() : !llvm.struct<(ptr)> {
  %0 = llvm.mlir.undef : !llvm.struct<(ptr)>
  llvm.return %0 : !llvm.struct<(ptr)>
}
