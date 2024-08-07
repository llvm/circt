// RUN: arcilator %s --run --jit-entry=main 2>&1 >/dev/null | FileCheck %s
// REQUIRES: arcilator-jit

// CHECK: - Init -

module {
  llvm.func @_arc_env_get_print_stream(i32) -> !llvm.ptr
  llvm.func @_arc_libc_fputs(!llvm.ptr, !llvm.ptr) -> i32
  llvm.mlir.global internal constant @global_init_str(" - Init -\0A\00") {addr_space = 0 : i32}

  arc.model @initmodel io !hw.modty<> {
  ^bb0(%arg0: !arc.storage):
    arc.passthrough {
      %dummy = llvm.mlir.constant(0 : i32) : i32
    }
    arc.initial {
      %cst0 = llvm.mlir.constant(0 : i32) : i32
      %stderr = llvm.call @_arc_env_get_print_stream(%cst0) : (i32) -> !llvm.ptr
      %str = llvm.mlir.addressof @global_init_str : !llvm.ptr
      %0 = llvm.call @_arc_libc_fputs(%str, %stderr) : (!llvm.ptr, !llvm.ptr) -> i32
    }
  }
  func.func @main() {
    arc.sim.instantiate @initmodel as %arg0 {
    }
    return
  }
}
