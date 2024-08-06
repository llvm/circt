// RUN: arcilator %s --run --jit-entry=main 2>&1 >/dev/null | FileCheck -strict-whitespace %s
// REQUIRES: arcilator-jit

// CHECK: Hello World!{{[[:space:]]}}

module {

  llvm.func @_arc_env_get_print_stream(i32) -> !llvm.ptr
  llvm.func @_arc_libc_fprintf(!llvm.ptr, !llvm.ptr, ...) -> i32
  llvm.func @_arc_libc_fputc(i32, !llvm.ptr) -> i32
  llvm.func @_arc_libc_fputs(!llvm.ptr, !llvm.ptr) -> i32

  llvm.mlir.global internal constant @global_hello("He%c%co \00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @global_world("World\00") {addr_space = 0 : i32}

  arc.model @dut io !hw.modty<> {
  ^bb0(%arg0: !arc.storage):
    %cst0 = llvm.mlir.constant(0 : i32) : i32
    %ascii_em = llvm.mlir.constant(33 : i32) : i32
    %ascii_lf = llvm.mlir.constant(10 : i32) : i32
    %ascii_l = llvm.mlir.constant(108 : i32) : i32
    %stderr = llvm.call @_arc_env_get_print_stream(%cst0) : (i32) -> !llvm.ptr

    %hello = llvm.mlir.addressof @global_hello : !llvm.ptr
    %0 = llvm.call @_arc_libc_fprintf(%stderr, %hello, %ascii_l, %ascii_l) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, i32, i32) -> i32

    %world = llvm.mlir.addressof @global_world : !llvm.ptr
    %1 = llvm.call @_arc_libc_fputs(%world, %stderr) : (!llvm.ptr, !llvm.ptr) -> i32

    %2 = llvm.call @_arc_libc_fputc(%ascii_em, %stderr) : (i32, !llvm.ptr) -> i32
    %3 = llvm.call @_arc_libc_fputc(%ascii_lf, %stderr) : (i32, !llvm.ptr) -> i32
  }

  func.func @main() {
    arc.sim.instantiate @dut as %arg0 {
      arc.sim.step %arg0 : !arc.sim.instance<@dut>
    }
    return
  }
}
