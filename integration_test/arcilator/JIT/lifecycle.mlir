// RUN: arcilator %s --run | FileCheck %s
// REQUIRES: arcilator-jit

// CHECK:      before
// CHECK-NEXT: initial
// CHECK-NEXT: eval
// CHECK-NEXT: eval
// CHECK-NEXT: eval
// CHECK-NEXT: final
// CHECK-NEXT: after

arc.model @Foo io !hw.modty<> {
^bb0(%arg0: !arc.storage):
  arc.initial {
    %0 = llvm.mlir.addressof @str_initial : !llvm.ptr
    func.call @puts(%0) : (!llvm.ptr) -> i32
  }
  arc.final {
    %1 = llvm.mlir.addressof @str_final : !llvm.ptr
    func.call @puts(%1) : (!llvm.ptr) -> i32
  }
  %2 = llvm.mlir.addressof @str_eval : !llvm.ptr
  func.call @puts(%2) : (!llvm.ptr) -> i32
}

func.func @entry() {
  %0 = llvm.mlir.addressof @str_before : !llvm.ptr
  func.call @puts(%0) : (!llvm.ptr) -> i32
  arc.sim.instantiate @Foo as %arg0 {
    arc.sim.step %arg0 : !arc.sim.instance<@Foo>
    arc.sim.step %arg0 : !arc.sim.instance<@Foo>
    arc.sim.step %arg0 : !arc.sim.instance<@Foo>
  }
  %1 = llvm.mlir.addressof @str_after : !llvm.ptr
  func.call @puts(%1) : (!llvm.ptr) -> i32
  return
}

func.func private @puts(!llvm.ptr) -> i32
llvm.mlir.global internal constant @str_before("before\00")
llvm.mlir.global internal constant @str_initial("initial\00")
llvm.mlir.global internal constant @str_eval("eval\00")
llvm.mlir.global internal constant @str_final("final\00")
llvm.mlir.global internal constant @str_after("after\00")
