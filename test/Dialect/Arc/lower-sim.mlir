// RUN: arcilator %s --emit-mlir | FileCheck %s

hw.module @id(in %i: i8, in %j: i8, out o: i8) {
    hw.output %i : i8
}

// CHECK-LABEL: llvm.func @basic
func.func @basic() -> i8 {
    %c = arith.constant 24 : i8
    // CHECK-DAG: %[[c:.*]] = llvm.mlir.constant(24 : i8)
    // CHECK-DAG: %[[zero:.*]] = llvm.mlir.constant(0 : i8)
    // CHECK-DAG: %[[size:.*]] = llvm.mlir.constant(3 : i64)
    // CHECK-DAG: %[[state:.*]] = llvm.alloca %[[size]] x i8
    // CHECK-NEXT: "llvm.intr.memset"(%[[state]], %[[zero]], %[[size]]) <{isVolatile = false}>
    %model = arc.sim.instantiate : !arc.sim.instance<"id">

    // CHECK-NEXT: llvm.store %[[c]], %[[state]] : i8
    arc.sim.set_input %model, "i" = %c : i8, !arc.sim.instance<"id">

    // CHECK-NEXT: %[[j_ptr:.*]] = llvm.getelementptr %[[state]][1] : (!llvm.ptr) -> !llvm.ptr, i8
    // CHECK-NEXT: llvm.store %[[c]], %[[j_ptr]] : i8
    arc.sim.set_input %model, "j" = %c : i8, !arc.sim.instance<"id">

    // CHECK-NEXT: llvm.call @id_eval(%[[state]])
    arc.sim.step %model : !arc.sim.instance<"id">

    // CHECK-NEXT: %[[o_ptr:.*]] = llvm.getelementptr %[[state]][2] : (!llvm.ptr) -> !llvm.ptr, i8
    // CHECK-NEXT: %[[result:.*]] = llvm.load %[[o_ptr]] : !llvm.ptr -> i8
    %result = arc.sim.get_port %model, "o" : i8, !arc.sim.instance<"id">

    // CHECK-NEXT: llvm.return %[[result]]
    return %result : i8
}
