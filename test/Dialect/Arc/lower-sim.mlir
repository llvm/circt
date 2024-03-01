// RUN: arcilator %s --emit-mlir | FileCheck %s

hw.module @id(in %i: i8, in %j: i8, out o: i8) {
    hw.output %i : i8
}

// CHECK-DAG: llvm.mlir.global internal constant @[[format_str:.*]]("result = %0.8p\0A\00")
// CHECK-DAG: llvm.mlir.global internal constant @[[format_str2:.*]]("result2 = %0.8p\0A\00")
// CHECK-LABEL: llvm.func @full
func.func @full() {
    %c = arith.constant 24 : i8

    // CHECK-DAG: %[[c:.*]] = llvm.mlir.constant(24 : i8)
    // CHECK-DAG: %[[zero:.*]] = llvm.mlir.constant(0 : i8)
    // CHECK-DAG: %[[size:.*]] = llvm.mlir.constant(3 : i64)
    // CHECK-DAG: %[[state:.*]] = llvm.call @malloc(%[[size:.*]]) :
    // CHECK: "llvm.intr.memset"(%[[state]], %[[zero]], %[[size]]) <{isVolatile = false}>
    arc.sim.instantiate @id as %model {
        // CHECK-NEXT: llvm.store %[[c]], %[[state]] : i8
        arc.sim.set_input %model, "i" = %c : i8, !arc.sim.instance<@id>

        // CHECK-NEXT: %[[j_ptr:.*]] = llvm.getelementptr %[[state]][1] : (!llvm.ptr) -> !llvm.ptr, i8
        // CHECK-NEXT: llvm.store %[[c]], %[[j_ptr]] : i8
        arc.sim.set_input %model, "j" = %c : i8, !arc.sim.instance<@id>

        // CHECK-NEXT: llvm.call @id_eval(%[[state]])
        arc.sim.step %model : !arc.sim.instance<@id>

        // CHECK-NEXT: %[[o_ptr:.*]] = llvm.getelementptr %[[state]][2] : (!llvm.ptr) -> !llvm.ptr, i8
        // CHECK-NEXT: %[[result:.*]] = llvm.load %[[o_ptr]] : !llvm.ptr -> i8
        %result = arc.sim.get_port %model, "o" : i8, !arc.sim.instance<@id>

        // CHECK-DAG: %[[to_print:.*]] = llvm.inttoptr %[[result]] : i8 to !llvm.ptr
        // CHECK-DAG: %[[format_str_ptr:.*]] = llvm.mlir.addressof @[[format_str]] : !llvm.ptr
        // CHECK: llvm.call @printf(%[[format_str_ptr]], %[[to_print]])
        arc.sim.emit "result", %result : i8

        // CHECK-DAG: %[[format_str2_ptr:.*]] = llvm.mlir.addressof @[[format_str2]] : !llvm.ptr
        // CHECK: llvm.call @printf(%[[format_str2_ptr]], %[[to_print]])
        arc.sim.emit "result2", %result : i8

        // CHECK: llvm.call @printf(%[[format_str_ptr]], %[[to_print]])
        arc.sim.emit "result", %result : i8
    }
    // CHECK: llvm.call @free(%[[state]])

    return
}
