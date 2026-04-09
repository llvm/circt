// RUN: circt-verilog --ir-moore %s | circt-opt --moore-create-vtables --convert-moore-to-core --convert-to-llvm --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.mlir.global internal constant @"testClassVirtualInt::vtable"()
// CHECK-DAG: llvm.mlir.global internal constant @"testDerivedVirtualInt::vtable"()

// CHECK-LABEL: llvm.func @"testClassVirtualInt::adjust"(
// CHECK-SAME: %[[THIS0:arg[0-9]+]]: !llvm.ptr, %[[X0:arg[0-9]+]]: i32) -> i32
// CHECK:      %[[BIAS_IDX:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:      %[[BIAS_PTR:.*]] = llvm.getelementptr %[[THIS0]][%[[BIAS_IDX]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"testClassVirtualInt", (struct<(ptr, ptr)>, i32)>
// CHECK:      %[[BIAS:.*]] = llvm.load %[[BIAS_PTR]] : !llvm.ptr -> i32
// CHECK:      %[[SUM0:.*]] = comb.add %[[X0]], %[[BIAS]] : i32
// CHECK:      llvm.return %[[SUM0]] : i32

// CHECK-LABEL: llvm.func @"testDerivedVirtualInt::adjust"(
// CHECK-SAME: %[[THIS1:arg[0-9]+]]: !llvm.ptr, %[[X1:arg[0-9]+]]: i32) -> i32
// CHECK:      %[[DELTA_IDX:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:      %[[DELTA_PTR:.*]] = llvm.getelementptr %[[THIS1]][%[[DELTA_IDX]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"testDerivedVirtualInt", (struct<(ptr, ptr)>, struct<"testClassVirtualInt", (struct<(ptr, ptr)>, i32)>, i32)>
// CHECK:      %[[DELTA:.*]] = llvm.load %[[DELTA_PTR]] : !llvm.ptr -> i32
// CHECK:      %[[CMP:.*]] = comb.icmp slt %[[X1]], %[[DELTA]] : i32
// CHECK:      cf.cond_br %[[CMP]], ^bb1, ^bb2
// CHECK:    ^bb1:
// CHECK:      %[[BIAS_IDX1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:      %[[BIAS_PTR1:.*]] = llvm.getelementptr %[[THIS1]][%[[BIAS_IDX1]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"testClassVirtualInt", (struct<(ptr, ptr)>, i32)>
// CHECK:      %[[BIAS1:.*]] = llvm.load %[[BIAS_PTR1]] : !llvm.ptr -> i32
// CHECK:      %[[SUM1:.*]] = comb.add %[[X1]], %[[BIAS1]] : i32
// CHECK:      llvm.return %[[SUM1]] : i32
// CHECK:    ^bb2:
// CHECK:      %[[DELTA2:.*]] = llvm.load %[[DELTA_PTR]] : !llvm.ptr -> i32
// CHECK:      %[[DIFF:.*]] = comb.sub %[[X1]], %[[DELTA2]] : i32
// CHECK:      llvm.return %[[DIFF]] : i32
// CHECK-NOT:  unrealized_conversion_cast
// CHECK-NOT:  llhd.prb

// CHECK-LABEL: hw.module @top()
// CHECK:      %[[ARG:.*]] = hw.constant 3 : i32
// CHECK:      %[[SIZE:.*]] = llvm.mlir.constant(24 : i64) : i64
// CHECK:      %[[OBJ:.*]] = llvm.call @malloc(%[[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:      %[[TYPEINFO:.*]] = llvm.mlir.addressof @"testClassVirtualInt::typeinfo" : !llvm.ptr
// CHECK:      %[[ZERO_INIT:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:      %[[ZERO_INIT2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:      %[[OBJ_HEADER_INIT:.*]] = llvm.getelementptr %[[OBJ]][%[[ZERO_INIT]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"testClassVirtualInt", (struct<(ptr, ptr)>, i32)>
// CHECK:      %[[TYPEINFO_PTR:.*]] = llvm.getelementptr %[[OBJ_HEADER_INIT]][%[[ZERO_INIT]], 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(ptr, ptr)>
// CHECK:      llvm.store %[[TYPEINFO]], %[[TYPEINFO_PTR]] : !llvm.ptr, !llvm.ptr
// CHECK:      %[[VTABLE:.*]] = llvm.mlir.addressof @"testClassVirtualInt::vtable" : !llvm.ptr
// CHECK:      %[[ONE_INIT:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:      %[[VTABLE_PTR_INIT:.*]] = llvm.getelementptr %[[OBJ_HEADER_INIT]][%[[ZERO_INIT]], 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(ptr, ptr)>
// CHECK:      llvm.store %[[VTABLE]], %[[VTABLE_PTR_INIT]] : !llvm.ptr, !llvm.ptr
// CHECK:      %[[YINIT:.*]] = hw.constant 0 : i32
// CHECK:      %[[Y:.*]] = llhd.sig %[[YINIT]] : i32
// CHECK:      llhd.process {
// CHECK:        %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:        %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:        %[[ZERO2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:        %[[OBJ_HEADER:.*]] = llvm.getelementptr %[[OBJ]][%[[ZERO]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"testClassVirtualInt", (struct<(ptr, ptr)>, i32)>
// CHECK:        %[[VTABLE_PTR_PTR:.*]] = llvm.getelementptr %[[OBJ_HEADER]][%[[ZERO]], 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(ptr, ptr)>
// CHECK:        %[[VTABLE_PTR:.*]] = llvm.load %[[VTABLE_PTR_PTR]] : !llvm.ptr -> !llvm.ptr
// CHECK:        %[[SLOT_PTR:.*]] = llvm.getelementptr %[[VTABLE_PTR]][%[[ZERO]], 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(ptr)>
// CHECK:        %[[CALLEE:.*]] = llvm.load %[[SLOT_PTR]] : !llvm.ptr -> !llvm.ptr
// CHECK:        %[[RESULT:.*]] = llvm.call %[[CALLEE]](%[[OBJ]], %[[ARG]]) : !llvm.ptr, (!llvm.ptr, i32) -> i32
// CHECK:        %[[DELAY:.*]] = llhd.constant_time <0ns, 0d, 1e>
// CHECK:        llhd.drv %[[Y]], %[[RESULT]] after %[[DELAY]] : i32
// CHECK-NOT:  llhd.sig %{{.*}} : !llvm.ptr
// CHECK-NOT:  llhd.prb %{{.*}} : !llvm.ptr

class testClassVirtualInt;
  int bias;
  virtual function int adjust(int x);
    return x + bias;
  endfunction
endclass

class testDerivedVirtualInt extends testClassVirtualInt;
  int delta;
  virtual function int adjust(int x);
    if (x < delta)
      return x + bias;
    return x - delta;
  endfunction
endclass

module top;
  testClassVirtualInt t = new;
  int y;
  initial begin
    y = t.adjust(3);
  end
endmodule
