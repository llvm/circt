// RUN: circt-opt %s -arc-insert-runtime -lower-arc-to-llvm -canonicalize | FileCheck %s

func.func @bar(%s: !arc.storage<17>, %v: i130) {
  %s1 = arc.storage.get %s[4] : !arc.storage<17> -> !arc.state<i130>
  arc.state_write %s1 = %v tap @foo[1] : <i130>
  return
}

arc.model @foo io !hw.modty<> traceTaps [#arc.trace_tap<i32, 0, ["sig32"]>, #arc.trace_tap<i130, 4, ["sig130"]>] {
  ^bb0(%arg0: !arc.storage<17>):
  %cst_i32 = hw.constant 123 : i32
  %cst_i130 = hw.constant -1 : i130
  %s0 = arc.storage.get %arg0[0] : !arc.storage<17> -> !arc.state<i32>
  func.call @bar(%arg0, %cst_i130): (!arc.storage<17>, i130) -> ()
  arc.state_write %s0 = %cst_i32 tap @foo[0] : <i32>
}

// CHECK-LABEL: llvm.func private @_arc_trace_instrument_i192(%arg0: !llvm.ptr, %arg1: i64, %arg2: i192)
// CHECK-DAG:      %[[CST128:.+]] = llvm.mlir.constant(128 : i192) : i192
// CHECK-DAG:      %[[CST64:.+]] = llvm.mlir.constant(64 : i192) : i192
// CHECK-DAG:      %[[CST4:.+]] = llvm.mlir.constant(4 : i32) : i32
// CHECK-DAG:      %[[CSTCAP:.+]] = llvm.mlir.constant({{.+}} : i32) : i32
// CHECK-NEXT:     %4 = llvm.getelementptr %arg0[-24] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT:     %5 = llvm.getelementptr %arg0[-16] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT:     %6 = llvm.load %4 : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:     %7 = llvm.load %5 : !llvm.ptr -> i32
// CHECK-NEXT:     %8 = llvm.add %7, %[[CST4]] : i32
// CHECK-NEXT:     %9 = llvm.getelementptr %6[%7] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:     %10 = llvm.icmp "ugt" %8, %[[CSTCAP]] : i32
// CHECK-NEXT:     llvm.cond_br %10 weights([0, 2147483647]), ^bb1, ^bb2(%9, %8 : !llvm.ptr, i32)
// CHECK-NEXT:   ^bb1:  // pred: ^bb0
// CHECK-NEXT:     %11 = llvm.call @arcRuntimeIR_swapTraceBuffer(%arg0) : (!llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:     llvm.store %11, %4 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:     llvm.br ^bb2(%11, %[[CST4]] : !llvm.ptr, i32)
// CHECK-NEXT:   ^bb2(%12: !llvm.ptr, %13: i32):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:     llvm.store %arg1, %12 : i64, !llvm.ptr
// CHECK-NEXT:     %14 = llvm.getelementptr %12[1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:     %15 = llvm.trunc %arg2 : i192 to i64
// CHECK-NEXT:     llvm.store %15, %14 : i64, !llvm.ptr
// CHECK-NEXT:     %16 = llvm.getelementptr %12[2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:     %17 = llvm.lshr %arg2, %[[CST64]] : i192
// CHECK-NEXT:     %18 = llvm.trunc %17 : i192 to i64
// CHECK-NEXT:     llvm.store %18, %16 : i64, !llvm.ptr
// CHECK-NEXT:     %19 = llvm.getelementptr %12[3] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:     %20 = llvm.lshr %arg2, %[[CST128]] : i192
// CHECK-NEXT:     %21 = llvm.trunc %20 : i192 to i64
// CHECK-NEXT:     llvm.store %21, %19 : i64, !llvm.ptr
// CHECK-NEXT:     llvm.store %13, %5 : i32, !llvm.ptr
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }


// CHECK-LABEL: llvm.func private @_arc_trace_instrument_i64(%arg0: !llvm.ptr, %arg1: i64, %arg2: i64) attributes {no_inline} {
// CHECK-DAG:     %[[CST2:.+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-DAG:     %[[CSTCAP:.+]] = llvm.mlir.constant({{.+}} : i32) : i32
// CHECK-NEXT:    %2 = llvm.getelementptr %arg0[-24] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT:    %3 = llvm.getelementptr %arg0[-16] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT:    %4 = llvm.load %2 : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %5 = llvm.load %3 : !llvm.ptr -> i32
// CHECK-NEXT:    %6 = llvm.add %5, %[[CST2]] : i32
// CHECK-NEXT:    %7 = llvm.getelementptr %4[%5] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    %8 = llvm.icmp "ugt" %6, %[[CSTCAP]] : i32
// CHECK-NEXT:    llvm.cond_br %8 weights([0, 2147483647]), ^bb1, ^bb2(%7, %6 : !llvm.ptr, i32)
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    %9 = llvm.call @arcRuntimeIR_swapTraceBuffer(%arg0) : (!llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:    llvm.store %9, %2 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb2(%9, %[[CST2]] : !llvm.ptr, i32)
// CHECK-NEXT:  ^bb2(%10: !llvm.ptr, %11: i32):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:    llvm.store %arg1, %10 : i64, !llvm.ptr
// CHECK-NEXT:    %12 = llvm.getelementptr %10[1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %arg2, %12 : i64, !llvm.ptr
// CHECK-NEXT:    llvm.store %11, %3 : i32, !llvm.ptr
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }

// CHECK-LABEL: llvm.func @bar(%arg0: !llvm.ptr, %arg1: i130) {
// CHECK-NEXT:    %[[CST1:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %1 = llvm.getelementptr %arg0[4] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT:    %2 = llvm.load %1 : !llvm.ptr -> i130
// CHECK-NEXT:    %3 = llvm.icmp "ne" %arg1, %2 : i130
// CHECK-NEXT:    llvm.cond_br %3, ^bb1, ^bb2
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    llvm.store %arg1, %1 : i130, !llvm.ptr
// CHECK-NEXT:    %4 = llvm.getelementptr %1[-4] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT:    %5 = llvm.zext %arg1 : i130 to i192
// CHECK-NEXT:    llvm.call @_arc_trace_instrument_i192(%4, %[[CST1]], %5) : (!llvm.ptr, i64, i192) -> ()
// CHECK-NEXT:    llvm.br ^bb2
// CHECK-NEXT:  ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }

// CHECK-LABEL: llvm.func @foo_eval(%arg0: !llvm.ptr) {
// CHECK-DAG:     %[[CST123_64:.+]] = llvm.mlir.constant(123 : i64) : i64
// CHECK-DAG:     %[[CST0:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:     %[[CST123_32:.+]] = llvm.mlir.constant(123 : i32) : i32
// CHECK-DAG:     %[[CSTM1_130:.+]] = llvm.mlir.constant(-1 : i130) : i130
// CHECK-NEXT:    llvm.call @bar(%arg0, %[[CSTM1_130]]) : (!llvm.ptr, i130) -> ()
// CHECK-NEXT:    %4 = llvm.load %arg0 : !llvm.ptr -> i32
// CHECK-NEXT:    %5 = llvm.icmp "ne" %[[CST123_32]], %4 : i32
// CHECK-NEXT:    llvm.cond_br %5, ^bb1, ^bb2
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    llvm.store %2, %arg0 : i32, !llvm.ptr
// CHECK-NEXT:    llvm.call @_arc_trace_instrument_i64(%arg0, %[[CST0]], %[[CST123_64]]) : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:    llvm.br ^bb2
// CHECK-NEXT:  ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
