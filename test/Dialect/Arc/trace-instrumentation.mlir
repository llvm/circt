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

// CHECK-DAG: llvm.func private @_arc_trace_instrument_i192(%arg0: !llvm.ptr, %arg1: i64, %arg2: i192)
// CHECK-DAG: llvm.func private @_arc_trace_instrument_i64(%arg0: !llvm.ptr, %arg1: i64, %arg2: i64)

// CHECK-LABEL: llvm.func @bar(%arg0: !llvm.ptr, %arg1: i130)
// CHECK-NOT  : llvm.store
// CHECK      : %[[OLDVAL:.+]] = llvm.load %[[STATE:.+]] : !llvm.ptr -> i130
// CHECK-NEXT : %[[CMP:.+]] = llvm.icmp "ne" %arg1, %[[OLDVAL]] : i130
// CHECK-NEXT : llvm.cond_br %[[CMP]], ^[[BB1:.+]], ^[[BB2:.+]]
// CHECK-NEXT : ^[[BB1]]:
// CHECK-DAG  : llvm.store %arg1, [[STATE]] : i130, !llvm.ptr
// CHECK-DAG  : llvm.call @_arc_trace_instrument_i192
// CHECK      : ^[[BB2]]:

// CHECK-LABEL: llvm.func @foo_eval(%arg0: !llvm.ptr)
// CHECK-NOT  : llvm.store
// CHECK      : %[[OLDVAL:.+]] = llvm.load %[[STATE:.+]] : !llvm.ptr -> i32
// CHECK-NEXT : %[[CMP:.+]] = llvm.icmp "ne" %{{.+}}, %[[OLDVAL]] : i130
// CHECK-NEXT : llvm.cond_br %[[CMP]], ^[[BB1:.+]], ^[[BB2:.+]]
// CHECK-NEXT : ^[[BB1]]:
// CHECK-DAG  : llvm.store %arg1, [[STATE]] : i32, !llvm.ptr
// CHECK-DAG  : llvm.call @_arc_trace_instrument_i64
// CHECK      : ^[[BB2]]:
