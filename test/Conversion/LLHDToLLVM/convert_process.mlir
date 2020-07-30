//RUN: circt-opt %s --convert-llhd-to-llvm --split-input-file | FileCheck %s

// CHECK-LABEL: @dummy_i1
func @dummy_i1 (%0 : i1) {
  return
}

// CHECK-LABEL: @dummy_i32
func @dummy_i32 (%0 : i32)  {
  return
}

// CHECK-LABEL: @convert_persistent_value
// CHECK-SAME: %[[STATE:.*]]: !llvm<"i8*">
// CHECK-SAME: %[[PROCSTATE:.*]]: !llvm<"{ i8*, i32, [2 x i1]*, { i1, i32 } }*">
// CHECK-SAME: %[[ARGTABLE:.*]]: !llvm<"{ i8*, i64, i64, i64 }*">
// CHECK-NEXT: %[[GIND1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[GEP0:.*]] = llvm.getelementptr %[[ARGTABLE]][%[[GIND1]]] : (!llvm<"{ i8*, i64, i64, i64 }*">, !llvm.i32) -> !llvm<"{ i8*, i64, i64, i64 }*">
// CHECK-NEXT: %[[GIND2:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ARGTABLE]][%[[GIND2]]] : (!llvm<"{ i8*, i64, i64, i64 }*">, !llvm.i32) -> !llvm<"{ i8*, i64, i64, i64 }*">
// CHECK-NEXT: %[[GIND3:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[GIND4:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[PROCSTATE]][%[[GIND3]], %[[GIND4]]] : (!llvm<"{ i8*, i32, [2 x i1]*, { i1, i32 } }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: %[[L2:.*]] = llvm.load %[[GEP2]] : !llvm<"i32*">
// CHECK-NEXT: llvm.br ^[[BB0:.*]]
// CHECK-NEXT: ^[[BB0]]:
// CHECK-NEXT: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[CMP0:.*]] = llvm.icmp "eq" %[[L2]], %[[C0]] : !llvm.i32
// CHECK-NEXT: llvm.cond_br %[[CMP0]], ^[[BB1:.*]], ^[[BB2:.*]]
// CHECK-NEXT: ^[[BB1]]:
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(false) : !llvm.i1
// CHECK-NEXT: %[[GIND5:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[GIND6:.*]] = llvm.mlir.constant(3 : i32) : !llvm.i32
// CHECK-NEXT: %[[GIND7:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[GEP3:.*]] = llvm.getelementptr %[[PROCSTATE]][%[[GIND5]], %[[GIND6]], %[[GIND7]]] : (!llvm<"{ i8*, i32, [2 x i1]*, { i1, i32 } }*">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"i1*">
// CHECK-NEXT: llvm.store %[[C1]], %[[GEP3]] : !llvm<"i1*">
// CHECK-NEXT: %[[C2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[GIND8:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[GIND9:.*]] = llvm.mlir.constant(3 : i32) : !llvm.i32
// CHECK-NEXT: %[[GIND10:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %[[GEP4:.*]] = llvm.getelementptr %[[PROCSTATE]][%[[GIND8]], %[[GIND9]], %[[GIND10]]] : (!llvm<"{ i8*, i32, [2 x i1]*, { i1, i32 } }*">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: llvm.store %[[C2]], %[[GEP4]] : !llvm<"i32*">
// CHECK-NEXT: llvm.br ^[[BB3:.*]]
// CHECK-NEXT: ^[[BB3]]:
// CHECK-NEXT: %[[GIND11:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[GIND12:.*]] = llvm.mlir.constant(3 : i32) : !llvm.i32
// CHECK-NEXT: %[[GIND13:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %[[GEP5:.*]] = llvm.getelementptr %[[PROCSTATE]][%[[GIND11]], %[[GIND12]], %[[GIND13]]] : (!llvm<"{ i8*, i32, [2 x i1]*, { i1, i32 } }*">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: %[[L3:.*]] = llvm.load %[[GEP5]] : !llvm<"i32*">
// CHECK-NEXT: %[[GIND14:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[GIND15:.*]] = llvm.mlir.constant(3 : i32) : !llvm.i32
// CHECK-NEXT: %[[GIND16:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[GEP6:.*]] = llvm.getelementptr %[[PROCSTATE]][%[[GIND14]], %[[GIND15]], %[[GIND16]]] : (!llvm<"{ i8*, i32, [2 x i1]*, { i1, i32 } }*">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"i1*">
// CHECK-NEXT: %[[L4:.*]] = llvm.load %[[GEP6]] : !llvm<"i1*">
// CHECK-NEXT: llvm.call @dummy_i1(%[[L4]]) : (!llvm.i1) -> ()
// CHECK-NEXT: llvm.call @dummy_i32(%[[L3]]) : (!llvm.i32) -> ()
// CHECK-NEXT: llvm.br ^[[BB3]]
// CHECK-NEXT: ^[[BB2]]:
// CHECK-NEXT: llvm.return
// CHECK-NEXT: }
llhd.proc @convert_persistent_value () -> (%out0 : !llhd.sig<i1>, %out1 : !llhd.sig<i32>) {
  br ^entry
^entry:
  %0 = llhd.const 0 : i1
  %1 = llhd.const 0 : i32
  br ^resume
^resume:
  %t = llhd.const #llhd.time<0ns, 0d, 1e> : !llhd.time
  call @dummy_i1(%0) : (i1) -> ()
  call @dummy_i32(%1) : (i32) -> ()
  br ^resume
}

// CHECK-LABEL: @convert_resume
// CHECK-SAME: %[[STATE:.*]]: !llvm<"i8*">
// CHECK-SAME: %[[PROCSTATE:.*]]: !llvm<"{ i8*, i32, [1 x i1]*, {} }*">
// CHECK-SAME: %[[ARGTABLE:.*]]:  !llvm<"{ i8*, i64, i64, i64 }*">)
// CHECK-NEXT: %[[GIND1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[GEP0:.*]] = llvm.getelementptr %[[ARGTABLE]][%[[GIND1]]] : (!llvm<"{ i8*, i64, i64, i64 }*">, !llvm.i32) -> !llvm<"{ i8*, i64, i64, i64 }*">
// CHECK-NEXT: %[[GIND2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[GIND3:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[PROCSTATE]][%[[GIND2]], %[[GIND3]]] : (!llvm<"{ i8*, i32, [1 x i1]*, {} }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: %[[L1:.*]] = llvm.load %[[GEP1]] : !llvm<"i32*">
// CHECK-NEXT: llvm.br ^[[BB0:.*]]
// CHECK-NEXT: ^[[BB0]]:
// CHECK-NEXT: %[[C0:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %[[CMP0:.*]] = llvm.icmp "eq" %[[L1]], %[[C0]] : !llvm.i32
// CHECK-NEXT: llvm.cond_br %[[CMP0]], ^[[BB1:.*]], ^[[BB2:.*]]
// CHECK-NEXT: ^[[BB2]]:
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[CMP1:.*]] = llvm.icmp "eq" %[[L1]], %[[C1]] : !llvm.i32
// CHECK-NEXT: llvm.cond_br %[[CMP1]], ^[[BB3:.*]], ^[[BB4:.*]]
// CHECK-NEXT: ^[[BB3]]:
// CHECK-NEXT: %[[GIND4:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[GIND5:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %[[GIND6:.*]] = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[PROCSTATE]][%[[GIND4]], %[[GIND6]]] : (!llvm<"{ i8*, i32, [1 x i1]*, {} }*">, !llvm.i32, !llvm.i32) -> !llvm<"[1 x i1]**">
// CHECK-NEXT: %[[L2:.*]] = llvm.load %[[GEP2]] : !llvm<"[1 x i1]**">
// CHECK-NEXT: %[[GIND7:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[C2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i1
// CHECK-NEXT: %[[GEP3:.*]] = llvm.getelementptr %[[L2]][%[[GIND4]], %[[GIND7]]] : (!llvm<"[1 x i1]*">, !llvm.i32, !llvm.i32) -> !llvm<"i1*">
// CHECK-NEXT: llvm.store %[[C2]], %[[GEP3]] : !llvm<"i1*">
// CHECK-NEXT: %[[T0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[T1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[T2:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %[[BC0:.*]] = llvm.bitcast %[[PROCSTATE]] : !llvm<"{ i8*, i32, [1 x i1]*, {} }*"> to !llvm<"i8*">
// CHECK-NEXT: %[[C3:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %[[GEP4:.*]] = llvm.getelementptr %[[PROCSTATE]][%[[GIND4]], %[[GIND5]]] : (!llvm<"{ i8*, i32, [1 x i1]*, {} }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: llvm.store %[[C3]], %[[GEP4]] : !llvm<"i32*">
// CHECK-NEXT: %{{.*}} = llvm.call @llhd_suspend(%[[STATE]], %[[BC0]], %[[T0]], %[[T1]], %[[T2]]) : (!llvm<"i8*">, !llvm<"i8*">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.void
// CHECK-NEXT: llvm.return
// CHECK-NEXT: ^[[BB1]]:
// CHECK-NEXT: %[[GIND8:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[GIND9:.*]] = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %[[GEP5:.*]] = llvm.getelementptr %[[PROCSTATE]][%[[GIND8]], %[[GIND9]]] : (!llvm<"{ i8*, i32, [1 x i1]*, {} }*">, !llvm.i32, !llvm.i32) -> !llvm<"[1 x i1]**">
// CHECK-NEXT: %[[L3:.*]] = llvm.load %[[GEP5]] : !llvm<"[1 x i1]**">
// CHECK-NEXT: %[[GIND10:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[C4:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i1
// CHECK-NEXT: %[[GEP6:.*]] = llvm.getelementptr %[[L3]][%[[GIND8]], %[[GIND10]]] : (!llvm<"[1 x i1]*">, !llvm.i32, !llvm.i32) -> !llvm<"i1*">
// CHECK-NEXT: llvm.store %[[C4]], %[[GEP6]] : !llvm<"i1*">
// CHECK-NEXT: llvm.return
// CHECK-NEXT: ^[[BB4]]:
// CHECK-NEXT: llvm.return
// CHECK-NEXT: }
llhd.proc @convert_resume (%in0 : !llhd.sig<i1>) -> () {
  br ^entry
^entry:
  %t = llhd.const #llhd.time<0ns, 0d, 1e> : !llhd.time
  llhd.wait for %t, ^resume
^resume:
  llhd.halt
}
