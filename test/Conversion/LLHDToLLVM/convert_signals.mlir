//RUN: circt-opt %s --convert-llhd-to-llvm | FileCheck %s

// CHECK-LABEL: @convert_sig
// CHECK-SAME: %{{.*}}: !llvm<"i8*">,
// CHECK-SAME: %[[SIGTAB:.*]]: !llvm<"i32*">,
// CHECK-SAME: %{{.*}}: !llvm<"i32*">
llhd.entity @convert_sig () -> () {
    // CHECK-NEXT: %[[C0:.*]] = llvm.mlir.constant(false) : !llvm.i1
    %init = llhd.const 0 : i1
    // CHECK-NEXT: %[[IDX0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[GEP0:.*]] = llvm.getelementptr %[[SIGTAB]][%[[IDX0]]] : (!llvm<"i32*">, !llvm.i32) -> !llvm<"i32*">
    // CHECK-NEXT: %[[L0:.*]] = llvm.load %[[GEP0]] : !llvm<"i32*">
    %s0 = llhd.sig "sig0" %init : i1
}

// CHECK-LABEL: @convert_prb
// CHECK-SAME: %[[STATE:.*]]: !llvm<"i8*">,
// CHECK-SAME:{{.*}}: !llvm<"i32*">,
// CHECK-SAME: %[[ARGTAB:.*]]: !llvm<"i32*">
llhd.entity @convert_prb (%sI1 : !llhd.sig<i1>) -> () {
    // CHECK-NEXT: %[[IDX0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[GEP0:.*]] = llvm.getelementptr %[[ARGTAB]][%[[IDX0]]] : (!llvm<"i32*">, !llvm.i32) -> !llvm<"i32*">
    // CHECK-NEXT: %[[L0:.*]] = llvm.load %[[GEP0]] : !llvm<"i32*">
    // CHECK-NEXT: %[[CALL0:.*]] = llvm.call @probe_signal(%[[STATE]], %[[L0]]) : (!llvm<"i8*">, !llvm.i32) -> !llvm<"{ i8*, i64 }*">
    // CHECK-NEXT: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[CALL0]][%[[C0]], %[[C0]]] : (!llvm<"{ i8*, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i8**">
    // CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[CALL0]][%[[C0]], %[[C1]]] : (!llvm<"{ i8*, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    // CHECK-NEXT: %[[L1:.*]] = llvm.load %[[GEP1]] : !llvm<"i8**">
    // CHECK-NEXT: %[[L2:.*]] = llvm.load %[[GEP2]] : !llvm<"i64*">
    // CHECK-NEXT: %[[BC0:.*]] = llvm.bitcast %[[L1]] : !llvm<"i8*"> to !llvm<"i16*">
    // CHECK-NEXT: %[[L3:.*]] = llvm.load %[[BC0]] : !llvm<"i16*">
    // CHECK-NEXT: %[[TRUNC0:.*]] = llvm.trunc %[[L2]] : !llvm.i64 to !llvm.i16
    // CHECK-NEXT: %[[SHR:.*]] = llvm.lshr %[[L3]], %[[TRUNC0]] : !llvm.i16
    // CHECK-NEXT: %{{.*}} = llvm.trunc %[[SHR]] : !llvm.i16 to !llvm.i1
    %p0 = llhd.prb %sI1 : !llhd.sig<i1>
}

// CHECK-LABEL: @convert_drv
// CHECK-SAME: %[[STATE:.*]]: !llvm<"i8*">,
// CHECK-SAME:{{.*}}: !llvm<"i32*">,
// CHECK-SAME: %[[ARGTAB:.*]]: !llvm<"i32*">
llhd.entity @convert_drv (%sI1 : !llhd.sig<i1>) -> () {
    // CHECK-NEXT: %[[IDX0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[GEP0:.*]] = llvm.getelementptr %[[ARGTAB]][%[[IDX0]]] : (!llvm<"i32*">, !llvm.i32) -> !llvm<"i32*">
    // CHECK-NEXT: %[[L0:.*]] = llvm.load %[[GEP0]] : !llvm<"i32*">
    // CHECK-NEXT: %[[C0:.*]] = llvm.mlir.constant(false) : !llvm.i1
    %cI1 = llhd.const 0 : i1
    // CHECK-NEXT: %[[WIDTH0:.*]] = llvm.mlir.constant(1 : i64) : !llvm.i64
    // CHECK-NEXT: %[[S0:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %[[A0:.*]] = llvm.alloca %[[S0]] x !llvm.i1 {alignment = 4 : i64} : (!llvm.i32) -> !llvm<"i1*">
    // CHECK-NEXT: llvm.store %[[C0]], %[[A0]] : !llvm<"i1*">
    // CHECK-NEXT: %[[BC0:.*]] = llvm.bitcast %[[A0]] : !llvm<"i1*"> to !llvm<"i8*">
    // CHECK-NEXT: %[[T0:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %[[T1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[T2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = llvm.call @drive_signal(%[[STATE]], %[[L0]], %[[BC0]], %[[WIDTH0]], %[[T0]], %[[T1]], %[[T2]]) : (!llvm<"i8*">, !llvm.i32, !llvm<"i8*">, !llvm.i64, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.void
    %t = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    llhd.drv %sI1, %cI1 after %t : !llhd.sig<i1>
}

