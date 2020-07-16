//RUN: circt-opt %s --convert-llhd-to-llvm | FileCheck %s

// CHECK-LABEL: @convert_const
llvm.func @convert_const() {
    // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(true) : !llvm.i1
    %0 = llhd.const 1 : i1

    // CHECK-NEXT %{{.*}} = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1 = llhd.const 0 : i32

    // this gets erased
    %2 = llhd.const #llhd.time<0ns, 0d, 0e> : !llhd.time

    // CHECK-NEXT %{{.*}} = llvm.mlir.constant(123 : i64) : !llvm.i64
    %3 = llhd.const 123 : i64

    llvm.return
}

// CHECK-LABEL: @convert_extract_slice_int
// CHECK-SAME: %[[CI32:.*]]: !llvm.i32
// CHECK-SAME: %[[CI100:.*]]: !llvm.i100
func @convert_extract_slice_int(%cI32 : i32, %cI100 : i100) {
    // CHECK-NEXT: %[[CIND0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
    // CHECK-NEXT: %[[CIND1:.*]] = llvm.mlir.constant(10 : index) : !llvm.i64
    // CHECK-NEXT: %[[ADJUST0:.*]] = llvm.trunc %[[CIND0]] : !llvm.i64 to !llvm.i32
    // CHECK-NEXT: %[[SHR0:.*]] = llvm.lshr %[[CI32]], %[[ADJUST0]] : !llvm.i32
    // CHECK-NEXT: %{{.*}} = llvm.trunc %[[SHR0]] : !llvm.i32 to !llvm.i10
    %0 = llhd.extract_slice %cI32, 0 : i32 -> i10
    // CHECK-NEXT: %[[CIND2:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
    // CHECK-NEXT: %[[CIND3:.*]] = llvm.mlir.constant(10 : index) : !llvm.i64
    // CHECK-NEXT: %[[ADJUST1:.*]] = llvm.zext %[[CIND2]] : !llvm.i64 to !llvm.i100
    // CHECK-NEXT: %[[SHR1:.*]] = llvm.lshr %[[CI100]], %[[ADJUST1]] : !llvm.i100
    // CHECK-NEXT: %{{.*}} = llvm.trunc %[[SHR1]] : !llvm.i100 to !llvm.i10
    %2 = llhd.extract_slice %cI100, 0 : i100 -> i10

    return
}

// CHECK-LABEL: @convert_extract_slice_sig
// CHECK-SAME: %[[STATE:.*]]: !llvm<"i8*">,
// CHECK-SAME: %[[SIGTAB:.*]]: !llvm<"i32*">,
// CHECK-SAME: %[[ARGTAB:.*]]: !llvm<"i32*">
llhd.entity @convert_extract_slice_sig (%sI32 : !llhd.sig<i32>) -> () {
    // CHECK-NEXT: %[[IDX0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[GEP0:.*]] = llvm.getelementptr %[[ARGTAB]][%[[IDX0]]] : (!llvm<"i32*">, !llvm.i32) -> !llvm<"i32*">
    // CHECK-NEXT: %[[L0:.*]] = llvm.load %[[GEP0]] : !llvm<"i32*">
    // CHECK-NEXT: %[[IDX1:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
    // CHECK-NEXT: %[[LEN0:.*]] = llvm.mlir.constant(10 : index) : !llvm.i64
    // CHECK-NEXT: %[[CALL0:.*]] = llvm.call @probe_signal(%[[STATE]], %[[L0]]) : (!llvm<"i8*">, !llvm.i32) -> !llvm<"{ i8*, i64 }*">
    // CHECK-NEXT: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[CALL0]][%[[C0]], %[[C0]]] : (!llvm<"{ i8*, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i8**">
    // CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[CALL0]][%[[C0]], %[[C1]]] : (!llvm<"{ i8*, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    // CHECK-NEXT: %[[L1:.*]] = llvm.load %[[GEP1]] : !llvm<"i8**">
    // CHECK-NEXT: %[[L2:.*]] = llvm.load %[[GEP2]] : !llvm<"i64*">
    // CHECK-NEXT: %[[IDX2:.*]] = llvm.add %[[L2]], %[[IDX1]] : !llvm.i64
    // CHECK-NEXT: %[[PTRTOINT0:.*]] = llvm.ptrtoint %[[L1]] : !llvm<"i8*"> to !llvm.i64
    // CHECK-NEXT: %[[C2:.*]] = llvm.mlir.constant(8 : i64) : !llvm.i64
    // CHECK-NEXT: %[[PTROFFSET0:.*]] = llvm.udiv %[[IDX2]], %[[C2]] : !llvm.i64
    // CHECK-NEXT: %[[ADD0:.*]] = llvm.add %[[PTRTOINT0]], %[[PTROFFSET0]] : !llvm.i64
    // CHECK-NEXT: %[[INTTTOPTR0:.*]] = llvm.inttoptr %[[ADD0]] : !llvm.i64 to !llvm<"i8*">
    // CHECK-NEXT: %[[BYTEOFFSET0:.*]] = llvm.urem %[[IDX2]], %[[C2]] : !llvm.i64
    // CHECK-NEXT: %{{.*}} = llvm.call @add_subsignal(%[[STATE]], %[[L0]], %[[INTTTOPTR0]], %[[LEN0]], %[[BYTEOFFSET0]]) : (!llvm<"i8*">, !llvm.i32, !llvm<"i8*">, !llvm.i64, !llvm.i64) -> !llvm.i32
    %0 = llhd.extract_slice %sI32, 0 : !llhd.sig<i32> -> !llhd.sig<i10>
}
