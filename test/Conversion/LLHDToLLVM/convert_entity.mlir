//RUN: circt-opt %s --convert-llhd-to-llvm --split-input-file | FileCheck %s

// CHECK: llvm.func @convert_empty(%{{.*}}:  !llvm<"i8*">, %{{.*}}: !llvm<"i32*">, %{{.*}}: !llvm<"i32*">) {
// CHECK-NEXT: llvm.return
// CHECK-NEXT: }
llhd.entity @convert_empty () -> () {}

// CHECK: llvm.func @convert_one_input(%{{.*}}:  !llvm<"i8*">, %{{.*}}: !llvm<"i32*">, %[[ARGTABLE:.*]]: !llvm<"i32*">) {
// CHECK-NEXT: %[[IDX0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[GEP0:.*]] = llvm.getelementptr %[[ARGTABLE]][%[[IDX0]]] : (!llvm<"i32*">, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: %{{.*}} = llvm.load %[[GEP0]] : !llvm<"i32*">
// CHECK-NEXT: llvm.return
// CHECK-NEXT: }
llhd.entity @convert_one_input (%in0 : !llhd.sig<i1>) -> () {}

// CHECK: llvm.func @convert_one_output(%{{.*}}:  !llvm<"i8*">, %{{.*}}: !llvm<"i32*">, %[[ARGTABLE:.*]]: !llvm<"i32*">) {
// CHECK-NEXT: %[[IDX0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[GEP0:.*]] = llvm.getelementptr %[[ARGTABLE]][%[[IDX0]]] : (!llvm<"i32*">, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: %{{.*}} = llvm.load %[[GEP0]] : !llvm<"i32*">
// CHECK-NEXT: llvm.return
// CHECK-NEXT: }
llhd.entity @convert_one_output () -> (%out0 : !llhd.sig<i1>) {}

// CHECK: llvm.func @convert_input_and_output(%{{.*}}:  !llvm<"i8*">, %{{.*}}: !llvm<"i32*">, %[[ARGTABLE:.*]]: !llvm<"i32*">) {
// CHECK-NEXT: %[[IDX0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[GEP0:.*]] = llvm.getelementptr %[[ARGTABLE]][%[[IDX0]]] : (!llvm<"i32*">, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: %{{.*}} = llvm.load %[[GEP0]] : !llvm<"i32*">
// CHECK-NEXT: %[[IDX1:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ARGTABLE]][%[[IDX1]]] : (!llvm<"i32*">, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: %{{.*}} = llvm.load %[[GEP1]] : !llvm<"i32*">
// CHECK-NEXT: llvm.return
// CHECK-NEXT: }
llhd.entity @convert_input_and_output (%in0 : !llhd.sig<i1>) -> (%out0 : !llhd.sig<i1>) {}