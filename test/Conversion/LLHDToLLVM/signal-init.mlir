// RUN: circt-opt %s --convert-llhd-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @llhd_init
// CHECK: llvm.mlir.constant(false) : i1
llhd.entity @root() -> () {
  llhd.inst "inst1" @initArraySig () -> () : () -> ()
  llhd.inst "inst2" @initStructSig () -> () : () -> ()
  llhd.inst "inst3" @initPartiallyLowered () -> () : () -> ()
  llhd.inst "inst4" @initMultipleResults () -> () : () -> ()
}

// CHECK: [[ZERO:%.+]] = llvm.mlir.constant(false) : i1
// CHECK: [[A1:%.+]] = llvm.mlir.undef : !llvm.array<2 x i1>
// CHECK: [[A2:%.+]] = llvm.insertvalue [[ZERO]], [[A1]][0] : !llvm.array<2 x i1>
// CHECK: [[A3:%.+]] = llvm.insertvalue [[ZERO]], [[A2]][1] : !llvm.array<2 x i1>
// CHECK: llvm.store [[A3]], {{%.+}} : !llvm.ptr<array<2 x i1>>
llhd.entity @initArraySig () -> () {
  %init = hw.constant 0 : i1
  %initArr = hw.array_create %init, %init : i1
  %0 = llhd.sig "sig" %initArr : !hw.array<2xi1>
}

// CHECK: llvm.mlir.constant(false) : i1

// CHECK: [[FALSE:%.+]] = llvm.mlir.constant(false) : i1
// CHECK: [[THREE:%.+]] = llvm.mlir.constant(3 : i5) : i5
// CHECK: [[S1:%.+]] = llvm.mlir.undef : !llvm.struct<(i5, i1)>
// CHECK: [[S2:%.+]] = llvm.insertvalue [[THREE]], [[S1]][0] : !llvm.struct<(i5, i1)>
// CHECK: [[S3:%.+]] = llvm.insertvalue [[FALSE]], [[S2]][1] : !llvm.struct<(i5, i1)>
// CHECK: llvm.store [[S3]], {{%.+}} : !llvm.ptr<struct<(i5, i1)>>
llhd.entity @initStructSig () -> () {
  %init = hw.constant 0 : i1
  %init1 = hw.constant 3 : i5
  %initStruct = hw.struct_create (%init, %init1) : !hw.struct<f1: i1, f2: i5>
  %0 = llhd.sig "sig" %initStruct : !hw.struct<f1: i1, f2: i5>
}

// CHECK: [[B1:%.+]] = llvm.mlir.undef : !llvm.array<2 x i1>
// CHECK: [[FALSE1:%.+]] = llvm.mlir.constant(false) : i1
// CHECK: [[B2:%.+]] = llvm.insertvalue [[FALSE1]], [[B1]][0] : !llvm.array<2 x i1>
// CHECK: [[B3:%.+]] = llvm.insertvalue [[FALSE1]], [[B2]][1] : !llvm.array<2 x i1>
// CHECK: llvm.store [[B3]], {{%.+}} : !llvm.ptr<array<2 x i1>>
llhd.entity @initPartiallyLowered () -> () {
  %0 = llvm.mlir.constant(false) : i1
  %1 = llvm.mlir.undef : !llvm.array<2 x i1>
  %2 = llvm.insertvalue %0, %1[0] : !llvm.array<2 x i1>
  %3 = llvm.insertvalue %0, %2[1] : !llvm.array<2 x i1>
  %4 = builtin.unrealized_conversion_cast %3 : !llvm.array<2 x i1> to !hw.array<2xi1>
  %5 = llhd.sig "sig" %4 : !hw.array<2xi1>
}

func.func @getInitValue() -> (i32, i32, i32) {
  %0 = hw.constant 0 : i32
  return %0, %0, %0 : i32, i32, i32
}

// CHECK: [[RETURN:%.+]] = llvm.call @getInitValue() : () -> !llvm.struct<(i32, i32, i32)>
// CHECK: [[E1:%.+]] = llvm.extractvalue [[RETURN]][0] : !llvm.struct<(i32, i32, i32)>
// CHECK: [[E2:%.+]] = llvm.extractvalue [[RETURN]][1] : !llvm.struct<(i32, i32, i32)>
// CHECK: [[E3:%.+]] = llvm.extractvalue [[RETURN]][2] : !llvm.struct<(i32, i32, i32)>
// CHECK: llvm.store [[E2]], {{%.+}} : !llvm.ptr<i32>
llhd.entity @initMultipleResults () -> () {
  %0, %1, %2 = func.call @getInitValue() : () -> (i32, i32, i32)
  %3 = llhd.sig "sig" %1 : i32
}

// CHECK: }
