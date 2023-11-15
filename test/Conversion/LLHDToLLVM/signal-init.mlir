// RUN: circt-opt %s --convert-llhd-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @llhd_init
// CHECK-SAME:    %arg0: !llvm.ptr) {
llhd.entity @Root() -> () {
  // CHECK: [[SIZE:%.+]] = llvm.ptrtoint
  // CHECK: [[MEM:%.+]] = llvm.call @malloc([[SIZE]])
  // CHECK: llvm.call @allocEntity(%arg0, [[OWNER:%.+]], [[MEM]])

  // sig0
  // CHECK: [[VALUE:%.+]] = llvm.mlir.constant(1337 : i42) : i42
  // CHECK: llvm.store [[VALUE]], [[BUF:%.+]] : i42, !llvm.ptr
  // CHECK: [[SIZE:%.+]] = llvm.mlir.constant(6 :
  // CHECK: llvm.call @allocSignal(%arg0, {{%.+}}, [[OWNER]], [[BUF]], [[SIZE]])

  // sig1
  // CHECK: [[TMP:%.+]] = llvm.getelementptr {{%.+}}[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i1>
  // CHECK: [[SIZE:%.+]] = llvm.ptrtoint [[TMP]] : !llvm.ptr to i64
  // CHECK: llvm.store {{%.+}}, [[BUF:%.+]] : !llvm.array<2 x i1>, !llvm.ptr
  // CHECK: [[SIGID1:%.+]] = llvm.call @allocSignal(%arg0, {{%.+}}, [[OWNER]], [[BUF]], [[SIZE]])
  // sig1 layout details
  // CHECK: [[ELEMENT_COUNT:%.+]] = llvm.mlir.constant(2 :
  // CHECK: [[TMP:%.+]] = llvm.getelementptr {{%.+}}[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i1>
  // CHECK: [[ELEMENT_SIZE:%.+]] = llvm.ptrtoint [[TMP]]
  // CHECK: llvm.call @addSigArrayElements(%arg0, [[SIGID1]], [[ELEMENT_SIZE]], [[ELEMENT_COUNT]])

  // sig2
  // CHECK: [[TMP:%.+]] = llvm.getelementptr {{%.+}}[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i5, i1)>
  // CHECK: [[SIZE:%.+]] = llvm.ptrtoint [[TMP]] : !llvm.ptr to i64
  // CHECK: llvm.store {{%.+}}, [[BUF:%.+]] : !llvm.struct<(i5, i1)>, !llvm.ptr
  // CHECK: [[SIGID2:%.+]] = llvm.call @allocSignal(%arg0, {{%.+}}, [[OWNER]], [[BUF]], [[SIZE]])
  // sig2 layout details f2
  // CHECK: [[TMP:%.+]] = llvm.getelementptr {{%.+}}[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i5, i1)>
  // CHECK: [[ELEMENT_OFFSET:%.+]] = llvm.ptrtoint [[TMP]]
  // CHECK: [[TMP:%.+]] = llvm.getelementptr {{%.+}}[1] : (!llvm.ptr) -> !llvm.ptr, i5
  // CHECK: [[ELEMENT_SIZE:%.+]] = llvm.ptrtoint [[TMP]]
  // CHECK: llvm.call @addSigStructElement(%arg0, [[SIGID2]], [[ELEMENT_OFFSET]], [[ELEMENT_SIZE]])
  // sig2 layout details f1
  // CHECK: [[TMP:%.+]] = llvm.getelementptr {{%.+}}[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i5, i1)>
  // CHECK: [[ELEMENT_OFFSET:%.+]] = llvm.ptrtoint [[TMP]]
  // CHECK: [[TMP:%.+]] = llvm.getelementptr {{%.+}}[1] : (!llvm.ptr) -> !llvm.ptr, i1
  // CHECK: [[ELEMENT_SIZE:%.+]] = llvm.ptrtoint [[TMP]]
  // CHECK: llvm.call @addSigStructElement(%arg0, [[SIGID2]], [[ELEMENT_OFFSET]], [[ELEMENT_SIZE]])

  llhd.inst "inst0" @Signals () -> () : () -> ()

  // CHECK: llvm.call @allocEntity(%arg0, [[OWNER:%.+]], {{%.+}})

  // sig3
  // CHECK-DAG: [[FALSE:%.+]] = llvm.mlir.constant(false)
  // CHECK-DAG: [[TMP1:%.+]] = llvm.mlir.undef
  // CHECK-DAG: [[TMP2:%.+]] = llvm.insertvalue [[FALSE]], [[TMP1]][0]
  // CHECK-DAG: [[TMP3:%.+]] = llvm.insertvalue [[FALSE]], [[TMP2]][1]
  // CHECK: llvm.store [[TMP3]], [[BUF:%.+]] : !llvm.array<2 x i1>, !llvm.ptr
  // CHECK: [[SIGID3:%.+]] = llvm.call @allocSignal(%arg0, {{%.+}}, [[OWNER]], [[BUF]], {{%.+}})

  llhd.inst "inst1" @PartiallyLowered () -> () : () -> ()
  // llhd.inst "inst2" @MultipleResults () -> () : () -> ()
}

llhd.entity @Signals () -> () {
  %0 = hw.constant 1337 : i42
  %1 = llhd.sig "sig0" %0 : i42
  %2 = hw.aggregate_constant [0 : i1, 1 : i1] : !hw.array<2xi1>
  %3 = llhd.sig "sig1" %2 : !hw.array<2xi1>
  %4 = hw.aggregate_constant [0 : i1, 1 : i5] : !hw.struct<f1: i1, f2: i5>
  %5 = llhd.sig "sig2" %4 : !hw.struct<f1: i1, f2: i5>
}

llhd.entity @PartiallyLowered () -> () {
  %0 = llvm.mlir.constant(false) : i1
  %1 = llvm.mlir.undef : !llvm.array<2 x i1>
  %2 = llvm.insertvalue %0, %1[0] : !llvm.array<2 x i1>
  %3 = llvm.insertvalue %0, %2[1] : !llvm.array<2 x i1>
  %4 = builtin.unrealized_conversion_cast %3 : !llvm.array<2 x i1> to !hw.array<2xi1>
  %5 = llhd.sig "sig3" %4 : !hw.array<2xi1>
}
