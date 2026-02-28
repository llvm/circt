// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s --verify-esi-connections

// CHECK-LABEL: hw.module @SenderVO(out x : !esi.channel<i1, ValidOnly>) {
hw.module @SenderVO(out x: !esi.channel<i1, ValidOnly>) {
  %0 = arith.constant 0 : i1
  // Don't transmit any data.
  %ch = esi.wrap.vo %0, %0 : i1
  hw.output %ch : !esi.channel<i1, ValidOnly>
}

// CHECK-LABEL: hw.module @RecieverVO(in %a : !esi.channel<i1, ValidOnly>) {
hw.module @RecieverVO(in %a: !esi.channel<i1, ValidOnly>) {
  // Receive bits -- no ready signal needed.
  %data, %valid = esi.unwrap.vo %a : !esi.channel<i1, ValidOnly>
}

// CHECK-LABEL: hw.module @StructSendVO(out x : !esi.channel<!hw.struct<a: si4, b: !hw.array<3xui4>>, ValidOnly>) {
!FooStruct = !hw.struct<a: si4, b: !hw.array<3 x ui4>>
hw.module @StructSendVO(out x: !esi.channel<!FooStruct, ValidOnly>) {
  %v = arith.constant 1 : i1
  %cst = hw.aggregate_constant [0 : si4, [0 : ui4, 0 : ui4, 0 : ui4]] : !FooStruct
  %ch = esi.wrap.vo %cst, %v : !FooStruct
  hw.output %ch : !esi.channel<!FooStruct, ValidOnly>
}

// CHECK-LABEL: hw.module @SnoopXactVO
hw.module @SnoopXactVO(in %a: !esi.channel<i32, ValidOnly>) {
  // SnoopTransactionOp should work with ValidOnly channels.
  %xact, %data = esi.snoop.xact %a : !esi.channel<i32, ValidOnly>
}

// CHECK-LABEL: hw.module @TestBuffer
hw.module @TestBuffer(in %clk: !seq.clock, in %rst: i1, in %a: !esi.channel<i1, ValidOnly>) {
  %buffered = esi.buffer %clk, %rst, %a : !esi.channel<i1, ValidOnly> -> !esi.channel<i1, ValidOnly>
}
