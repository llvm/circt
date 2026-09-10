// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK:       hw.module @foo(in %0 "" : !dc.token, in %1 "" : !dc.value<i1>, in %2 "" : i32) {
// CHECK-NEXT:    %3 = dc.buffer[2] %0 : !dc.token
// CHECK-NEXT:    %4 = dc.buffer[2] %1 [1, 2] : !dc.value<i1>
// CHECK-NEXT:    %5:2 = dc.fork [2] %0 
// CHECK-NEXT:    %6 = dc.pack %0, %2 : i32
// CHECK-NEXT:    %7 = dc.merge %3, %0
// CHECK-NEXT:    %token, %output = dc.unpack %1 : !dc.value<i1>
// CHECK-NEXT:    %8 = dc.to_esi %0 : !dc.token
// CHECK-NEXT:    %9 = dc.to_esi %1 : !dc.value<i1>
// CHECK-NEXT:    %10 = dc.from_esi %8 : <i0>
// CHECK-NEXT:    %11 = dc.from_esi %9 : <i1>
// CHECK-NEXT:    hw.output
// CHECK-NEXT:  }

hw.module @foo(in %0 : !dc.token, in %1 : !dc.value<i1>, in %2 : i32) {
  %buffer = dc.buffer [2] %0 : !dc.token
  %bufferInit = dc.buffer [2] %1 [1, 2] : !dc.value<i1>
  %f1, %f2 = dc.fork [2] %0
  %pack = dc.pack %0, %2 : i32
  %merge = dc.merge %buffer, %0
  %unpack_token, %unpack_value = dc.unpack %1 : !dc.value<i1>
  %esi_token = dc.to_esi %0 : !dc.token
  %esi_value = dc.to_esi %1 : !dc.value<i1>
  %from_esi_token = dc.from_esi %esi_token : !esi.channel<i0>
  %from_esi_value = dc.from_esi %esi_value : !esi.channel<i1>
}
