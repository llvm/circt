// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func private @memory_tick(
// CHECK-SAME: !llvm.ptr
// CHECK-SAME: !llvm.ptr
// CHECK-SAME: !llvm.ptr
// CHECK-SAME: !llhd.ref<i1>
func.func private @memory_tick(!moore.chandle, !moore.open_uarray<i8>, !moore.ref<open_uarray<i8>>, !moore.ref<i1>)

// CHECK-LABEL: func.func @call_memory_tick(
// CHECK-SAME: %[[CHANNEL:.*]]: !llvm.ptr, %[[RDATA:.*]]: !llhd.ref<!hw.array<8xi8>>, %[[WDATA:.*]]: !hw.array<8xi8>, %[[READY:.*]]: !llhd.ref<i1>
func.func @call_memory_tick(%channel: !moore.chandle, %r_data: !moore.ref<uarray<8 x i8>>, %w_data: !moore.uarray<8 x i8>, %ready: !moore.ref<i1>) {
  // CHECK: %[[WCAST:.*]] = builtin.unrealized_conversion_cast %[[WDATA]] : !hw.array<8xi8> to !llvm.ptr
  %0 = moore.open_uarray_from_uarray %w_data : !moore.uarray<8 x i8> -> !moore.open_uarray<i8>
  // CHECK: %[[RCAST:.*]] = builtin.unrealized_conversion_cast %[[RDATA]] : !llhd.ref<!hw.array<8xi8>> to !llvm.ptr
  %1 = moore.open_uarray_from_uarray_ref %r_data : !moore.ref<uarray<8 x i8>> -> !moore.ref<open_uarray<i8>>
  // CHECK: call @memory_tick(%[[CHANNEL]], %[[WCAST]], %[[RCAST]], %[[READY]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llhd.ref<i1>) -> ()
  call @memory_tick(%channel, %0, %1, %ready) : (!moore.chandle, !moore.open_uarray<i8>, !moore.ref<open_uarray<i8>>, !moore.ref<i1>) -> ()
  return
}

// CHECK-LABEL: func.func private @packed_dpi(
// CHECK-SAME: !llvm.ptr
// CHECK-SAME: !llvm.ptr
func.func private @packed_dpi(!moore.open_array<i8>, !moore.ref<open_array<i8>>)

// CHECK-LABEL: func.func @call_packed_dpi(
// CHECK-SAME: %[[WDATA:.*]]: !hw.array<8xi8>, %[[RDATA:.*]]: !llhd.ref<!hw.array<8xi8>>
func.func @call_packed_dpi(%w_data: !moore.array<8 x i8>, %r_data: !moore.ref<array<8 x i8>>) {
  // CHECK: %[[WCAST:.*]] = builtin.unrealized_conversion_cast %[[WDATA]] : !hw.array<8xi8> to !llvm.ptr
  %0 = moore.open_array_from_array %w_data : !moore.array<8 x i8> -> !moore.open_array<i8>
  // CHECK: %[[RCAST:.*]] = builtin.unrealized_conversion_cast %[[RDATA]] : !llhd.ref<!hw.array<8xi8>> to !llvm.ptr
  %1 = moore.open_array_from_array_ref %r_data : !moore.ref<array<8 x i8>> -> !moore.ref<open_array<i8>>
  // CHECK: call @packed_dpi(%[[WCAST]], %[[RCAST]]) : (!llvm.ptr, !llvm.ptr) -> ()
  call @packed_dpi(%0, %1) : (!moore.open_array<i8>, !moore.ref<open_array<i8>>) -> ()
  return
}
