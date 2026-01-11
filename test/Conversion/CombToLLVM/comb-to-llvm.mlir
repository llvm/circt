// RUN: circt-opt --convert-to-llvm %s | FileCheck %s

// Test that Comb operations are converted to LLVM via the comprehensive conversion pipeline
// This includes Comb->Arith->LLVM, Func->LLVM, and other dialect conversions

// CHECK-LABEL: llvm.func @test_comb_extract
func.func @test_comb_extract(%arg0: i16) -> i2 {
  // Extract from bit 0 should optimize to just truncation
  // CHECK: %{{.*}} = llvm.trunc %arg0 : i16 to i2
  // CHECK: llvm.return
  %0 = comb.extract %arg0 from 0 : (i16) -> i2
  return %0 : i2
}

// CHECK-LABEL: llvm.func @test_comb_extract_nonzero
func.func @test_comb_extract_nonzero(%arg0: i16) -> i3 {
  // Extract from non-zero bit should use shift + truncate
  // CHECK: %[[C5:.*]] = llvm.mlir.constant(5 : i16) : i16
  // CHECK: %[[SHIFT:.*]] = llvm.lshr %arg0, %[[C5]] : i16
  // CHECK: %[[RESULT:.*]] = llvm.trunc %[[SHIFT]] : i16 to i3
  // CHECK: llvm.return %[[RESULT]]
  %0 = comb.extract %arg0 from 5 : (i16) -> i3
  return %0 : i3
}

// CHECK-LABEL: llvm.func @test_comb_concat
func.func @test_comb_concat(%arg0: i8, %arg1: i8) -> i16 {
  // Concat should use zero-extend, shift, and or
  // CHECK: %[[EXT0:.*]] = llvm.zext %arg1 : i8 to i16
  // CHECK: %[[C8:.*]] = llvm.mlir.constant(8 : i16) : i16
  // CHECK: %[[EXT1:.*]] = llvm.zext %arg0 : i8 to i16
  // CHECK: %[[SHIFT:.*]] = llvm.shl %[[EXT1]], %[[C8]] : i16
  // CHECK: %[[RESULT:.*]] = llvm.or %[[EXT0]], %[[SHIFT]] : i16
  // CHECK: llvm.return %[[RESULT]]
  %0 = comb.concat %arg0, %arg1 : i8, i8
  return %0 : i16
}

// CHECK-LABEL: llvm.func @test_comb_add
func.func @test_comb_add(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: %{{.*}} = llvm.add %arg0, %arg1 : i32
  // CHECK: llvm.return
  %0 = comb.add %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: llvm.func @test_comb_sub
func.func @test_comb_sub(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: %{{.*}} = llvm.sub %arg0, %arg1 : i32
  // CHECK: llvm.return
  %0 = comb.sub %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: llvm.func @test_comb_mul
func.func @test_comb_mul(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: %{{.*}} = llvm.mul %arg0, %arg1 : i32
  // CHECK: llvm.return
  %0 = comb.mul %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: llvm.func @test_comb_and
func.func @test_comb_and(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: %{{.*}} = llvm.and %arg0, %arg1 : i32
  // CHECK: llvm.return
  %0 = comb.and %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: llvm.func @test_comb_or
func.func @test_comb_or(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: %{{.*}} = llvm.or %arg0, %arg1 : i32
  // CHECK: llvm.return
  %0 = comb.or %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: llvm.func @test_comb_xor
func.func @test_comb_xor(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: %{{.*}} = llvm.xor %arg0, %arg1 : i32
  // CHECK: llvm.return
  %0 = comb.xor %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: llvm.func @test_comb_parity
func.func @test_comb_parity(%arg0: i32) -> i1 {
  // Parity should use ctpop + trunc (no Comb->Arith pattern exists)
  // CHECK: %[[CTPOP:.*]] = llvm.intr.ctpop(%arg0) : (i32) -> i32
  // CHECK: %{{.*}} = llvm.trunc %[[CTPOP]] : i32 to i1
  // CHECK: llvm.return
  %0 = comb.parity %arg0 : i32
  return %0 : i1
}

// CHECK-LABEL: llvm.func @test_comb_reverse
func.func @test_comb_reverse(%arg0: i32) -> i32 {
  // Reverse should use llvm.intr.bitreverse (no Comb->Arith pattern exists)
  // CHECK: %{{.*}} = llvm.intr.bitreverse(%arg0) : (i32) -> i32
  // CHECK: llvm.return
  %0 = comb.reverse %arg0 : i32
  return %0 : i32
}

// CHECK-LABEL: llvm.func @test_comb_shifts
func.func @test_comb_shifts(%arg0: i32, %arg1: i32) -> (i32, i32, i32) {
  // Test shift operations (converted via Comb->Arith->LLVM)
  // CHECK: llvm.shl
  // CHECK: llvm.lshr
  // CHECK: llvm.ashr
  // CHECK: llvm.return
  %0 = comb.shl %arg0, %arg1 : i32
  %1 = comb.shru %arg0, %arg1 : i32
  %2 = comb.shrs %arg0, %arg1 : i32
  return %0, %1, %2 : i32, i32, i32
}

// CHECK-LABEL: llvm.func @test_comb_icmp
func.func @test_comb_icmp(%arg0: i32, %arg1: i32) -> (i1, i1, i1, i1) {
  // Test comparison operations (converted via Comb->Arith->LLVM)
  // CHECK: llvm.icmp "eq"
  // CHECK: llvm.icmp "ne"
  // CHECK: llvm.icmp "slt"
  // CHECK: llvm.icmp "ult"
  // CHECK: llvm.return
  %0 = comb.icmp eq %arg0, %arg1 : i32
  %1 = comb.icmp ne %arg0, %arg1 : i32
  %2 = comb.icmp slt %arg0, %arg1 : i32
  %3 = comb.icmp ult %arg0, %arg1 : i32
  return %0, %1, %2, %3 : i1, i1, i1, i1
}

// CHECK-LABEL: llvm.func @test_comb_mux
func.func @test_comb_mux(%cond: i1, %arg0: i32, %arg1: i32) -> i32 {
  // Test mux operation (converted via Comb->Arith->LLVM)
  // CHECK: llvm.select
  // CHECK: llvm.return
  %0 = comb.mux %cond, %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: llvm.func @test_comb_replicate
func.func @test_comb_replicate(%arg0: i4) -> i16 {
  // Test replicate operation (converted via Comb->Arith->LLVM)
  // CHECK: llvm.zext
  // CHECK: llvm.or
  // CHECK: llvm.return
  %0 = comb.replicate %arg0 : (i4) -> i16
  return %0 : i16
}

// CHECK-LABEL: llvm.func @test_variadic_operations
func.func @test_variadic_operations(%arg0: i32, %arg1: i32, %arg2: i32) -> (i32, i32, i32) {
  // Test variadic operations (converted via Comb->Arith->LLVM)
  // CHECK: llvm.add
  // CHECK: llvm.add
  // CHECK: llvm.mul
  // CHECK: llvm.mul
  // CHECK: llvm.and
  // CHECK: llvm.and
  // CHECK: llvm.return
  %0 = comb.add %arg0, %arg1, %arg2 : i32
  %1 = comb.mul %arg0, %arg1, %arg2 : i32
  %2 = comb.and %arg0, %arg1, %arg2 : i32
  return %0, %1, %2 : i32, i32, i32
}

// CHECK-LABEL: llvm.func @test_control_flow
func.func @test_control_flow(%cond: i1, %arg0: i32) -> i32 {
  // Test that control flow is also converted (SCF->ControlFlow->LLVM)
  // CHECK: llvm.cond_br
  // CHECK: llvm.br
  // CHECK: llvm.return
  %result = scf.if %cond -> i32 {
    %0 = comb.add %arg0, %arg0 : i32
    scf.yield %0 : i32
  } else {
    %1 = comb.mul %arg0, %arg0 : i32
    scf.yield %1 : i32
  }
  return %result : i32
}

// -----

// Test HW module and SV operations (should NOT be converted - only func.func operations are converted)
hw.module @test_hw_sv_module(in %scan_data : i9) {
  %0 = comb.extract %scan_data from 0 : (i9) -> i8
  hw.output
}


// CHECK-LABEL: hw.module @test_hw_sv_module
// CHECK-SAME: (in %scan_data : i9)
// CHECK: %[[EXTRACT1:.+]] = comb.extract %scan_data from 0 : (i9) -> i8
