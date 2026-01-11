// RUN: circt-opt --cse %s | FileCheck %s

// CHECK-LABEL: @CanCSECurrentTimeWithoutSideEffectsInBetween
func.func @CanCSECurrentTimeWithoutSideEffectsInBetween() -> (!llhd.time, !llhd.time) {
  // CHECK-NEXT: [[TMP0:%.+]] = llhd.current_time
  %0 = llhd.current_time
  // CHECK-NOT: llhd.current_time
  %1 = llhd.current_time
  // CHECK-NEXT: return [[TMP0]], [[TMP0]]
  return %0, %1 : !llhd.time, !llhd.time
}

// CHECK-LABEL: @CannotCSECurrentTimeWithSideEffectsInBetween
func.func @CannotCSECurrentTimeWithSideEffectsInBetween() -> (!llhd.time, !llhd.time) {
  // CHECK-NEXT: [[TMP0:%.+]] = llhd.current_time
  %0 = llhd.current_time
  // CHECK-NEXT: call @UnknownSideEffects
  call @UnknownSideEffects() : () -> ()
  // CHECK-NEXT: [[TMP1:%.+]] = llhd.current_time
  %1 = llhd.current_time
  // CHECK-NEXT: return [[TMP0]], [[TMP1]]
  return %0, %1 : !llhd.time, !llhd.time
}

func.func private @UnknownSideEffects()
