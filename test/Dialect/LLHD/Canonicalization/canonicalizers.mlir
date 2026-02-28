// RUN: circt-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: @CanRemoveUnusedCurrentTime
func.func @CanRemoveUnusedCurrentTime() {
  // CHECK-NOT: llhd.current_time
  llhd.current_time
  // CHECK-NEXT: return
  return
}
