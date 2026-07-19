// RUN: circt-opt %s | FileCheck %s

func.func @trace(%step: i32, %value: i8) {
  // CHECK: verif.bmc.trace %arg0, "state_q", %arg1 : i32, i8
  verif.bmc.trace %step, "state_q", %value : i32, i8
  return
}
