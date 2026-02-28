// RUN: circt-opt --lower-to-bmc="top-module=NoComb bound=10" %s | FileCheck %s

// CHECK:  func.func @NoComb() {
hw.module @NoComb(in %clk: !seq.clock) attributes {num_regs = 0 : i32, initial_values = []} {
  %true = hw.constant true
  verif.assert %true : i1
}
