// RUN: circt-opt %s --lower-comb | FileCheck %s
// RUN: circt-opt %s --lower-comb=assign-to-wire=true | FileCheck %s --check-prefix=WIRE

// CHECK-LABEL: hw.module @err
// CHECK:         [[R0:%.+]] = comb.mux bin %b, %true, %false : i1
// CHECK:         [[R1:%.+]] = comb.mux bin %b, %false, %true : i1
// CHECK:         [[R2:%.+]] = comb.mux bin %a, [[R0]], [[R1]] {sv.namehint = "lut1"} : i1
// WIRE:        hw.module @err
// WIRE:          [[R0:%.+]] = comb.mux bin %b, %true, %false : i1
// WIRE:          [[R1:%.+]] = comb.mux bin %b, %false, %true : i1
// WIRE:          [[R2:%.+]] = comb.mux bin %a, [[R0]], [[R1]] : i1
// WIRE:          %lut1 = hw.wire [[R2]] sym @__truth_table_mux_0  {sv.namehint = "lut1"} : i1
hw.module @err(%a: i1, %b: i1) -> (x: i1) {
  %0 = comb.truth_table %a, %b -> [true, false, false, true] {sv.namehint="lut1"}
  hw.output %0 : i1
}
