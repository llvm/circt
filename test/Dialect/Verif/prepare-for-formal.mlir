// RUN: circt-opt --prepare-for-formal %s | FileCheck %s
module {
  // CHECK-LABEL: @FlattenWires
  hw.module @FlattenWires(in %clock : !seq.clock, in %reset : i1, in %in : i1, out out : i32) {

    %false = hw.constant false
    %c0_i32 = hw.constant 0 : i32
    %true = hw.constant true
    %c-2_i2 = hw.constant -2 : i2
    %count = seq.compreg %6, %clock reset %reset, %c0_i32 : i32
    %b = hw.wire %2  : i32
    %d = hw.wire %b  : i32
    %c = hw.wire %5  : i32
    %w_in = hw.wire %in : i1
    %ww_in = hw.wire %w_in sym @in_w : i1
    %0 = comb.concat %false, %count : i1, i32
    %c1_i33 = hw.constant 1 : i33
    %1 = comb.sub bin %0, %c1_i33 : i33
    %2 = comb.extract %1 from 0 : (i33) -> i32
    %3 = comb.concat %false, %d : i1, i32
    %c2_i33 = hw.constant 2 : i33
    %4 = comb.add bin %3, %c2_i33 : i33
    %5 = comb.extract %4 from 0 : (i33) -> i32
    %6 = comb.mux bin %ww_in, %b, %c : i32
    hw.output %count : i32
  }

}

