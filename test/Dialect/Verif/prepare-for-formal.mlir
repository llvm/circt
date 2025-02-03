// RUN: circt-opt --prepare-for-formal %s | FileCheck %s
module {
  // CHECK-LABEL: @FlattenWires
  hw.module @FlattenWires(in %clock : !seq.clock, in %reset : i1, in %in : i1, out out : i32) {

    // CHECK:  %false = hw.constant false
    %false = hw.constant false
    // CHECK:  %c0_i32 = hw.constant 0 : i32
    %c0_i32 = hw.constant 0 : i32
    // CHECK:  %true = hw.constant true
    %true = hw.constant true
    // CHECK:  %c-2_i2 = hw.constant -2 : i2
    %c-2_i2 = hw.constant -2 : i2
    // CHECK:  %count = seq.compreg %6, %clock reset %reset, %c0_i32 : i32  
    %count = seq.compreg %6, %clock reset %reset, %c0_i32 : i32
    // CHECK-NOT: hw.wire
    %wf = hw.wire %false sym @f : i1
    %b = hw.wire %2 sym @b : i32
    %d = hw.wire %b sym @d : i32
    %c = hw.wire %5 sym @c : i32
    %w_in = hw.wire %in : i1
    %ww_in = hw.wire %w_in sym @in_w : i1
    // CHECK:  %0 = comb.concat %false, %count : i1, i32
    %0 = comb.concat %wf, %count : i1, i32
    // CHECK:  %c1_i33 = hw.constant 1 : i33
    %c1_i33 = hw.constant 1 : i33
    // CHECK:  %1 = comb.sub %0, %c1_i33 : i33
    %1 = comb.sub %0, %c1_i33 : i33
    // CHECK:  %2 = comb.extract %1 from 0 : (i33) -> i32
    %2 = comb.extract %1 from 0 : (i33) -> i32
    // CHECK:  %3 = comb.concat %false, %2 : i1, i32
    %3 = comb.concat %wf, %d : i1, i32
    // CHECK:  %c2_i33 = hw.constant 2 : i33
    %c2_i33 = hw.constant 2 : i33
    // CHECK:  %4 = comb.add %3, %c2_i33 : i33
    %4 = comb.add %3, %c2_i33 : i33
    // CHECK:  %5 = comb.extract %4 from 0 : (i33) -> i32
    %5 = comb.extract %4 from 0 : (i33) -> i32
    // CHECK:  %6 = comb.mux %in, %2, %5 : i32
    %6 = comb.mux %ww_in, %b, %c : i32
    // CHECK:  hw.output %count : i32
    hw.output %count : i32
  }

}

