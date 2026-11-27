// RUN: arcilator %s --lookup-tables=true --run --jit-entry=main | FileCheck --match-full-lines %s
// RUN: arcilator %s --lookup-tables=false --run --jit-entry=main | FileCheck --match-full-lines %s
// REQUIRES: arcilator-jit

// End-to-end JIT test for constant arrays lowered via arc.arrayref.constant.

// CHECK:      lut_00 = 0
// CHECK-NEXT: lut_10 = b
// CHECK-NEXT: lut_53 = 4
// CHECK-NEXT: i1_0 = 0
// CHECK-NEXT: i1_1 = 1
// CHECK-NEXT: i3_0 = 7
// CHECK-NEXT: i3_2 = 3
// CHECK-NEXT: slice_0 = 1
// CHECK-NEXT: slice_1 = 3
// CHECK-NEXT: mux_0 = 00000014
// CHECK-NEXT: mux_1 = 0000001e
// CHECK-NEXT: i24_0 = abcdef
// CHECK-NEXT: i24_1 = 123456
// CHECK-NEXT: i72_lo_0 = 0203040506070809
// CHECK-NEXT: i72_hi_0 = 01
// CHECK-NEXT: i72_lo_1 = 2233445566778899
// CHECK-NEXT: i72_hi_1 = 11
// CHECK-NEXT: i129_hi_0 = 0
// CHECK-NEXT: i129_hi_1 = 1
// CHECK-NEXT: cow_orig = 000001c8
// CHECK-NEXT: cow_inj = 000003e7

hw.module @LookupTable(in %a: i4, in %b: i4, out y: i4) {
  %0 = comb.add %a, %b : i4
  %1 = comb.xor %0, %a : i4
  %2 = comb.mul %1, %b : i4
  %3 = comb.add %2, %0 : i4
  %4 = comb.xor %3, %1 : i4
  %5 = comb.mul %4, %a : i4
  %6 = comb.add %5, %b : i4
  %7 = comb.xor %6, %2 : i4
  %8 = comb.mul %7, %3 : i4
  %9 = comb.add %8, %4 : i4
  %10 = comb.xor %9, %5 : i4
  %11 = comb.mul %10, %6 : i4
  %12 = comb.add %11, %7 : i4
  %13 = comb.xor %12, %8 : i4
  %14 = comb.mul %13, %9 : i4
  %15 = comb.add %14, %10 : i4
  %16 = comb.xor %15, %11 : i4
  %17 = comb.mul %16, %12 : i4
  %18 = comb.add %17, %13 : i4
  %19 = comb.xor %18, %14 : i4
  %20 = comb.mul %19, %15 : i4
  hw.output %20 : i4
}

hw.module @ConstArrays(
    in %idx1: i1, in %idx2: i2,
    out out_i1: i1, out out_i3: i3, out out_slice: i3, out out_mux: i32,
    out out_i24: i24, out out_i72_lo: i64, out out_i72_hi: i8,
    out out_i129_hi: i1, out cow_orig: i32, out cow_inj: i32) {
  %arr_i1 = hw.aggregate_constant [true, false] : !hw.array<2xi1>
  %val_i1 = hw.array_get %arr_i1[%idx1] : !hw.array<2xi1>, i1

  %arr_i3 = hw.aggregate_constant [5 : i3, 3 : i3, 1 : i3, 7 : i3] : !hw.array<4xi3>
  %val_i3 = hw.array_get %arr_i3[%idx2] : !hw.array<4xi3>, i2
  %c1_i2 = hw.constant 1 : i2
  %arr_i3_slice = hw.array_slice %arr_i3[%c1_i2] : (!hw.array<4xi3>) -> !hw.array<2xi3>
  %val_slice = hw.array_get %arr_i3_slice[%idx1] : !hw.array<2xi3>, i1

  %arr_m0 = hw.aggregate_constant [10 : i32, 20 : i32] : !hw.array<2xi32>
  %arr_m1 = hw.aggregate_constant [30 : i32, 40 : i32] : !hw.array<2xi32>
  %arr_mux = comb.mux %idx1, %arr_m1, %arr_m0 : !hw.array<2xi32>
  %val_mux = hw.array_get %arr_mux[%idx1] : !hw.array<2xi32>, i1

  %arr_i24 = hw.aggregate_constant [0x123456 : i24, 0xabcdef : i24] : !hw.array<2xi24>
  %val_i24 = hw.array_get %arr_i24[%idx1] : !hw.array<2xi24>, i1

  %arr_i72 = hw.aggregate_constant [0x112233445566778899 : i72, 0x010203040506070809 : i72] : !hw.array<2xi72>
  %val_i72 = hw.array_get %arr_i72[%idx1] : !hw.array<2xi72>, i1
  %val_i72_lo = comb.extract %val_i72 from 0 : (i72) -> i64
  %val_i72_hi = comb.extract %val_i72 from 64 : (i72) -> i8

  %arr_i129 = hw.aggregate_constant [0x100000000000000000000000000000000 : i129, 0x0 : i129] : !hw.array<2xi129>
  %val_i129 = hw.array_get %arr_i129[%idx1] : !hw.array<2xi129>, i1
  %val_i129_hi = comb.extract %val_i129 from 128 : (i129) -> i1

  %arr_i32 = hw.aggregate_constant [123 : i32, 456 : i32] : !hw.array<2xi32>
  %c0_i1 = hw.constant 0 : i1
  %c999 = hw.constant 999 : i32
  %arr_inj = hw.array_inject %arr_i32[%c0_i1], %c999 : !hw.array<2xi32>, i1
  %orig_val = hw.array_get %arr_i32[%c0_i1] : !hw.array<2xi32>, i1
  %inj_val = hw.array_get %arr_inj[%c0_i1] : !hw.array<2xi32>, i1

  hw.output %val_i1, %val_i3, %val_slice, %val_mux, %val_i24, %val_i72_lo,
            %val_i72_hi, %val_i129_hi, %orig_val, %inj_val :
      i1, i3, i3, i32, i24, i64, i8, i1, i32, i32
}

func.func @main() {
  %c0_i4 = arith.constant 0 : i4
  %c1_i4 = arith.constant 1 : i4
  %c3_i4 = arith.constant 3 : i4
  %c5_i4 = arith.constant 5 : i4
  %c0_i1 = arith.constant 0 : i1
  %c1_i1 = arith.constant 1 : i1
  %c0_i2 = arith.constant 0 : i2
  %c2_i2 = arith.constant 2 : i2

  arc.sim.instantiate @LookupTable as %lut {
    arc.sim.set_input %lut, "a" = %c0_i4 : i4, !arc.sim.instance<@LookupTable>
    arc.sim.set_input %lut, "b" = %c0_i4 : i4, !arc.sim.instance<@LookupTable>
    arc.sim.step %lut : !arc.sim.instance<@LookupTable>
    %y00 = arc.sim.get_port %lut, "y" : i4, !arc.sim.instance<@LookupTable>
    arc.sim.emit "lut_00", %y00 : i4

    arc.sim.set_input %lut, "a" = %c1_i4 : i4, !arc.sim.instance<@LookupTable>
    arc.sim.set_input %lut, "b" = %c0_i4 : i4, !arc.sim.instance<@LookupTable>
    arc.sim.step %lut : !arc.sim.instance<@LookupTable>
    %y10 = arc.sim.get_port %lut, "y" : i4, !arc.sim.instance<@LookupTable>
    arc.sim.emit "lut_10", %y10 : i4

    arc.sim.set_input %lut, "a" = %c5_i4 : i4, !arc.sim.instance<@LookupTable>
    arc.sim.set_input %lut, "b" = %c3_i4 : i4, !arc.sim.instance<@LookupTable>
    arc.sim.step %lut : !arc.sim.instance<@LookupTable>
    %y53 = arc.sim.get_port %lut, "y" : i4, !arc.sim.instance<@LookupTable>
    arc.sim.emit "lut_53", %y53 : i4
  }

  arc.sim.instantiate @ConstArrays as %m {
    arc.sim.set_input %m, "idx1" = %c0_i1 : i1, !arc.sim.instance<@ConstArrays>
    arc.sim.set_input %m, "idx2" = %c0_i2 : i2, !arc.sim.instance<@ConstArrays>
    arc.sim.step %m : !arc.sim.instance<@ConstArrays>
    %i1_0 = arc.sim.get_port %m, "out_i1" : i1, !arc.sim.instance<@ConstArrays>
    %i3_0 = arc.sim.get_port %m, "out_i3" : i3, !arc.sim.instance<@ConstArrays>
    %slice_0 = arc.sim.get_port %m, "out_slice" : i3, !arc.sim.instance<@ConstArrays>
    %mux_0 = arc.sim.get_port %m, "out_mux" : i32, !arc.sim.instance<@ConstArrays>
    %i24_0 = arc.sim.get_port %m, "out_i24" : i24, !arc.sim.instance<@ConstArrays>
    %i72_lo_0 = arc.sim.get_port %m, "out_i72_lo" : i64, !arc.sim.instance<@ConstArrays>
    %i72_hi_0 = arc.sim.get_port %m, "out_i72_hi" : i8, !arc.sim.instance<@ConstArrays>
    %i129_hi_0 = arc.sim.get_port %m, "out_i129_hi" : i1, !arc.sim.instance<@ConstArrays>

    arc.sim.set_input %m, "idx1" = %c1_i1 : i1, !arc.sim.instance<@ConstArrays>
    arc.sim.set_input %m, "idx2" = %c2_i2 : i2, !arc.sim.instance<@ConstArrays>
    arc.sim.step %m : !arc.sim.instance<@ConstArrays>
    %i1_1 = arc.sim.get_port %m, "out_i1" : i1, !arc.sim.instance<@ConstArrays>
    %i3_2 = arc.sim.get_port %m, "out_i3" : i3, !arc.sim.instance<@ConstArrays>
    %slice_1 = arc.sim.get_port %m, "out_slice" : i3, !arc.sim.instance<@ConstArrays>
    %mux_1 = arc.sim.get_port %m, "out_mux" : i32, !arc.sim.instance<@ConstArrays>
    %i24_1 = arc.sim.get_port %m, "out_i24" : i24, !arc.sim.instance<@ConstArrays>
    %i72_lo_1 = arc.sim.get_port %m, "out_i72_lo" : i64, !arc.sim.instance<@ConstArrays>
    %i72_hi_1 = arc.sim.get_port %m, "out_i72_hi" : i8, !arc.sim.instance<@ConstArrays>
    %i129_hi_1 = arc.sim.get_port %m, "out_i129_hi" : i1, !arc.sim.instance<@ConstArrays>
    %cow_orig = arc.sim.get_port %m, "cow_orig" : i32, !arc.sim.instance<@ConstArrays>
    %cow_inj = arc.sim.get_port %m, "cow_inj" : i32, !arc.sim.instance<@ConstArrays>

    arc.sim.emit "i1_0", %i1_0 : i1
    arc.sim.emit "i1_1", %i1_1 : i1
    arc.sim.emit "i3_0", %i3_0 : i3
    arc.sim.emit "i3_2", %i3_2 : i3
    arc.sim.emit "slice_0", %slice_0 : i3
    arc.sim.emit "slice_1", %slice_1 : i3
    arc.sim.emit "mux_0", %mux_0 : i32
    arc.sim.emit "mux_1", %mux_1 : i32
    arc.sim.emit "i24_0", %i24_0 : i24
    arc.sim.emit "i24_1", %i24_1 : i24
    arc.sim.emit "i72_lo_0", %i72_lo_0 : i64
    arc.sim.emit "i72_hi_0", %i72_hi_0 : i8
    arc.sim.emit "i72_lo_1", %i72_lo_1 : i64
    arc.sim.emit "i72_hi_1", %i72_hi_1 : i8
    arc.sim.emit "i129_hi_0", %i129_hi_0 : i1
    arc.sim.emit "i129_hi_1", %i129_hi_1 : i1
    arc.sim.emit "cow_orig", %cow_orig : i32
    arc.sim.emit "cow_inj", %cow_inj : i32
  }
  return
}
