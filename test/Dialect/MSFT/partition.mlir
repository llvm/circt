// RUN: circt-opt %s --msft-partition --msft-wire-cleanup -verify-diagnostics -split-input-file | FileCheck --check-prefix=CLEANUP %s

hw.globalRef @ref1 [#hw.innerNameRef<@top::@b>, #hw.innerNameRef<@B::@unit1>] {
  "loc" = #msft.physloc<M20K, 0, 0, 0>
}

hw.globalRef @ref2 [#hw.innerNameRef<@top::@b>, #hw.innerNameRef<@B::@c>, #hw.innerNameRef<@C::@unit3>] {
  "loc" = #msft.physloc<M20K, 0, 0, 1>
}

msft.module @top {} (%clk : i1) -> (out1: i2, out2: i2) {
  msft.partition @part1, "dp"

  %res1 = msft.instance @b @B(%clk) { circt.globalRef = [#hw.globalNameRef<@ref1>, #hw.globalNameRef<@ref2>], inner_sym = "b" } : (i1) -> (i2)

  %c0 = hw.constant 0 : i2
  %res2 = msft.instance @unit1 @Extern(%c0) { targetDesignPartition = @top::@part1 }: (i2) -> (i2)

  msft.output %res1, %res2 : i2, i2
}

msft.module.extern @Extern (%foo_a: i2) -> (foo_x: i2)

msft.module @B {} (%clk : i1) -> (x: i2, y: i2)  {
  %c1 = hw.constant 1 : i2
  %0 = msft.instance @unit1 @Extern(%c1) { targetDesignPartition = @top::@part1, circt.globalRef = [#hw.globalNameRef<@ref1>], inner_sym = "unit1" }: (i2) -> (i2)
  %1 = seq.compreg %0, %clk { targetDesignPartition = @top::@part1 } : i2

  %2 = msft.instance @unit2 @Extern(%1) { targetDesignPartition = @top::@part1 }: (i2) -> (i2)

  %3 = msft.instance @c @C(%2) { targetDesignPartition = @top::@part1, circt.globalRef = [#hw.globalNameRef<@ref2>], inner_sym = "c" }: (i2) -> (i2)

  msft.output %2, %3: i2, i2
}

msft.module @C {} (%in : i2) -> (out: i2)  {
  %0 = msft.instance @unit3 @Extern(%in) { targetDesignPartition = @top::@part1, circt.globalRef = [#hw.globalNameRef<@ref2>], inner_sym = "unit3" } : (i2) -> (i2)
  msft.output %0 : i2
}

msft.module @TopComplex {} (%clk : i1, %arr_in: !hw.array<4xi5>) -> (out2: i5) {
  msft.partition @part2, "dp_complex"

  %mut_arr = msft.instance @b @Array(%arr_in) : (!hw.array<4xi5>) -> (!hw.array<4xi5>)
  %c0 = hw.constant 0 : i2
  %a0 = hw.array_get %mut_arr[%c0] : !hw.array<4xi5>
  %c1 = hw.constant 1 : i2
  %a1 = hw.array_get %mut_arr[%c1] : !hw.array<4xi5>
  %c2 = hw.constant 2 : i2
  %a2 = hw.array_get %mut_arr[%c2] : !hw.array<4xi5>
  %c3 = hw.constant 3 : i2
  %a3 = hw.array_get %mut_arr[%c3] : !hw.array<4xi5>

  %res1 = comb.add %a0, %a1, %a2, %a3 { targetDesignPartition = @TopComplex::@part2 } : i5

  msft.output %res1 : i5
}

msft.module.extern @ExternI5 (%foo_a: i5) -> (foo_x: i5)

msft.module @Array {} (%arr_in: !hw.array<4xi5>) -> (arr_out: !hw.array<4xi5>) {
  %c0 = hw.constant 0 : i2
  %in0 = hw.array_get %arr_in[%c0] : !hw.array<4xi5>
  %out0 = msft.instance @unit2 @ExternI5(%in0) { targetDesignPartition = @TopComplex::@part2 }: (i5) -> (i5)
  %c1 = hw.constant 1 : i2
  %in1 = hw.array_get %arr_in[%c1] : !hw.array<4xi5>
  %out1 = msft.instance @unit2 @ExternI5(%in1) { targetDesignPartition = @TopComplex::@part2 }: (i5) -> (i5)
  %c2 = hw.constant 2 : i2
  %in2 = hw.array_get %arr_in[%c2] : !hw.array<4xi5>
  %out2 = msft.instance @unit2 @ExternI5(%in2) { targetDesignPartition = @TopComplex::@part2 }: (i5) -> (i5)
  %c3 = hw.constant 3 : i2
  %in3 = hw.array_get %arr_in[%c3] : !hw.array<4xi5>
  %out3 = msft.instance @unit2 @ExternI5(%in3) { targetDesignPartition = @TopComplex::@part2 }: (i5) -> (i5)
  %arr_out = hw.array_create %out0, %out1, %out2, %out3 : i5
  msft.output %arr_out : !hw.array<4xi5>
}

// CLEANUP:  hw.globalRef @ref1 [#hw.innerNameRef<@top::@part1>, #hw.innerNameRef<@dp::@b.unit1>] {loc = #msft.physloc<M20K, 0, 0, 0>}
// CLEANUP:  hw.globalRef @ref2 [#hw.innerNameRef<@top::@part1>, #hw.innerNameRef<@dp::@b.c.unit3>] {loc = #msft.physloc<M20K, 0, 0, 1>}

// CLEANUP-LABEL:  msft.module @top {} (%clk: i1) -> (out1: i2, out2: i2) {
// CLEANUP:          %part1.b.unit2.foo_x, %part1.unit1.foo_x = msft.instance @part1 @dp(%clk) {circt.globalRef = [{{.+}}], inner_sym = "part1"} : (i1) -> (i2, i2)
// CLEANUP:          msft.instance @b @B() {{.*}} : () -> ()
// CLEANUP:          msft.output %part1.b.unit2.foo_x, %part1.unit1.foo_x : i2, i2
// CLEANUP-LABEL:  msft.module.extern @Extern(%foo_a: i2) -> (foo_x: i2)
// CLEANUP-LABEL:  msft.module @B {} ()
// CLEANUP:          msft.output

// CLEANUP-LABEL:  msft.module @TopComplex {} (%arr_in: !hw.array<4xi5>) -> (out2: i5)
// CLEANUP:          %part2.comb.add = msft.instance @part2 @dp_complex(%arr_in, %arr_in, %arr_in, %arr_in) {{.*}} : (!hw.array<4xi5>, !hw.array<4xi5>, !hw.array<4xi5>, !hw.array<4xi5>) -> i5
// CLEANUP:          msft.instance @b @Array()  : () -> ()
// CLEANUP:          msft.output %part2.comb.add : i5
// CLEANUP-LABEL:  msft.module.extern @ExternI5(%foo_a: i5) -> (foo_x: i5)
// CLEANUP-LABEL:  msft.module @Array {} ()
// CLEANUP:          msft.output

// CLEANUP-LABEL:  msft.module @dp {} (%b.seq.compreg.in1: i1) -> (b.unit2.foo_x: i2, unit1.foo_x: i2) {
// CLEANUP:          %c1_i2 = hw.constant 1 : i2
// CLEANUP:          %b.unit1.foo_x = msft.instance @b.unit1 @Extern(%c1_i2) {circt.globalRef = [#hw.globalNameRef<@ref1>], inner_sym = "b.unit1"} : (i2) -> i2
// CLEANUP:          %b.seq.compreg = seq.compreg %b.unit1.foo_x, %b.seq.compreg.in1 : i2
// CLEANUP:          %b.unit2.foo_x = msft.instance @b.unit2 @Extern(%b.seq.compreg)  : (i2) -> i2
// CLEANUP:          %{{.+}} = msft.instance @b.c.unit3 {{.+}} {circt.globalRef = [#hw.globalNameRef<@ref2>], inner_sym = "b.c.unit3"}
// CLEANUP:          %c0_i2 = hw.constant 0 : i2
// CLEANUP:          %unit1.foo_x = msft.instance @unit1 @Extern(%c0_i2)  : (i2) -> i2
// CLEANUP:          msft.output %b.unit2.foo_x, %unit1.foo_x : i2, i2

// CLEANUP-LABEL:  msft.module @dp_complex {} (%hw.array_get.in0: !hw.array<4xi5>, %hw.array_get.in0_0: !hw.array<4xi5>, %hw.array_get.in0_1: !hw.array<4xi5>, %hw.array_get.in0_2: !hw.array<4xi5>) -> (comb.add: i5) attributes {argNames = ["hw.array_get.in0", "hw.array_get.in0", "hw.array_get.in0", "hw.array_get.in0"]} {
// CLEANUP:          %c0_i2 = hw.constant 0 : i2
// CLEANUP:          %0 = hw.array_get %hw.array_get.in0[%c0_i2] : !hw.array<4xi5>
// CLEANUP:          %b.unit2.foo_x = msft.instance @b.unit2 @ExternI5(%0)  : (i5) -> i5
// CLEANUP:          %c1_i2 = hw.constant 1 : i2
// CLEANUP:          %1 = hw.array_get %hw.array_get.in0_0[%c1_i2] : !hw.array<4xi5>
// CLEANUP:          %b.unit2.foo_x_3 = msft.instance @b.unit2 @ExternI5(%1)  : (i5) -> i5
// CLEANUP:          %c-2_i2 = hw.constant -2 : i2
// CLEANUP:          %2 = hw.array_get %hw.array_get.in0_1[%c-2_i2] : !hw.array<4xi5>
// CLEANUP:          %b.unit2.foo_x_4 = msft.instance @b.unit2 @ExternI5(%2)  : (i5) -> i5
// CLEANUP:          %c-1_i2 = hw.constant -1 : i2
// CLEANUP:          %3 = hw.array_get %hw.array_get.in0_2[%c-1_i2] : !hw.array<4xi5>
// CLEANUP:          %b.unit2.foo_x_5 = msft.instance @b.unit2 @ExternI5(%3)  : (i5) -> i5
// CLEANUP:          %4 = hw.array_create %b.unit2.foo_x, %b.unit2.foo_x_3, %b.unit2.foo_x_4, %b.unit2.foo_x_5 : i5
// CLEANUP:          %c0_i2_6 = hw.constant 0 : i2
// CLEANUP:          %5 = hw.array_get %4[%c0_i2_6] : !hw.array<4xi5>
// CLEANUP:          %c1_i2_7 = hw.constant 1 : i2
// CLEANUP:          %6 = hw.array_get %4[%c1_i2_7] : !hw.array<4xi5>
// CLEANUP:          %c-2_i2_8 = hw.constant -2 : i2
// CLEANUP:          %7 = hw.array_get %4[%c-2_i2_8] : !hw.array<4xi5>
// CLEANUP:          %c-1_i2_9 = hw.constant -1 : i2
// CLEANUP:          %8 = hw.array_get %4[%c-1_i2_9] : !hw.array<4xi5>
// CLEANUP:          %9 = comb.add %5, %6, %7, %8 : i5
// CLEANUP:          msft.output %9 : i5
