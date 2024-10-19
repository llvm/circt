// REQUIRES: libz3
// REQUIRES: circt-lec-jit

hw.module @basic(in %in: i1, out out: i1) {
  hw.output %in : i1
}

// hw.constant
//  RUN: circt-lec %s -c1=basic -c2=notnot --shared-libs=%libz3 | FileCheck %s --check-prefix=HW_CONSTANT
//  HW_CONSTANT: c1 == c2

hw.module @onePlusTwo(out out: i2) {
  %one = hw.constant 1 : i2
  %two = hw.constant 2 : i2
  %three = comb.add bin %one, %two : i2
  hw.output %three : i2
}

hw.module @three(out out: i2) {
  %three = hw.constant 3 : i2
  hw.output %three : i2
}

// hw.instance
//  RUN: circt-lec %s -c1=basic -c2=notnot --shared-libs=%libz3 | FileCheck %s --check-prefix=HW_INSTANCE
//  HW_INSTANCE: c1 == c2

hw.module @not(in %in: i1, out out: i1) {
  %true = hw.constant true
  %out = comb.xor bin %in, %true : i1
  hw.output %out : i1
}

hw.module @notnot(in %in: i1, out out: i1) {
  %n = hw.instance "n" @not(in: %in: i1) -> (out: i1)
  %nn = hw.instance "nn" @not(in: %n: i1) -> (out: i1)
  hw.output %nn : i1
}

// hw.output
//  RUN: circt-lec %s -c1=basic -c2=basic --shared-libs=%libz3 | FileCheck %s --check-prefix=HW_OUTPUT
//  HW_OUTPUT: c1 == c2

hw.module @constZeroZero(in %in: i1, out o1: i1, out o2: i1) {
  %zero = hw.constant 0 : i1
  hw.output %zero, %zero : i1, i1
}

hw.module @xorZeroZero(in %in: i1, out o1: i1, out o2: i1) {
  %zero = comb.xor bin %in, %in : i1
  hw.output %zero, %zero : i1, i1
}

hw.module @constZeroOne(in %in: i1, out o1: i1, out o2: i1) {
  %zero = hw.constant 0 : i1
  %one  = hw.constant 1 : i1
  hw.output %zero, %one : i1, i1
}

// Equivalent modules with two outputs
//  RUN: circt-lec %s -c1=constZeroZero -c2=xorZeroZero --shared-libs=%libz3 | FileCheck %s --check-prefix=TWOOUTPUTS
//  TWOOUTPUTS: c1 == c2

// Modules with one equivalent and one non-equivalent output
//  RUN: circt-lec %s -c1=constZeroZero -c2=constZeroOne --shared-libs=%libz3 | FileCheck %s --check-prefix=TWOOUTPUTSFAIL
//  TWOOUTPUTSFAIL: c1 != c2

hw.module @onePlusTwoNonSSA(out out: i2) {
  %three = comb.add bin %one, %two : i2
  %one = hw.constant 1 : i2
  %two = hw.constant 2 : i2
  hw.output %three : i2
}

// hw.module graph region check
//  RUN: circt-lec %s -c1=onePlusTwo -c2=onePlusTwoNonSSA --shared-libs=%libz3 | FileCheck %s --check-prefix=HW_MODULE_GRAPH
//  HW_MODULE_GRAPH: c1 == c2

// array_create + array_get test
//  RUN: circt-lec %s -c1=MultibitMux -c2=MultibitMux2 --shared-libs=%libz3 | FileCheck %s --check-prefix=ARRAY_GET
//  ARRAY_GET: c1 == c2

hw.module @MultibitMux(in %a_0 : i1, in %a_1 : i1, in %sel : i1, out b : i1) {
  %0 = hw.array_create %a_1, %a_0 : i1
  %1 = hw.array_get %0[%sel] : !hw.array<2xi1>, i1
  hw.output %1 : i1
}

hw.module @MultibitMux2(in %a_0 : i1, in %a_1 : i1, in %sel : i1, out b : i1) {
  %0 = comb.mux bin %sel, %a_1, %a_0 : i1
  hw.output %0 : i1
}

// array_get out-of-bounds must not be equivalent
//  RUN: circt-lec %s -c1=ArrayOOB -c2=ArrayOOB --shared-libs=%libz3 | FileCheck %s --check-prefix=ARRAY_OOB
//  ARRAY_OOB: c1 != c2

hw.module @ArrayOOB(in %a : !hw.array<3xi1>, out b : i1) {
  %0 = hw.constant 3 : i2
  %1 = hw.array_get %a[%0] : !hw.array<3xi1>, i2
  hw.output %1 : i1
}
