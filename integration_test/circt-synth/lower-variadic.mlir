// REQUIRES: libz3
// REQUIRES: circt-lec-jit

// RUN: circt-opt %s -convert-synth-to-comb -o %t.before.mlir
// RUN: circt-opt %s -synth-lower-variadic -convert-synth-to-comb -o %t.after.mlir
// RUN: circt-lec %t.before.mlir %t.after.mlir -c1=AndInverter -c2=AndInverter --shared-libs=%libz3 | FileCheck %s --check-prefix=AND_INVERTER_LEC
// AND_INVERTER_LEC: c1 == c2
hw.module @AndInverter(in %a: i2, in %b: i2, in %c: i2, in %d: i2, in %e: i2, in %f: i2, in %g: i2, out o1: i2) {
  %0 = synth.aig.and_inv %d, not %e : i2
  %1 = synth.aig.and_inv not %c, not %0, %f : i2
  %2 = synth.aig.and_inv %a, not %b, not %1, %g : i2
  hw.output %2 : i2
}

// RUN: circt-lec %t.before.mlir %t.after.mlir -c1=VariadicCombOps -c2=VariadicCombOps --shared-libs=%libz3 | FileCheck %s --check-prefix=VARIADIC_COMB_OPS_LEC
// VARIADIC_COMB_OPS_LEC: c1 == c2
hw.module @VariadicCombOps(in %a: i2, in %b: i2, in %c: i2, in %d: i2, in %e: i2, in %f: i2, 
                           out out_and: i2, out out_or: i2, out out_xor: i2) {
  %0 = comb.and %a, %b, %c, %d, %e, %f : i2
  %1 = comb.or %a, %b, %c, %d, %e, %f : i2
  %2 = comb.xor %a, %b, %c, %d, %e, %f : i2
  hw.output %0, %1, %2 : i2, i2, i2
}
