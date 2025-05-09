// RUN: circt-opt %s -split-input-file -verify-diagnostics

hw.module @err(in %a: i1, in %b: i1) {
  // expected-error @+1 {{Expected lookup table of 2^n length}}
  %0 = comb.truth_table %a, %b -> [true, false]
}

// -----

hw.module @err(in %a: i1, in %b: i1) {
  // expected-error @+1 {{Truth tables support a maximum of 63 inputs}}
  %0 = comb.truth_table %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b -> [false]
}

// -----

hw.module @err(in %a: i4, out out: i7) {
  // expected-note @+1 {{prior use here}}
  %0 = comb.concat %a, %a : i4, i4
  // expected-error @+1 {{use of value '%0' expects different type than prior uses: 'i7' vs 'i8'}}
  hw.output %0 : i7
}

// -----

// Check return type is still inferred correctly with generic parser
hw.module @err(in %a: i4, out out: i7) {
  // expected-error @+2 {{'comb.concat' op inferred type(s) 'i8' are incompatible with return type(s) of operation 'i7'}}
  // expected-error @+1 {{'comb.concat' op failed to infer returned types}}
  %0 = "comb.concat"(%a, %a) : (i4, i4) -> i7
  hw.output %0 : i7
}
