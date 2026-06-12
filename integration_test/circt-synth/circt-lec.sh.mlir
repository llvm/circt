// REQUIRES: z3

// Positive test: equivalent modules should return 0.
// RUN: circt-lec.sh %s %s -c1=foo -c2=foo

// Negative test: non-equivalent modules should return non-zero.
// RUN: not circt-lec.sh %s %s -c1=foo -c2=bar

hw.module @foo(in %a : i8, in %b : i8, out c : i8) {
  %add = comb.add %a, %b : i8
  hw.output %add : i8
}

hw.module @bar(in %a : i8, in %b : i8, out c : i8) {
  %add = comb.add %b, %a : i8
  %x = comb.add %add, %a : i8
  hw.output %x : i8
}
