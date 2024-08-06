hw.module @foo(in %a : i8, out b : i8) {
  %c1_i8 = hw.constant 1 : i8
  %add = comb.add %a, %c1_i8: i8
  hw.output %add : i8
}
hw.module @top_a(in %a : i8, out b : i8) {
  %foo.b = hw.instance "foo" @foo(a: %a: i8) -> (b: i8)
  hw.output %foo.b : i8
}
