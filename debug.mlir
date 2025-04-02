hw.module @Foo(in %a: i42, in %b: i42, in %c: i1, out z: i42) {
  %0 = llhd.process -> i42 {
    cf.br ^bb2(%a, %b : i42, i42)
  ^bb1:
    cf.br ^bb2(%a, %b : i42, i42)
  ^bb2(%1: i42, %2: i42):
    cf.cond_br %c, ^bb3(%1 : i42), ^bb3(%2 : i42)
  ^bb3(%3: i42):
    llhd.wait yield (%3 : i42), (%a, %b, %c, %0 : i42, i42, i1, i42), ^bb1
  }
  hw.output %0 : i42
}
