hw.module @basic(in %in: i4, in %in2: i4, out out: i4) {
  %0 = comb.add %in, %in2: i4
  hw.output %0 : i4
}

hw.module @counter(in %clk: i1, in %srst: i1, out o: i4) {
  %seq_clk = seq.to_clock %clk

  %0 = hw.constant 0: i4
  %reg = seq.firreg %added clock %seq_clk reset sync %srst, %0 : i4

  %one = hw.constant 1 : i4
  %added = comb.add %reg, %one : i4

  hw.output %reg : i4
}
