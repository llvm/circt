// RUN: true

// CHECK: 1 tests FAILED, 1 passed

hw.module @FullAdder(in %a: i1, in %b: i1, in %ci: i1, out s: i1, out co: i1) {
  %0 = comb.xor %a, %b : i1
  %1 = comb.xor %0, %ci : i1
  %2 = comb.and %a, %b : i1
  %3 = comb.and %0, %ci : i1
  %4 = comb.or %2, %3 : i1
  hw.output %1, %4 : i1, i1
}

hw.module @Add4C(in %a: i4, in %b: i4, in %ci: i1, out z: i4, out co: i1) {
  %a0 = comb.extract %a from 0 : (i4) -> i1
  %a1 = comb.extract %a from 1 : (i4) -> i1
  %a2 = comb.extract %a from 2 : (i4) -> i1
  %a3 = comb.extract %a from 3 : (i4) -> i1
  %b0 = comb.extract %b from 0 : (i4) -> i1
  %b1 = comb.extract %b from 1 : (i4) -> i1
  %b2 = comb.extract %b from 2 : (i4) -> i1
  %b3 = comb.extract %b from 3 : (i4) -> i1
  %adder0.s, %adder0.co = hw.instance "adder0" @FullAdder(a: %a0: i1, b: %b0: i1, ci: %ci: i1) -> (s: i1, co: i1)
  %adder1.s, %adder1.co = hw.instance "adder1" @FullAdder(a: %a1: i1, b: %b1: i1, ci: %adder0.co: i1) -> (s: i1, co: i1)
  %adder2.s, %adder2.co = hw.instance "adder2" @FullAdder(a: %a2: i1, b: %b2: i1, ci: %adder1.co: i1) -> (s: i1, co: i1)
  %adder3.s, %adder3.co = hw.instance "adder3" @FullAdder(a: %a3: i1, b: %b3: i1, ci: %adder2.co: i1) -> (s: i1, co: i1)
  %z = comb.concat %adder3.s, %adder2.s, %adder1.s, %adder0.s : i1, i1, i1, i1
  hw.output %z, %adder3.co : i4, i1
}

hw.module @Adder4(in %a: i4, in %b: i4, out z: i4) {
  %false = hw.constant false
  %z, %co = hw.instance "adder" @Add4C(a: %a: i4, b: %b: i4, ci: %false: i1) -> (z: i4, co: i1)
  %0 = verif.contract %z :i4 {
    %1 = comb.add %a, %b : i4
    %2 = comb.icmp eq %0, %1 : i4
    verif.ensure %2 : i1
  }
  hw.output %0 : i4
}

hw.module @Adder4Failed(in %a: i4, in %b: i4, out z: i4) {
  %false = hw.constant false
  %z, %co = hw.instance "adder" @Add4C(a: %a: i4, b: %b: i4, ci: %false: i1) -> (z: i4, co: i1)
  %0 = verif.contract %z :i4 {
    // BUG: adding a to itself is incorrect
    %1 = comb.add %a, %a : i4
    %2 = comb.icmp eq %0, %1 : i4
    verif.ensure %2 : i1
  }
  hw.output %0 : i4
}
