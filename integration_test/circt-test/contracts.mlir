// RUN: true

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
  // CHECK: test Adder4{{.*}} ... passed
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
  // CHECK: test Adder4Failed{{.*}} ... FAILED
  %0 = verif.contract %z :i4 {
    // BUG: adding a to itself is incorrect
    %1 = comb.add %a, %a : i4
    %2 = comb.icmp eq %0, %1 : i4
    verif.ensure %2 : i1
  }
  hw.output %0 : i4
}

hw.module @Mul3(in %a: i2, in %b: i2, out z: i4) {
  %a0 = comb.extract %a from 0 : (i2) -> i1
  %a1 = comb.extract %a from 1 : (i2) -> i1

  %b0 = comb.extract %b from 0 : (i2) -> i1
  %b1 = comb.extract %b from 1 : (i2) -> i1

  %a0b0 = comb.and %a0, %b0 : i1
  %a1b0 = comb.and %a1, %b0 : i1

  %a0b1 = comb.and %a0, %b1 : i1
  %a1b1 = comb.and %a1, %b1 : i1

  %false = hw.constant false
  %w = comb.concat %false, %false, %a1b0, %a0b0 : i1, i1, i1, i1
  %x = comb.concat %false, %a1b1, %a0b1, %false : i1, i1, i1, i1
  %y = hw.instance "adder" @Adder4(a: %w : i4, b: %x: i4) -> (z: i4)

  // CHECK: test Mul3{{.*}} ... passed
  %z = verif.contract %y : i4 {
    %c = comb.concat %false, %false, %a : i1, i1, i2
    %d = comb.concat %false, %false, %b : i1, i1, i2
    %e = comb.mul %c, %d : i4
    %f = comb.icmp eq %z, %e : i4
    verif.ensure %f : i1
  }
  hw.output %z : i4
}

hw.module @Mul3Failed(in %a: i2, in %b: i2, out z: i4) {
  %a0 = comb.extract %a from 0 : (i2) -> i1
  %a1 = comb.extract %a from 1 : (i2) -> i1

  %b0 = comb.extract %b from 0 : (i2) -> i1
  %b1 = comb.extract %b from 1 : (i2) -> i1

  %a0b0 = comb.and %a0, %b0 : i1
  %a1b0 = comb.and %a1, %b0 : i1

  %a0b1 = comb.and %a0, %b1 : i1
  %a1b1 = comb.and %a1, %b1 : i1

  %false = hw.constant false
  %w = comb.concat %false, %false, %a1b0, %a0b0 : i1, i1, i1, i1
  %x = comb.concat %false, %a1b1, %a0b1, %false : i1, i1, i1, i1
  %y = hw.instance "adder" @Adder4(a: %w : i4, b: %x: i4) -> (z: i4)

  // CHECK: test Mul3Failed{{.*}} ... FAILED
  %z = verif.contract %y : i4 {
    %true = hw.constant true
    // BUG: Wrong concat value
    %c = comb.concat %true, %true, %a : i1, i1, i2
    %d = comb.concat %false, %false, %b : i1, i1, i2
    %e = comb.mul %c, %d : i4
    %f = comb.icmp eq %z, %e : i4
    verif.ensure %f : i1
  }
  hw.output %z : i4
}

hw.module @HalfAdder(in %a: i1, in %b: i1, out s: i1, out co: i1) {
  %0 = comb.xor %a, %b : i1
  %1 = comb.and %a, %b : i1
  hw.output %0, %1 : i1, i1
}

hw.module @BoothMul(in %a: i4, in %b: i4, out z: i8) {
  %a0 = comb.extract %a from 0 : (i4) -> i1
  %a1 = comb.extract %a from 1 : (i4) -> i1
  %a2 = comb.extract %a from 2 : (i4) -> i1
  %a3 = comb.extract %a from 3 : (i4) -> i1
  %b0 = comb.extract %b from 0 : (i4) -> i1
  %b1 = comb.extract %b from 1 : (i4) -> i1
  %b2 = comb.extract %b from 2 : (i4) -> i1
  %b3 = comb.extract %b from 3 : (i4) -> i1

  %p0_0 = comb.and %a0, %b0 : i1
  %p0_1 = comb.and %a0, %b1 : i1
  %p0_2 = comb.and %a0, %b2 : i1
  %p0_3 = comb.and %a0, %b3 : i1

  %p1_0 = comb.and %a1, %b0 : i1
  %p1_1 = comb.and %a1, %b1 : i1
  %p1_2 = comb.and %a1, %b2 : i1
  %p1_3 = comb.and %a1, %b3 : i1

  %p2_0 = comb.and %a2, %b0 : i1
  %p2_1 = comb.and %a2, %b1 : i1
  %p2_2 = comb.and %a2, %b2 : i1
  %p2_3 = comb.and %a2, %b3 : i1

  %p3_0 = comb.and %a3, %b0 : i1
  %p3_1 = comb.and %a3, %b1 : i1
  %p3_2 = comb.and %a3, %b2 : i1
  %p3_3 = comb.and %a3, %b3 : i1

  %z_1, %c_0 = hw.instance "h0" @HalfAdder(a: %p0_1: i1, b: %p1_0: i1) -> (s: i1, co: i1)
  %s_0, %c_1 = hw.instance "h1" @HalfAdder(a: %p1_1: i1, b: %p2_0: i1) -> (s: i1, co: i1)
  %s_1, %c_2 = hw.instance "h2" @HalfAdder(a: %p2_1: i1, b: %p3_0: i1) -> (s: i1, co: i1)

  %z_2, %c_3 = hw.instance "f0" @FullAdder(a: %p0_2: i1, b: %c_0: i1, ci: %s_0: i1) -> (s: i1, co: i1)
  %s_2, %c_4 = hw.instance "f1" @FullAdder(a: %p1_2: i1, b: %c_1: i1, ci: %s_1: i1) -> (s: i1, co: i1)
  %s_3, %c_5 = hw.instance "f2" @FullAdder(a: %p2_2: i1, b: %c_2: i1, ci: %p3_1: i1) -> (s: i1, co: i1)

  %z_3, %c_6 = hw.instance "f3" @FullAdder(a: %p0_3: i1, b: %c_3: i1, ci: %s_2: i1) -> (s: i1, co: i1)
  %s_4, %c_7 = hw.instance "f4" @FullAdder(a: %p1_3: i1, b: %c_4: i1, ci: %s_3: i1) -> (s: i1, co: i1)
  %s_5, %c_8 = hw.instance "f5" @FullAdder(a: %p2_3: i1, b: %c_5: i1, ci: %p3_2: i1) -> (s: i1, co: i1)

  %z_4, %c_9 = hw.instance "h3" @HalfAdder(a: %c_6: i1, b: %s_4: i1) -> (s: i1, co: i1)
  %z_5, %c_10 = hw.instance "f6" @FullAdder(a: %c_9: i1, b: %c_7: i1, ci: %s_5: i1) -> (s: i1, co: i1)
  %z_6, %z_7 = hw.instance "f7" @FullAdder(a: %c_10: i1, b: %c_8: i1, ci: %p3_3: i1) -> (s: i1, co: i1)

  %y = comb.concat %z_7, %z_6, %z_5, %z_4, %z_3, %z_2, %z_1, %p0_0: i1, i1, i1, i1, i1, i1, i1, i1 
  // CHECK: test BoothMul{{.*}} ... passed
  %z = verif.contract %y :i8 {
    %false = hw.constant false
    %0 = comb.concat %false, %false, %false, %false, %a : i1, i1, i1, i1, i4
    %1 = comb.concat %false, %false, %false, %false, %b : i1, i1, i1, i1, i4
    %2 = comb.mul %0, %1 : i8
    %3 = comb.icmp eq %z, %2 : i8
    verif.ensure %3 : i1
  }
  hw.output %z : i8
}

hw.module @BoothMulFailed(in %a: i4, in %b: i4, out z: i8) {
  %a0 = comb.extract %a from 0 : (i4) -> i1
  %a1 = comb.extract %a from 1 : (i4) -> i1
  %a2 = comb.extract %a from 2 : (i4) -> i1
  %a3 = comb.extract %a from 3 : (i4) -> i1
  %b0 = comb.extract %b from 0 : (i4) -> i1
  %b1 = comb.extract %b from 1 : (i4) -> i1
  %b2 = comb.extract %b from 2 : (i4) -> i1
  %b3 = comb.extract %b from 3 : (i4) -> i1

  %p0_0 = comb.and %a0, %b0 : i1
  %p0_1 = comb.and %a0, %b1 : i1
  %p0_2 = comb.and %a0, %b2 : i1
  %p0_3 = comb.and %a0, %b3 : i1

  %p1_0 = comb.and %a1, %b0 : i1
  %p1_1 = comb.and %a1, %b1 : i1
  %p1_2 = comb.and %a1, %b2 : i1
  %p1_3 = comb.and %a1, %b3 : i1

  %p2_0 = comb.and %a2, %b0 : i1
  %p2_1 = comb.and %a2, %b1 : i1
  %p2_2 = comb.and %a2, %b2 : i1
  %p2_3 = comb.and %a2, %b3 : i1

  %p3_0 = comb.and %a3, %b0 : i1
  %p3_1 = comb.and %a3, %b1 : i1
  %p3_2 = comb.and %a3, %b2 : i1
  %p3_3 = comb.and %a3, %b3 : i1

  %z_1, %c_0 = hw.instance "h0" @HalfAdder(a: %p0_1: i1, b: %p1_0: i1) -> (s: i1, co: i1)
  %s_0, %c_1 = hw.instance "h1" @HalfAdder(a: %p1_1: i1, b: %p2_0: i1) -> (s: i1, co: i1)
  %s_1, %c_2 = hw.instance "h2" @HalfAdder(a: %p2_1: i1, b: %p3_0: i1) -> (s: i1, co: i1)

  %z_2, %c_3 = hw.instance "f0" @FullAdder(a: %p0_2: i1, b: %c_0: i1, ci: %s_0: i1) -> (s: i1, co: i1)
  %s_2, %c_4 = hw.instance "f1" @FullAdder(a: %p1_2: i1, b: %c_1: i1, ci: %s_1: i1) -> (s: i1, co: i1)
  %s_3, %c_5 = hw.instance "f2" @FullAdder(a: %p2_2: i1, b: %c_2: i1, ci: %p3_1: i1) -> (s: i1, co: i1)

  %z_3, %c_6 = hw.instance "f3" @FullAdder(a: %p0_3: i1, b: %c_3: i1, ci: %s_2: i1) -> (s: i1, co: i1)
  %s_4, %c_7 = hw.instance "f4" @FullAdder(a: %p1_3: i1, b: %c_4: i1, ci: %s_3: i1) -> (s: i1, co: i1)
  %s_5, %c_8 = hw.instance "f5" @FullAdder(a: %p2_3: i1, b: %c_5: i1, ci: %p3_2: i1) -> (s: i1, co: i1)

  %z_4, %c_9 = hw.instance "h3" @HalfAdder(a: %c_6: i1, b: %s_4: i1) -> (s: i1, co: i1)
  %z_5, %c_10 = hw.instance "f6" @FullAdder(a: %c_9: i1, b: %c_7: i1, ci: %s_5: i1) -> (s: i1, co: i1)
  %z_6, %z_7 = hw.instance "f7" @FullAdder(a: %c_10: i1, b: %c_8: i1, ci: %p3_3: i1) -> (s: i1, co: i1)

  %y = comb.concat %z_7, %z_6, %z_5, %z_4, %z_3, %z_2, %z_1, %p0_0: i1, i1, i1, i1, i1, i1, i1, i1 
  // CHECK: test BoothMulFailed{{.*}} ... FAILED
  %z = verif.contract %y :i8 {
    %false = hw.constant false
    %true = hw.constant true
    // BUG: Wrong concat
    %0 = comb.concat %true, %true, %true, %true, %a : i1, i1, i1, i1, i4
    %1 = comb.concat %false, %false, %false, %false, %b : i1, i1, i1, i1, i4
    %2 = comb.mul %0, %1 : i8
    %3 = comb.icmp eq %z, %2 : i8
    verif.ensure %3 : i1
  }
  hw.output %z : i8
}

// CHECK: 3 of 6 tests FAILED; 3 passed
