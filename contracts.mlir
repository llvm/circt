hw.module @HalfAdder(in %a: i1, in %b: i1, out s: i1, out co: i1) {
  %0 = comb.and bin %a, %b : i1
  %1 = comb.xor bin %a, %b : i1
  %2 = comb.concat %0, %1 : i1, i1
  %3 = verif.contract %2 : i2 {
    %false = hw.constant false
    %6 = comb.concat %false, %a : i1, i1
    %7 = comb.concat %false, %b : i1, i1
    %8 = comb.add bin %6, %7 : i2
    %9 = comb.icmp bin eq %3, %8 : i2
    verif.ensure %9 : i1
  }
  %4 = comb.extract %3 from 0 : (i2) -> i1
  %5 = comb.extract %3 from 1 : (i2) -> i1
  hw.output %4, %5 : i1, i1
}

hw.module @FullAdder(in %a: i1, in %b: i1, in %ci: i1, out s: i1, out co: i1) {
  %ha1.s, %ha1.co = hw.instance "ha1" @HalfAdder(a: %a: i1, b: %b: i1) -> (s: i1, co: i1)
  %ha2.s, %ha2.co = hw.instance "ha2" @HalfAdder(a: %ha1.s: i1, b: %ci: i1) -> (s: i1, co: i1)
  %0 = comb.or %ha1.co, %ha2.co : i1
  %1 = comb.concat %0, %ha2.s : i1, i1
  %2 = verif.contract %1 : i2 {
    %false = hw.constant false
    %5 = comb.concat %false, %a : i1, i1
    %6 = comb.concat %false, %b : i1, i1
    %7 = comb.concat %false, %ci : i1, i1
    %8 = comb.add bin %5, %6, %7 : i2
    %9 = comb.icmp bin eq %2, %8 : i2
    verif.ensure %9 : i1
  }
  %3 = comb.extract %2 from 0 : (i2) -> i1
  %4 = comb.extract %2 from 1 : (i2) -> i1
  hw.output %3, %4 : i1, i1
}

hw.module @Adder2(in %a: i2, in %b: i2, in %ci: i1, out s: i2, out co: i1) {
  %a0 = comb.extract %a from 0 : (i2) -> i1
  %a1 = comb.extract %a from 1 : (i2) -> i1
  %b0 = comb.extract %b from 0 : (i2) -> i1
  %b1 = comb.extract %b from 1 : (i2) -> i1
  %u1.s, %u1.co = hw.instance "u1" @FullAdder(a: %a0: i1, b: %b0: i1, ci: %ci: i1) -> (s: i1, co: i1)
  %u2.s, %u2.co = hw.instance "u2" @FullAdder(a: %a1: i1, b: %b1: i1, ci: %u1.co: i1) -> (s: i1, co: i1)
  %0 = comb.concat %u2.co, %u2.s, %u1.s : i1, i1, i1
  %1 = verif.contract %0 : i3 {
    %false = hw.constant false
    %c0_i2 = hw.constant 0 : i2
    %4 = comb.concat %false, %a : i1, i2
    %5 = comb.concat %false, %b : i1, i2
    %6 = comb.concat %c0_i2, %ci : i2, i1
    %7 = comb.add bin %4, %5, %6 : i3
    %8 = comb.icmp bin eq %1, %7 : i3
    verif.ensure %8 : i1
  }
  %2 = comb.extract %1 from 0 : (i3) -> i2
  %3 = comb.extract %1 from 2 : (i3) -> i1
  hw.output %2, %3 : i2, i1
}

hw.module @Adder4(in %a: i4, in %b: i4, in %ci: i1, out s: i4, out co: i1) {
  %a0 = comb.extract %a from 0 : (i4) -> i2
  %a1 = comb.extract %a from 2 : (i4) -> i2
  %b0 = comb.extract %b from 0 : (i4) -> i2
  %b1 = comb.extract %b from 2 : (i4) -> i2
  %u1.s, %u1.co = hw.instance "u1" @Adder2(a: %a0: i2, b: %b0: i2, ci: %ci: i1) -> (s: i2, co: i1)
  %u2.s, %u2.co = hw.instance "u2" @Adder2(a: %a1: i2, b: %b1: i2, ci: %u1.co: i1) -> (s: i2, co: i1)
  %0 = comb.concat %u2.co, %u2.s, %u1.s : i1, i2, i2
  %1 = verif.contract %0 : i5 {
    %false = hw.constant false
    %c0_i4 = hw.constant 0 : i4
    %4 = comb.concat %false, %a : i1, i4
    %5 = comb.concat %false, %b : i1, i4
    %6 = comb.concat %c0_i4, %ci : i4, i1
    %7 = comb.add bin %4, %5, %6 : i5
    %8 = comb.icmp bin eq %1, %7 : i5
    verif.ensure %8 : i1
  }
  %2 = comb.extract %1 from 0 : (i5) -> i4
  %3 = comb.extract %1 from 4 : (i5) -> i1
  hw.output %2, %3 : i4, i1
}

hw.module @Adder8(in %a: i8, in %b: i8, in %ci: i1, out s: i8, out co: i1) {
  %a0 = comb.extract %a from 0 : (i8) -> i4
  %a1 = comb.extract %a from 4 : (i8) -> i4
  %b0 = comb.extract %b from 0 : (i8) -> i4
  %b1 = comb.extract %b from 4 : (i8) -> i4
  %u1.s, %u1.co = hw.instance "u1" @Adder4(a: %a0: i4, b: %b0: i4, ci: %ci: i1) -> (s: i4, co: i1)
  %u2.s, %u2.co = hw.instance "u2" @Adder4(a: %a1: i4, b: %b1: i4, ci: %u1.co: i1) -> (s: i4, co: i1)
  %0 = comb.concat %u2.co, %u2.s, %u1.s : i1, i4, i4
  %1 = verif.contract %0 : i9 {
    %false = hw.constant false
    %c0_i8 = hw.constant 0 : i8
    %4 = comb.concat %false, %a : i1, i8
    %5 = comb.concat %false, %b : i1, i8
    %6 = comb.concat %c0_i8, %ci : i8, i1
    %7 = comb.add bin %4, %5, %6 : i9
    %8 = comb.icmp bin eq %1, %7 : i9
    verif.ensure %8 : i1
  }
  %2 = comb.extract %1 from 0 : (i9) -> i8
  %3 = comb.extract %1 from 8 : (i9) -> i1
  hw.output %2, %3 : i8, i1
}

hw.module @Adder16(in %a: i16, in %b: i16, in %ci: i1, out s: i16, out co: i1) {
  %a0 = comb.extract %a from 0 : (i16) -> i8
  %a1 = comb.extract %a from 8 : (i16) -> i8
  %b0 = comb.extract %b from 0 : (i16) -> i8
  %b1 = comb.extract %b from 8 : (i16) -> i8
  %u1.s, %u1.co = hw.instance "u1" @Adder8(a: %a0: i8, b: %b0: i8, ci: %ci: i1) -> (s: i8, co: i1)
  %u2.s, %u2.co = hw.instance "u2" @Adder8(a: %a1: i8, b: %b1: i8, ci: %u1.co: i1) -> (s: i8, co: i1)
  %0 = comb.concat %u2.co, %u2.s, %u1.s : i1, i8, i8
  %1 = verif.contract %0 : i17 {
    %false = hw.constant false
    %c0_i16 = hw.constant 0 : i16
    %4 = comb.concat %false, %a : i1, i16
    %5 = comb.concat %false, %b : i1, i16
    %6 = comb.concat %c0_i16, %ci : i16, i1
    %7 = comb.add bin %4, %5, %6 : i17
    %8 = comb.icmp bin eq %1, %7 : i17
    verif.ensure %8 : i1
  }
  %2 = comb.extract %1 from 0 : (i17) -> i16
  %3 = comb.extract %1 from 16 : (i17) -> i1
  hw.output %2, %3 : i16, i1
}

hw.module @Adder32(in %a: i32, in %b: i32, in %ci: i1, out s: i32, out co: i1) {
  %a0 = comb.extract %a from 0 : (i32) -> i16
  %a1 = comb.extract %a from 16 : (i32) -> i16
  %b0 = comb.extract %b from 0 : (i32) -> i16
  %b1 = comb.extract %b from 16 : (i32) -> i16
  %u1.s, %u1.co = hw.instance "u1" @Adder16(a: %a0: i16, b: %b0: i16, ci: %ci: i1) -> (s: i16, co: i1)
  %u2.s, %u2.co = hw.instance "u2" @Adder16(a: %a1: i16, b: %b1: i16, ci: %u1.co: i1) -> (s: i16, co: i1)
  %0 = comb.concat %u2.co, %u2.s, %u1.s : i1, i16, i16
  %1 = verif.contract %0 : i33 {
    %false = hw.constant false
    %c0_i32 = hw.constant 0 : i32
    %4 = comb.concat %false, %a : i1, i32
    %5 = comb.concat %false, %b : i1, i32
    %6 = comb.concat %c0_i32, %ci : i32, i1
    %7 = comb.add bin %4, %5, %6 : i33
    %8 = comb.icmp bin eq %1, %7 : i33
    verif.ensure %8 : i1
  }
  %2 = comb.extract %1 from 0 : (i33) -> i32
  %3 = comb.extract %1 from 32 : (i33) -> i1
  hw.output %2, %3 : i32, i1
}

hw.module @Adder64(in %a: i64, in %b: i64, in %ci: i1, out s: i64, out co: i1) {
  %a0 = comb.extract %a from 0 : (i64) -> i32
  %a1 = comb.extract %a from 32 : (i64) -> i32
  %b0 = comb.extract %b from 0 : (i64) -> i32
  %b1 = comb.extract %b from 32 : (i64) -> i32
  %u1.s, %u1.co = hw.instance "u1" @Adder32(a: %a0: i32, b: %b0: i32, ci: %ci: i1) -> (s: i32, co: i1)
  %u2.s, %u2.co = hw.instance "u2" @Adder32(a: %a1: i32, b: %b1: i32, ci: %u1.co: i1) -> (s: i32, co: i1)
  %0 = comb.concat %u2.co, %u2.s, %u1.s : i1, i32, i32
  %1 = verif.contract %0 : i65 {
    %false = hw.constant false
    %c0_i64 = hw.constant 0 : i64
    %4 = comb.concat %false, %a : i1, i64
    %5 = comb.concat %false, %b : i1, i64
    %6 = comb.concat %c0_i64, %ci : i64, i1
    %7 = comb.add bin %4, %5, %6 : i65
    %8 = comb.icmp bin eq %1, %7 : i65
    verif.ensure %8 : i1
  }
  %2 = comb.extract %1 from 0 : (i65) -> i64
  %3 = comb.extract %1 from 64 : (i65) -> i1
  hw.output %2, %3 : i64, i1
}

hw.module @Adder64Same(in %a: i64, in %b: i64, out z: i64) {
  %false = hw.constant false
  %dut.s, %dut.co = hw.instance "dut" @Adder64(a: %a: i64, b: %b: i64, ci: %false: i1) -> (s: i64, co: i1)
  %0 = verif.contract %dut.s : i64 {
    %1 = comb.add %a, %b : i64
    %2 = comb.icmp eq %0, %1 : i64
    verif.ensure %2 : i1
  }
  hw.output %0 : i64
}

hw.module @Multiply64x2(in %a: i64, in %b: i2, out z: i64) {
  %false = hw.constant false
  %c0_i64 = hw.constant 0 : i64
  %c1_i64 = hw.constant 1 : i64
  %0 = comb.extract %b from 0 : (i2) -> i1
  %1 = comb.extract %b from 1 : (i2) -> i1
  %2 = comb.shl %a, %c1_i64 : i64
  %3 = comb.mux %0, %a, %c0_i64 : i64
  %4 = comb.mux %1, %2, %c0_i64 : i64
  %add.z = hw.instance "add" @Adder64Same(a: %3: i64, b: %4: i64) -> (z: i64)
  %5 = verif.contract %add.z : i64 {
    %c0_i62 = hw.constant 0 : i62
    %6 = comb.concat %c0_i62, %b : i62, i2
    %7 = comb.mul %a, %6 : i64
    %8 = comb.icmp eq %5, %7 : i64
    verif.ensure %8 : i1
  }
  hw.output %5 : i64
}

// hw.module @Multiply64x4(in %a: i64, in %b: i4, out z: i64) {
//   %false = hw.constant false
//   %c2_i64 = hw.constant 2 : i64
//   %0 = comb.extract %b from 0 : (i4) -> i2
//   %1 = comb.extract %b from 2 : (i4) -> i2
//   %2 = comb.shl %a, %c2_i64 : i64
//   %u1.z = hw.instance "u1" @Multiply64x2(a: %a: i64, b: %0: i2) -> (z: i64)
//   %u2.z = hw.instance "u2" @Multiply64x2(a: %2: i64, b: %1: i2) -> (z: i64)
//   %add.z = hw.instance "add" @Adder64Same(a: %u1.z: i64, b: %u2.z: i64) -> (z: i64)
//   %3 = verif.contract %add.z : i64 {
//     %c0_i60 = hw.constant 0 : i60
//     %4 = comb.concat %c0_i60, %b : i60, i4
//     %5 = comb.mul %a, %4 : i64
//     %6 = comb.icmp eq %3, %5 : i64
//     verif.ensure %6 : i1
//   }
//   hw.output %3 : i64
// }

// hw.module @Multiply64x8(in %a: i64, in %b: i8, out z: i64) {
//   %false = hw.constant false
//   %c4_i64 = hw.constant 4 : i64
//   %0 = comb.extract %b from 0 : (i8) -> i4
//   %1 = comb.extract %b from 4 : (i8) -> i4
//   %2 = comb.shl %a, %c4_i64 : i64
//   %u1.z = hw.instance "u1" @Multiply64x4(a: %a: i64, b: %0: i4) -> (z: i64)
//   %u2.z = hw.instance "u2" @Multiply64x4(a: %2: i64, b: %1: i4) -> (z: i64)
//   %add.z = hw.instance "add" @Adder64Same(a: %u1.z: i64, b: %u2.z: i64) -> (z: i64)
//   %3 = verif.contract %add.z : i64 {
//     %c0_i56 = hw.constant 0 : i56
//     %4 = comb.concat %c0_i56, %b : i56, i8
//     %5 = comb.mul %a, %4 : i64
//     %6 = comb.icmp eq %3, %5 : i64
//     verif.ensure %6 : i1
//   }
//   hw.output %3 : i64
// }

verif.formal @Full {} {
  %a = verif.symbolic_value : i64
  %b = verif.symbolic_value : i64
  %false = hw.constant false
  %dut.s, %dut.co = hw.instance "dut" @Adder64(a: %a: i64, b: %b: i64, ci: %false: i1) -> (s: i64, co: i1)
  %0 = comb.add %a, %b : i64
  %1 = comb.icmp eq %dut.s, %0 : i64
  verif.assert %1 : i1
}
