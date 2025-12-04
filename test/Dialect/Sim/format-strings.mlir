// RUN: circt-opt %s --canonicalize | FileCheck --strict-whitespace %s

// CHECK-LABEL: hw.module @constant_fold0
// CHECK: sim.fmt.literal ",0,0,;0,0, 0,0;1,1,-1,1;0011, 3, 3,3;01010,10, 10,0a;10000000,128,-128,80;0000001100101011111110,  51966,   51966,00cafe"
hw.module @constant_fold0(in %zeroWitdh: i0, out res: !sim.fstring) {
  %comma = sim.fmt.literal ","
  %semicolon = sim.fmt.literal ";"

  %nocat = sim.fmt.concat ()

  %w0b = sim.fmt.bin %zeroWitdh : i0
  %w0u = sim.fmt.dec %zeroWitdh : i0
  %w0s = sim.fmt.dec signed %zeroWitdh : i0
  %w0h = sim.fmt.hex %zeroWitdh : i0
  %catw0 = sim.fmt.concat (%w0b, %comma, %w0u, %comma, %w0s, %comma, %w0h)

  %cst0_1 = hw.constant 0 : i1
  %w1b0 = sim.fmt.bin %cst0_1 : i1
  %w1u0 = sim.fmt.dec %cst0_1 : i1
  %w1s0 = sim.fmt.dec signed %cst0_1 : i1
  %w1h0 = sim.fmt.hex %cst0_1 : i1
  %catw1_0 = sim.fmt.concat (%w1b0, %comma, %w1u0, %comma, %w1s0, %comma, %w1h0)

  %cst1_1 = hw.constant -1 : i1
  %w1b1 = sim.fmt.bin %cst1_1 : i1
  %w1u1 = sim.fmt.dec %cst1_1 : i1
  %w1s1 = sim.fmt.dec signed %cst1_1 : i1
  %w1h1 = sim.fmt.hex %cst1_1 : i1
  %catw1_1 = sim.fmt.concat (%w1b1, %comma, %w1u1, %comma, %w1s1, %comma, %w1h1)

  %cst3_4 = hw.constant 3 : i4
  %w4b3 = sim.fmt.bin %cst3_4 : i4
  %w4u3 = sim.fmt.dec %cst3_4 : i4
  %w4s3 = sim.fmt.dec signed %cst3_4 : i4
  %w4h3 = sim.fmt.hex %cst3_4 : i4
  %catw4_3 = sim.fmt.concat (%w4b3, %comma, %w4u3, %comma, %w4s3, %comma, %w4h3)

  %cst10_5 = hw.constant 10 : i5
  %w5b10 = sim.fmt.bin %cst10_5 : i5
  %w5u10 = sim.fmt.dec %cst10_5 : i5
  %w5s10 = sim.fmt.dec signed %cst10_5 : i5
  %w5h10 = sim.fmt.hex %cst10_5 : i5
  %catw5_10 = sim.fmt.concat (%w5b10, %comma, %w5u10, %comma, %w5s10, %comma, %w5h10)

  %cst128_8 = hw.constant 128 : i8
  %w8b128 = sim.fmt.bin %cst128_8 : i8
  %w8u128 = sim.fmt.dec %cst128_8 : i8
  %w8s128 = sim.fmt.dec signed %cst128_8 : i8
  %w8h128 = sim.fmt.hex %cst128_8 : i8
  %catw8_128 = sim.fmt.concat (%w8b128, %comma, %w8u128, %comma, %w8s128, %comma, %w8h128, %nocat)

  %cstcafe_22 = hw.constant 0xcafe : i22
  %w22bcafe = sim.fmt.bin %cstcafe_22 : i22
  %w22ucafe = sim.fmt.dec %cstcafe_22 : i22
  %w22scafe = sim.fmt.dec signed %cstcafe_22 : i22
  %w22hcafe = sim.fmt.hex %cstcafe_22 : i22
  %catw22_cafe = sim.fmt.concat (%w22bcafe, %comma, %w22ucafe, %comma, %w22scafe, %comma, %w22hcafe)

  %catout = sim.fmt.concat (%catw0, %semicolon, %catw1_0, %semicolon, %catw1_1, %nocat, %semicolon, %catw4_3, %semicolon, %catw5_10, %semicolon, %catw8_128, %semicolon, %catw22_cafe)
  %catcatout = sim.fmt.concat (%catout)
  hw.output %catcatout : !sim.fstring
}

// CHECK-LABEL: hw.module @constant_fold1
// CHECK: sim.fmt.literal " %b: '111111111111111111111111111111111111111111111111111000110100000010010001001010111001101011110010101010110010011011001001110' %u: '10633823966279322740806214058000332366' %d: '               -4242424242424242424242' %x: '7ffffffffffff1a04895cd79559364e'"
hw.module @constant_fold1(out res: !sim.fstring) {
  %preb = sim.fmt.literal " %b: '"
  %preu = sim.fmt.literal " %u: '"
  %pres = sim.fmt.literal " %d: '"
  %preh = sim.fmt.literal " %x: '"
  %q = sim.fmt.literal "'"

  %cst42_123 = hw.constant -4242424242424242424242 : i123
  %w123b42 = sim.fmt.bin %cst42_123 : i123
  %w123u42 = sim.fmt.dec %cst42_123 : i123
  %w123s42 = sim.fmt.dec signed %cst42_123 : i123
  %w123h42 = sim.fmt.hex %cst42_123 : i123
  %res = sim.fmt.concat (%preb, %w123b42, %q, %preu, %w123u42, %q, %pres, %w123s42, %q, %preh, %w123h42, %q)

  hw.output %res : !sim.fstring
}

// CHECK-LABEL: hw.module @constant_fold2
hw.module @constant_fold2(in %foo: i1027, out res: !sim.fstring) {
  // CHECK: [[SDS:%.+]] = sim.fmt.literal " - "
  // CHECK: [[HEX:%.+]] = sim.fmt.hex %foo : i1027
  // CHECK: [[CAT:%.+]] = sim.fmt.concat ([[SDS]], [[HEX]], [[SDS]])
  // CHECK: hw.output [[CAT]] : !sim.fstring

  %space = sim.fmt.literal " "
  %dash = sim.fmt.literal "-"
  %spaceDashSpace = sim.fmt.literal " - "
  %hex = sim.fmt.hex %foo : i1027

  %res = sim.fmt.concat (%spaceDashSpace, %hex, %space, %dash, %space)
  hw.output %res : !sim.fstring
}

// CHECK-LABEL: hw.module @constant_fold3
// CHECK: sim.fmt.literal "Foo\0A\0D\00Foo\00\C8"
hw.module @constant_fold3(in %zeroWitdh: i0, out res: !sim.fstring) {
  %F = hw.constant 70 : i7
  %o = hw.constant 111 : i8
  %cr = hw.constant 13 : i4
  %lf = hw.constant 10 : i5
  %ext = hw.constant 200: i8

  %cF = sim.fmt.char %F : i7
  %co = sim.fmt.char %o : i8
  %ccr = sim.fmt.char %cr : i4
  %clf = sim.fmt.char %lf : i5
  %cext = sim.fmt.char %ext : i8

  %null = sim.fmt.char %zeroWitdh : i0

  %foo = sim.fmt.concat (%cF, %co, %co)
  %cat = sim.fmt.concat (%foo, %clf, %ccr, %null, %foo, %null, %cext)
  hw.output %cat : !sim.fstring
}


// CHECK-LABEL: hw.module @flatten_concat1
// CHECK-DAG:   %[[LH:.+]] = sim.fmt.literal "Hex: "
// CHECK-DAG:   %[[LD:.+]] = sim.fmt.literal "Dec: "
// CHECK-DAG:   %[[LB:.+]] = sim.fmt.literal "Bin: "
// CHECK-DAG:   %[[FH:.+]] = sim.fmt.hex %val : i8
// CHECK-DAG:   %[[FD:.+]] = sim.fmt.dec %val : i8
// CHECK-DAG:   %[[FB:.+]] = sim.fmt.bin %val : i8
// CHECK-DAG:   %[[CAT:.+]] = sim.fmt.concat (%[[LB]], %[[FB]], %[[LD]], %[[FD]], %[[LH]], %[[FH]])
// CHECK:       hw.output %[[CAT]] : !sim.fstring

hw.module @flatten_concat1(in %val : i8, out res: !sim.fstring) {
  %binLit = sim.fmt.literal "Bin: "
  %binVal = sim.fmt.bin %val : i8
  %binCat = sim.fmt.concat (%binLit, %binVal)

  %decLit = sim.fmt.literal "Dec: "
  %decVal = sim.fmt.dec %val : i8
  %decCat = sim.fmt.concat (%decLit, %decVal, %nocat)

  %nocat = sim.fmt.concat ()

  %hexLit = sim.fmt.literal "Hex: "
  %hexVal = sim.fmt.hex %val : i8
  %hexCat = sim.fmt.concat (%hexLit, %hexVal)

  %catcat = sim.fmt.concat (%binCat, %nocat, %decCat, %nocat, %hexCat)
  hw.output %catcat : !sim.fstring
}

// CHECK-LABEL: hw.module @flatten_concat2
// CHECK-DAG:   %[[F:.+]] = sim.fmt.literal "Foo"
// CHECK-DAG:   %[[FF:.+]] = sim.fmt.literal "FooFoo"
// CHECK-DAG:   %[[CHR:.+]] = sim.fmt.char %val : i8
// CHECK-DAG:   %[[CAT:.+]] = sim.fmt.concat (%[[F]], %[[CHR]], %[[FF]], %[[CHR]], %[[FF]], %[[CHR]], %[[FF]], %[[CHR]], %[[FF]], %[[CHR]], %[[F]])
// CHECK:       hw.output %[[CAT]] : !sim.fstring

hw.module @flatten_concat2(in %val : i8, out res: !sim.fstring) {
  %foo = sim.fmt.literal "Foo"
  %char = sim.fmt.char %val : i8

  %c = sim.fmt.concat (%foo, %char, %foo)
  %cc = sim.fmt.concat (%c, %c)
  %ccccc = sim.fmt.concat (%cc, %c, %cc)

  hw.output %ccccc : !sim.fstring
}
