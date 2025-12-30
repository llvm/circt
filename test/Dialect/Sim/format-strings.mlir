// RUN: circt-opt %s --canonicalize | FileCheck --strict-whitespace %s

// CHECK-LABEL: hw.module @constant_fold0
// CHECK: sim.fmt.literal ",0,0,,,;0,0, 0,0,0,0;1,1,-1,1,1,1;0011, 3, 3,3,3,03;01010,10, 10,0a,0A,12;10000000,128,-128,80,80,200;0000001100101011111110,  51966,   51966,00cafe,00CAFE,00145376"
hw.module @constant_fold0(in %zeroWitdh: i0, out res: !sim.fstring) {
  %comma = sim.fmt.literal ","
  %semicolon = sim.fmt.literal ";"

  %nocat = sim.fmt.concat ()

  %w0b = sim.fmt.bin %zeroWitdh : i0
  %w0u = sim.fmt.dec %zeroWitdh : i0
  %w0s = sim.fmt.dec %zeroWitdh signed : i0
  %w0h = sim.fmt.hex %zeroWitdh, isUpper false : i0
  %w0H = sim.fmt.hex %zeroWitdh, isUpper true : i0
  %w0o = sim.fmt.oct %zeroWitdh : i0
  %catw0 = sim.fmt.concat (%w0b, %comma, %w0u, %comma, %w0s, %comma, %w0h, %comma, %w0H, %comma, %w0o)

  %cst0_1 = hw.constant 0 : i1
  %w1b0 = sim.fmt.bin %cst0_1 : i1
  %w1u0 = sim.fmt.dec %cst0_1 : i1
  %w1s0 = sim.fmt.dec %cst0_1 signed : i1
  %w1h0 = sim.fmt.hex %cst0_1, isUpper false : i1
  %w1H0 = sim.fmt.hex %cst0_1, isUpper true : i1
  %w1o0 = sim.fmt.oct %cst0_1 : i1
  %catw1_0 = sim.fmt.concat (%w1b0, %comma, %w1u0, %comma, %w1s0, %comma, %w1h0, %comma, %w1H0, %comma, %w1o0)

  %cst1_1 = hw.constant -1 : i1
  %w1b1 = sim.fmt.bin %cst1_1 : i1
  %w1u1 = sim.fmt.dec %cst1_1 : i1
  %w1s1 = sim.fmt.dec %cst1_1 signed : i1
  %w1h1 = sim.fmt.hex %cst1_1, isUpper false : i1
  %w1H1 = sim.fmt.hex %cst1_1, isUpper true : i1
  %w1o1 = sim.fmt.oct %cst1_1 : i1
  %catw1_1 = sim.fmt.concat (%w1b1, %comma, %w1u1, %comma, %w1s1, %comma, %w1h1, %comma, %w1H1, %comma, %w1o1)

  %cst3_4 = hw.constant 3 : i4
  %w4b3 = sim.fmt.bin %cst3_4 : i4
  %w4u3 = sim.fmt.dec %cst3_4 : i4
  %w4s3 = sim.fmt.dec %cst3_4 signed : i4
  %w4h3 = sim.fmt.hex %cst3_4, isUpper false : i4
  %w4H3 = sim.fmt.hex %cst3_4, isUpper true : i4
  %w4o3 = sim.fmt.oct %cst3_4 : i4
  %catw4_3 = sim.fmt.concat (%w4b3, %comma, %w4u3, %comma, %w4s3, %comma, %w4h3, %comma, %w4H3, %comma, %w4o3)

  %cst10_5 = hw.constant 10 : i5
  %w5b10 = sim.fmt.bin %cst10_5 : i5
  %w5u10 = sim.fmt.dec %cst10_5 : i5
  %w5s10 = sim.fmt.dec %cst10_5 signed : i5
  %w5h10 = sim.fmt.hex %cst10_5, isUpper false : i5
  %w5H10 = sim.fmt.hex %cst10_5, isUpper true : i5
  %w5o10 = sim.fmt.oct %cst10_5 : i5
  %catw5_10 = sim.fmt.concat (%w5b10, %comma, %w5u10, %comma, %w5s10, %comma, %w5h10, %comma, %w5H10, %comma, %w5o10)

  %cst128_8 = hw.constant 128 : i8
  %w8b128 = sim.fmt.bin %cst128_8 : i8
  %w8u128 = sim.fmt.dec %cst128_8 : i8
  %w8s128 = sim.fmt.dec %cst128_8 signed : i8
  %w8h128 = sim.fmt.hex %cst128_8, isUpper false : i8
  %w8H128 = sim.fmt.hex %cst128_8, isUpper true : i8
  %w8o128 = sim.fmt.oct %cst128_8 : i8
  %catw8_128 = sim.fmt.concat (%w8b128, %comma, %w8u128, %comma, %w8s128, %comma, %w8h128, %comma, %w8H128, %comma, %w8o128, %nocat)

  %cstcafe_22 = hw.constant 0xcafe : i22
  %w22bcafe = sim.fmt.bin %cstcafe_22 : i22
  %w22ucafe = sim.fmt.dec %cstcafe_22 : i22
  %w22scafe = sim.fmt.dec %cstcafe_22 signed : i22
  %w22hcafe = sim.fmt.hex %cstcafe_22, isUpper false : i22
  %w22Hcafe = sim.fmt.hex %cstcafe_22, isUpper true : i22
  %w22ocafe = sim.fmt.oct %cstcafe_22 : i22
  %catw22_cafe = sim.fmt.concat (%w22bcafe, %comma, %w22ucafe, %comma, %w22scafe, %comma, %w22hcafe, %comma, %w22Hcafe, %comma, %w22ocafe)

  %catout = sim.fmt.concat (%catw0, %semicolon, %catw1_0, %semicolon, %catw1_1, %nocat, %semicolon, %catw4_3, %semicolon, %catw5_10, %semicolon, %catw8_128, %semicolon, %catw22_cafe)
  %catcatout = sim.fmt.concat (%catout)
  hw.output %catcatout : !sim.fstring
}

// CHECK-LABEL: hw.module @constant_fold1
// CHECK: sim.fmt.literal " %b: '111111111111111111111111111111111111111111111111111000110100000010010001001010111001101011110010101010110010011011001001110' %u: '10633823966279322740806214058000332366' %d: '               -4242424242424242424242' %x: '7ffffffffffff1a04895cd79559364e' %X: '7FFFFFFFFFFFF1A04895CD79559364E' %o: '77777777777777777064022112715362526233116"
hw.module @constant_fold1(out res: !sim.fstring) {
  %preb = sim.fmt.literal " %b: '"
  %preu = sim.fmt.literal " %u: '"
  %pres = sim.fmt.literal " %d: '"
  %preh = sim.fmt.literal " %x: '"
  %preH = sim.fmt.literal " %X: '"
  %preo = sim.fmt.literal " %o: '"
  %q = sim.fmt.literal "'"

  %cst42_123 = hw.constant -4242424242424242424242 : i123
  %w123b42 = sim.fmt.bin %cst42_123 : i123
  %w123u42 = sim.fmt.dec %cst42_123 : i123
  %w123s42 = sim.fmt.dec %cst42_123 signed : i123
  %w123h42 = sim.fmt.hex %cst42_123, isUpper false : i123
  %w123H42 = sim.fmt.hex %cst42_123, isUpper true : i123
  %w123o42 = sim.fmt.oct %cst42_123 : i123
  %res = sim.fmt.concat (%preb, %w123b42, %q, %preu, %w123u42, %q, %pres, %w123s42, %q, %preh, %w123h42, %q, %preH, %w123H42, %q, %preo, %w123o42)

  hw.output %res : !sim.fstring
}

// CHECK-LABEL: hw.module @constant_fold2
hw.module @constant_fold2(in %foo: i1027, out res: !sim.fstring) {
  // CHECK: [[SDS:%.+]] = sim.fmt.literal " - "
  // CHECK: [[HEX:%.+]] = sim.fmt.hex %foo, isUpper false : i1027
  // CHECK: [[CAT:%.+]] = sim.fmt.concat ([[SDS]], [[HEX]], [[SDS]])
  // CHECK: hw.output [[CAT]] : !sim.fstring

  %space = sim.fmt.literal " "
  %dash = sim.fmt.literal "-"
  %spaceDashSpace = sim.fmt.literal " - "
  %hex = sim.fmt.hex %foo, isUpper false : i1027

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

// CHECK-LABEL: hw.module @constant_fold4
// CHECK: sim.fmt.literal "  106,106  ,   106,106   ,       106,106       ,       106,106       ;006A,006A,   006A,006A   ;000152,000152,   000152,000152   ;0000000001101010,0000000001101010,   0000000001101010,0000000001101010   "
hw.module @constant_fold4(out res: !sim.fstring) {
  %1 = hw.constant 106 : i16
  %2 = hw.constant -106 : i16
  %comma = sim.fmt.literal ","
  %semicolon = sim.fmt.literal ";"

  %wrd = sim.fmt.dec %1 : i16
  %wld = sim.fmt.dec %1 isLeftAligned true : i16
  %wrds = sim.fmt.dec %1 signed : i16
  %wlds = sim.fmt.dec %1 isLeftAligned true signed: i16
  %wrdp = sim.fmt.dec %1 specifierWidth 10 : i16
  %wldp = sim.fmt.dec %1 isLeftAligned true specifierWidth 10 : i16
  %wrdps = sim.fmt.dec %1 specifierWidth 10 signed: i16
  %wldps = sim.fmt.dec %1 isLeftAligned true specifierWidth 10 signed: i16
  %dec = sim.fmt.concat (%wrd, %comma, %wld, %comma, %wrds, %comma, %wlds, %comma, %wrdp, %comma, %wldp, %comma, %wrdps, %comma, %wldps)

  %wrh = sim.fmt.hex %1, isUpper true : i16
  %wlh = sim.fmt.hex %1, isUpper true isLeftAligned true : i16
  %wrhp = sim.fmt.hex %1, isUpper true specifierWidth 7 : i16
  %wlhp = sim.fmt.hex %1, isUpper true isLeftAligned true specifierWidth 7: i16
  %hex = sim.fmt.concat (%wrh, %comma, %wlh, %comma, %wrhp, %comma, %wlhp)

  %wro = sim.fmt.oct %1 : i16
  %wlo = sim.fmt.oct %1 isLeftAligned true : i16
  %wrop = sim.fmt.oct %1 specifierWidth 9 : i16
  %wlop = sim.fmt.oct %1 isLeftAligned true specifierWidth 9: i16
  %oct = sim.fmt.concat (%wro, %comma, %wlo, %comma, %wrop, %comma, %wlop)

  %wrb = sim.fmt.bin %1 : i16
  %wlb = sim.fmt.bin %1 isLeftAligned true : i16
  %wrbp = sim.fmt.bin %1 specifierWidth 19 : i16
  %wlbp = sim.fmt.bin %1 isLeftAligned true specifierWidth 19 : i16
  %bin = sim.fmt.concat (%wrb, %comma, %wlb, %comma, %wrbp, %comma, %wlbp)

  %catout = sim.fmt.concat (%dec, %semicolon, %hex, %semicolon, %oct, %semicolon, %bin)

  hw.output %catout : !sim.fstring
}

// CHECK: sim.fmt.literal "  106,106  ,006A,006A,   006A,006A   "
hw.module @constant_fold5(out res: !sim.fstring) {
  %1 = hw.constant 106 : i16
  %2 = hw.constant -106 : i16
  %comma = sim.fmt.literal ","
  %wRightDec = sim.fmt.dec %1 : i16
  %wLeftDec = sim.fmt.dec %1 isLeftAligned true : i16
  %wRightHex = sim.fmt.hex %1, isUpper true : i16
  %wLeftHex = sim.fmt.hex %1, isUpper true isLeftAligned true : i16
  %wRightHexPad = sim.fmt.hex %1, isUpper true specifierWidth 7 : i16
  %wLeftHexPad = sim.fmt.hex %1, isUpper true isLeftAligned true specifierWidth 7: i16
  %cat = sim.fmt.concat (%wRightDec, %comma, %wLeftDec, %comma, %wRightHex, %comma, %wLeftHex, %comma, %wRightHexPad, %comma, %wLeftHexPad)
  hw.output %cat : !sim.fstring
}

// CHECK-LABEL: hw.module @constant_fold_real_formats
// CHECK: sim.fmt.literal "5.0e-01             ,0.500000,       0.5;1.6e+00,                           1.6,2;9.999990e+05,999999.000000,999999;1.000000e+126,999999999999999924867761618992882042544670869834838461439225972225294199975793026603163493762817653751530058413655532282839040.000000,1e+126"
hw.module @constant_fold_real_formats(out res: !sim.fstring) {
  %comma = sim.fmt.literal ","
  %semicolon = sim.fmt.literal ";"
  %cst_05 = arith.constant 5.000000e-01 : f64
  %e0 = sim.fmt.exp %cst_05 {fieldWidth = 20 : i32, fracDigits = 1 : i32, isLeftAligned = true} : f64
  %f0 = sim.fmt.flt %cst_05 : f64
  %g0 = sim.fmt.gen %cst_05 {fieldWidth = 10 : i32, fracDigits = 9 : i32} : f64
  %cst_05_comma = sim.fmt.concat(%e0, %comma, %f0, %comma, %g0)
  %cst_165 = arith.constant 1.650000e+00 : f64
  %e1 = sim.fmt.exp %cst_165 {fracDigits = 1 : i32} : f64
  %f1 = sim.fmt.flt %cst_165 {fieldWidth = 30 : i32, fracDigits = 1 : i32} : f64
  %g1 = sim.fmt.gen %cst_165 {fracDigits = 1 : i32} : f64
  %cst_165_comma = sim.fmt.concat(%e1, %comma, %f1, %comma, %g1)
  %cst_999999 = arith.constant 9.999990e+05 : f64
  %e2 = sim.fmt.exp %cst_999999 : f64
  %f2 = sim.fmt.flt %cst_999999 : f64
  %g2 = sim.fmt.gen %cst_999999 : f64
  %cst_999999_comma = sim.fmt.concat(%e2, %comma, %f2, %comma, %g2)
  %cst_big = arith.constant 1.000000e+126 : f64
  %e3 = sim.fmt.exp %cst_big : f64
  %f3 = sim.fmt.flt %cst_big : f64
  %g3 = sim.fmt.gen %cst_big : f64
  %cst_big_comma = sim.fmt.concat(%e3, %comma, %f3, %comma, %g3)
  %res = sim.fmt.concat(%cst_05_comma, %semicolon, %cst_165_comma, %semicolon, %cst_999999_comma, %semicolon, %cst_big_comma)
  hw.output %res : !sim.fstring
}

// CHECK-LABEL: hw.module @constant_fold_real_gen_precision
// CHECK: sim.fmt.literal "0.001,0.001,0.0012,0.00123,0.001235,0.001235,0.001235,0.001235;1e-07,1e-07,1.2e-07,1.23e-07,1.235e-07,1.235e-07,1.235e-07,1.235e-07;1e+06,1e+06,1.2e+06,1.24e+06,1.235e+06,1.235e+06,1.235e+06,1235000"
hw.module @constant_fold_real_gen_precision(out res : !sim.fstring) {
  %comma     = sim.fmt.literal ","
  %semicolon = sim.fmt.literal ";"
  %cst_a = arith.constant 1.235000e-03 : f64
  %a0 = sim.fmt.gen %cst_a {fracDigits = 0 : i32} : f64
  %a1 = sim.fmt.gen %cst_a {fracDigits = 1 : i32} : f64
  %a2 = sim.fmt.gen %cst_a {fracDigits = 2 : i32} : f64
  %a3 = sim.fmt.gen %cst_a {fracDigits = 3 : i32} : f64
  %a4 = sim.fmt.gen %cst_a {fracDigits = 4 : i32} : f64
  %a5 = sim.fmt.gen %cst_a {fracDigits = 5 : i32} : f64
  %a6 = sim.fmt.gen %cst_a {fracDigits = 6 : i32} : f64
  %a7 = sim.fmt.gen %cst_a {fracDigits = 7 : i32} : f64
  %grp_a = sim.fmt.concat(%a0, %comma, %a1, %comma, %a2, %comma, %a3, %comma, %a4, %comma, %a5, %comma, %a6, %comma, %a7)

  %cst_b = arith.constant 1.235000e-07 : f64
  %b0 = sim.fmt.gen %cst_b {fracDigits = 0 : i32} : f64
  %b1 = sim.fmt.gen %cst_b {fracDigits = 1 : i32} : f64
  %b2 = sim.fmt.gen %cst_b {fracDigits = 2 : i32} : f64
  %b3 = sim.fmt.gen %cst_b {fracDigits = 3 : i32} : f64
  %b4 = sim.fmt.gen %cst_b {fracDigits = 4 : i32} : f64
  %b5 = sim.fmt.gen %cst_b {fracDigits = 5 : i32} : f64
  %b6 = sim.fmt.gen %cst_b {fracDigits = 6 : i32} : f64
  %b7 = sim.fmt.gen %cst_b {fracDigits = 7 : i32} : f64
  %grp_b = sim.fmt.concat(%b0, %comma, %b1, %comma, %b2, %comma, %b3, %comma, %b4, %comma, %b5, %comma, %b6, %comma, %b7)

  %cst_c = arith.constant 1.235000e+06 : f64
  %c0 = sim.fmt.gen %cst_c {fracDigits = 0 : i32} : f64
  %c1 = sim.fmt.gen %cst_c {fracDigits = 1 : i32} : f64
  %c2 = sim.fmt.gen %cst_c {fracDigits = 2 : i32} : f64
  %c3 = sim.fmt.gen %cst_c {fracDigits = 3 : i32} : f64
  %c4 = sim.fmt.gen %cst_c {fracDigits = 4 : i32} : f64
  %c5 = sim.fmt.gen %cst_c {fracDigits = 5 : i32} : f64
  %c6 = sim.fmt.gen %cst_c {fracDigits = 6 : i32} : f64
  %c7 = sim.fmt.gen %cst_c {fracDigits = 7 : i32} : f64
  %grp_c = sim.fmt.concat(%c0, %comma, %c1, %comma, %c2, %comma, %c3, %comma, %c4, %comma, %c5, %comma, %c6, %comma, %c7)

  %res = sim.fmt.concat(%grp_a, %semicolon, %grp_b, %semicolon, %grp_c)
  hw.output %res : !sim.fstring
}

// CHECK-LABEL: hw.module @flatten_concat1
// CHECK-DAG:   %[[LHL:.+]] = sim.fmt.literal "HexLower: "
// CHECK-DAG:   %[[LHU:.+]] = sim.fmt.literal "HexUpper: "
// CHECK-DAG:   %[[LD:.+]] = sim.fmt.literal "Dec: "
// CHECK-DAG:   %[[LB:.+]] = sim.fmt.literal "Bin: "
// CHECK-DAG:   %[[LO:.+]] = sim.fmt.literal "Oct: "

// CHECK-DAG:   %[[FHL:.+]] = sim.fmt.hex %val, isUpper false : i8
// CHECK-DAG:   %[[FHU:.+]] = sim.fmt.hex %val, isUpper true : i8
// CHECK-DAG:   %[[FD:.+]] = sim.fmt.dec %val : i8
// CHECK-DAG:   %[[FB:.+]] = sim.fmt.bin %val : i8
// CHECK-DAG:   %[[FO:.+]] = sim.fmt.oct %val : i8

// CHECK-DAG:   %[[CAT:.+]] = sim.fmt.concat (%[[LB]], %[[FB]], %[[LD]], %[[FD]], %[[LHL]], %[[FHL]], %[[LHU]], %[[FHU]], %[[LO]], %[[FO]])
// CHECK:       hw.output %[[CAT]] : !sim.fstring

hw.module @flatten_concat1(in %val : i8, out res: !sim.fstring) {
  %binLit = sim.fmt.literal "Bin: "
  %binVal = sim.fmt.bin %val : i8
  %binCat = sim.fmt.concat (%binLit, %binVal)

  %decLit = sim.fmt.literal "Dec: "
  %decVal = sim.fmt.dec %val : i8
  %decCat = sim.fmt.concat (%decLit, %decVal, %nocat)

  %nocat = sim.fmt.concat ()

  %hexLowLit = sim.fmt.literal "HexLower: "
  %hexLowVal = sim.fmt.hex %val, isUpper false : i8
  %hexLowCat = sim.fmt.concat (%hexLowLit, %hexLowVal)

  %hexUpLit = sim.fmt.literal "HexUpper: "
  %hexUpVal = sim.fmt.hex %val, isUpper true : i8
  %hexUpCat = sim.fmt.concat (%hexUpLit, %hexUpVal)

  %octLit = sim.fmt.literal "Oct: "
  %octVal = sim.fmt.oct %val : i8
  %octCat = sim.fmt.concat (%octLit, %octVal)
  
  %catcat = sim.fmt.concat (%binCat, %nocat, %decCat, %nocat, %hexLowCat, %nocat, %hexUpCat, %nocat, %octCat)
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
