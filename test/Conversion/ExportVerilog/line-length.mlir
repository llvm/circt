// RUN: circt-opt --lowering-options=emittedLineLength=40 --export-verilog %s | FileCheck %s --check-prefix=SHORT
// RUN: circt-opt --export-verilog %s | FileCheck %s --check-prefix=DEFAULT
// RUN: circt-opt --lowering-options=emittedLineLength=180 --export-verilog %s | FileCheck %s --check-prefix=LONG
// RUN: circt-opt --lowering-options=maximumNumberOfTokensPerExpression=30 --export-verilog %s | FileCheck %s --check-prefix=LIMIT

hw.module @longvariadic(%a: i8) -> (b: i8) {
  %1 = comb.add %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a
                 : i8
  hw.output %1 : i8
}

// SHORT-LABEL: module longvariadic
// SHORT: assign b =  a + a + a + a + a + a + a + a + a + a + a
// SHORT:             + a + a + a + a + a + a + a + a + a + a +
// SHORT:             a + a + a + a + a + a + a + a + a + a + a
// SHORT:             + a + a + a + a + a + a + a + a + a + a +
// SHORT:             a + a + a + a + a + a + a + a + a + a + a
// SHORT:             + a + a + a + a + a + a + a + a + a + a +
// SHORT:             a;

// DEFAULT-LABEL: module longvariadic
// DEFAULT: assign b = a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a +
// DEFAULT:            a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a +
// DEFAULT:            a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a; 

// LONG-LABEL: module longvariadic
// LONG: assign b = a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a
//                        + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a;

// LIMIT-LABEL: module longvariadic
// LIMIT:   wire [7:0] _tmp = a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a;
// LIMIT:   wire [7:0] _tmp_0 = a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a;
// LIMIT:   wire [7:0] _tmp_1 = a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a;
// LIMIT:   wire [7:0] _tmp_2 = a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a;
// LIMIT:   assign b = _tmp + _tmp_0 + _tmp_1 + _tmp_2;
