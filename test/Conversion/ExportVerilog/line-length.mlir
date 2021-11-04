// RUN: circt-opt --lowering-options=emittedLineLength=40 --export-verilog %s | FileCheck %s --check-prefix=SHORT
// RUN: circt-opt --export-verilog %s | FileCheck %s --check-prefix=DEFAULT
// RUN: circt-opt --lowering-options=emittedLineLength=180 --export-verilog %s | FileCheck %s --check-prefix=LONG
// RUN: circt-opt --lowering-options=emittedLineLength=40,maximumNumberOfTokensPerExpression=60 --export-verilog %s | FileCheck %s --check-prefix=LIMIT_SHORT
// RUN: circt-opt --lowering-options=maximumNumberOfTokensPerExpression=120 --export-verilog %s | FileCheck %s --check-prefix=LIMIT_LONG

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
//                  + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a;

// LIMIT_SHORT-LABEL: module longvariadic
// LIMIT_SHORT:  wire [7:0] _tmp = a + a + a + a + a + a + a + a + a + a + a
// LIMIT_SHORT:                    + a + a + a + a + a;
// LIMIT_SHORT:  wire [7:0] _tmp_0 = a + a + a + a + a + a + a + a + a + a + a
// LIMIT_SHORT:                      + a + a + a + a + a;
// LIMIT_SHORT:  wire [7:0] _tmp_1 = a + a + a + a + a + a + a + a + a + a + a
// LIMIT_SHORT:                      + a + a + a + a + a;
// LIMIT_SHORT:  wire [7:0] _tmp_2 = a + a + a + a + a + a + a + a + a + a + a
// LIMIT_SHORT:                      + a + a + a + a + a;
// LIMIT_SHORT:  assign b = _tmp + _tmp_0 + _tmp_1 + _tmp_2;

// LIMIT_-LABEL: module longvariadic
// LIMIT_LONG:   wire [7:0] _tmp = a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a +
// LIMIT_LONG:                     a + a + a + a + a + a + a + a + a;
// LIMIT_LONG:   wire [7:0] _tmp_0 = a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a +
// LIMIT_LONG:                       a + a + a + a + a + a + a + a + a;
// LIMIT_LONG:   assign b = _tmp + _tmp_0;
