// RUN: circt-opt --lowering-options=emittedLineLength=40 --export-verilog %s | FileCheck %s --check-prefixes=CHECK,SHORT
// RUN: circt-opt --export-verilog %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: circt-opt --lowering-options=emittedLineLength=180 --export-verilog %s | FileCheck %s --check-prefixes=CHECK,LONG
// RUN: circt-opt --lowering-options=emittedLineLength=40,maximumNumberOfTokensPerExpression=60 --export-verilog %s | FileCheck %s --check-prefixes=CHECK,LIMIT_SHORT
// RUN: circt-opt --lowering-options=maximumNumberOfTokensPerExpression=120 --export-verilog %s | FileCheck %s --check-prefixes=CHECK,LIMIT_LONG

hw.module @longvariadic(%a: i8) -> (b: i8) {
  %1 = comb.add %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a
                 : i8
  hw.output %1 : i8
}

// CHECK-LABEL: module longvariadic

// SHORT:       assign b = a + a + a + a + a + a + a + a + a + a + a
// SHORT-NEXT:             + a + a + a + a + a + a + a + a + a + a +
// SHORT-NEXT:             a + a + a + a + a + a + a + a + a + a + a
// SHORT-NEXT:             + a + a + a + a + a + a + a + a + a + a +
// SHORT-NEXT:             a + a + a + a + a + a + a + a + a + a + a
// SHORT-NEXT:             + a + a + a + a + a + a + a + a + a + a +
// SHORT-NEXT:             a;

// DEFAULT:       assign b = a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a +
// DEFAULT-NEXT:             a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a +
// DEFAULT-NEXT:             a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a;

// LONG:       assign b = a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a
// LONG-NEXT:             + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a;

// LIMIT_SHORT:       wire [7:0] _tmp = a + a + a + a + a + a + a + a + a + a + a
// LIMIT_SHORT-NEXT:                    + a + a + a + a + a;
// LIMIT_SHORT-NEXT:  wire [7:0] _tmp_0 = a + a + a + a + a + a + a + a + a + a + a
// LIMIT_SHORT-NEXT:                      + a + a + a + a + a;
// LIMIT_SHORT-NEXT:  wire [7:0] _tmp_1 = a + a + a + a + a + a + a + a + a + a + a
// LIMIT_SHORT-NEXT:                      + a + a + a + a + a;
// LIMIT_SHORT-NEXT:  wire [7:0] _tmp_2 = a + a + a + a + a + a + a + a + a + a + a
// LIMIT_SHORT-NEXT:                      + a + a + a + a + a;
// LIMIT_SHORT-NEXT:  assign b = _tmp + _tmp_0 + _tmp_1 + _tmp_2;

// LIMIT_LONG:        wire [7:0] _tmp = a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a +
// LIMIT_LONG-NEXT:                     a + a + a + a + a + a + a + a + a;
// LIMIT_LONG-NEXT:   wire [7:0] _tmp_0 = a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a +
// LIMIT_LONG-NEXT:                       a + a + a + a + a + a + a + a + a;
// LIMIT_LONG-NEXT:   assign b = _tmp + _tmp_0;
