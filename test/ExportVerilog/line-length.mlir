// RUN: circt-translate --lowering-options=emittedLineLength=40 --export-verilog %s | FileCheck %s --check-prefix=SHORT
// RUN: circt-translate  --export-verilog %s | FileCheck %s --check-prefix=DEFAULT
// RUN: circt-translate --lowering-options=emittedLineLength=180 --export-verilog %s | FileCheck %s --check-prefix=LONG

hw.module @longvariadic(%a: i8) -> (%b: i8) {
  %1 = comb.add %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
                %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a
                 : i8
  hw.output %1 : i8
}

// SHORT-LABEL: module longvariadic
// SHORT: wire [7:0] _tmp = a + a + a + a + a + a + a + a;
// SHORT: wire [7:0] _tmp_0 = a + a + a + a + a + a + a + a;
// SHORT: wire [7:0] _tmp_1 = a + a + a + a + a + a + a + a;
// SHORT: wire [7:0] _tmp_2 = a + a + a + a + a + a + a + a;
// SHORT: wire [7:0] _tmp_3 = _tmp + _tmp_0 + _tmp_1 + _tmp_2;
// SHORT: wire [7:0] _tmp_4 = a + a + a + a + a + a + a + a;
// SHORT: wire [7:0] _tmp_5 = a + a + a + a + a + a + a + a;
// SHORT: wire [7:0] _tmp_6 = a + a + a + a + a + a + a + a;
// SHORT: wire [7:0] _tmp_7 = a + a + a + a + a + a + a + a;
// SHORT: wire [7:0] _tmp_8 = _tmp_4 + _tmp_5 + _tmp_6 + _tmp_7;
// SHORT: assign b = _tmp_3 + _tmp_8;

// DEFAULT-LABEL: module longvariadic
// DEFAULT: wire [7:0] _tmp = a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a;
// DEFAULT: wire [7:0] _tmp_0 = a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a;
// DEFAULT: wire [7:0] _tmp_1 = a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a;
// DEFAULT: wire [7:0] _tmp_2 = a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a;
// DEFAULT: assign b = _tmp + _tmp_0 + _tmp_1 + _tmp_2;

// LONG-LABEL: module longvariadic
// LONG: wire [7:0] _tmp = a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a;
// LONG: wire [7:0] _tmp_0 = a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a;
// LONG: assign b = _tmp + _tmp_0;