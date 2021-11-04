// RUN: circt-opt --lowering-options=emittedLineLength=40 --export-verilog %s | FileCheck %s --check-prefix=SHORT
// RUN: circt-opt --export-verilog %s | FileCheck %s --check-prefix=DEFAULT
// RUN: circt-opt --lowering-options=emittedLineLength=180 --export-verilog %s | FileCheck %s --check-prefix=LONG

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
// LONG-LABEL: assign b = a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a
//                        + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a;
