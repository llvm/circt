// RUN: circt-opt --lowering-options=emittedLineLength=40 --export-verilog %s | FileCheck %s --check-prefixes=CHECK,SHORT
// RUN: circt-opt --export-verilog %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: circt-opt --lowering-options=emittedLineLength=180 --export-verilog %s | FileCheck %s --check-prefixes=CHECK,LONG
// RUN: circt-opt --lowering-options=emittedLineLength=40,maximumNumberOfTermsPerExpression=16 --export-verilog %s | FileCheck %s --check-prefixes=CHECK,LIMIT_SHORT
// RUN: circt-opt --lowering-options=maximumNumberOfTermsPerExpression=32 --export-verilog %s | FileCheck %s --check-prefixes=CHECK,LIMIT_LONG

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

// LIMIT_SHORT:       wire [7:0] _GEN;
// LIMIT_SHORT:       assign _GEN = a + a + a + a + a + a + a + a + a + a + a
// LIMIT_SHORT-NEXT:                + a + a + a + a + a + a + a + a + a + a +
// LIMIT_SHORT-NEXT:                a + a + a + a + a + a + a + a + a + a + a
// LIMIT_SHORT-NEXT:                + a + a + a + a + a + a + a + a + a + a +
// LIMIT_SHORT-NEXT:                a + a + a + a + a + a + a + a + a + a + a
// LIMIT_SHORT-NEXT:                + a + a + a + a + a + a + a + a + a + a +
// LIMIT_SHORT-NEXT:                a;
// LIMIT_SHORT:       assign b = _GEN;

// LIMIT_LONG:        wire [7:0] _GEN;
// LIMIT_LONG:        assign _GEN = a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a +
// LIMIT_LONG-NEXT:                 a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a +
// LIMIT_LONG-NEXT:                 a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a + a;
// LIMIT_LONG:        assign b = _GEN;

hw.module @moduleWithComment()
  attributes {comment = "The quick brown fox jumps over the lazy dog.  The quick brown fox jumps over the lazy dog.\naaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"} {}

// SHORT-LABEL:   // The quick brown fox jumps over the
// SHORT-NEXT:    // lazy dog.  The quick brown fox jumps
// SHORT-NEXT:    // over the lazy dog.
// SHORT-NEXT:    // aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
// SHORT-NEXT:    module moduleWithComment
//
// DEFAULT-LABEL: // The quick brown fox jumps over the lazy dog.  The quick brown fox jumps over the lazy
// DEFAULT-NEXT:  // dog.
// DEFAULT-NEXT:  // aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
// DEFAULT-NEXT:  module moduleWithComment
//
// LONG-LABEL:    // The quick brown fox jumps over the lazy dog.  The quick brown fox jumps over the lazy dog.
// LONG-NEXT:     // aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
// LONG-NEXT:     module moduleWithComment
