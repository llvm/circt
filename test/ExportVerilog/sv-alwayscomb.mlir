// RUN: circt-translate --split-input-file --export-verilog %s | FileCheck %s --check-prefix=DEFAULT
// RUN: circt-translate --lowering-options= --split-input-file --export-verilog %s | FileCheck %s --check-prefix=CLEAR
// RUN: circt-translate --lowering-options=alwaysComb --split-input-file --export-verilog %s | FileCheck %s --check-prefix=ALWAYSCOMB

hw.module @test() {
  sv.alwayscomb {
  }
}

// DEFAULT: always @(*) begin
// DEFAULT: end // always @(*)

// CLEAR: always @(*) begin
// CLEAR: end // always @(*)

// ALWAYSCOMB: always_comb begin
// ALWAYSCOMB: end // always_comb

// -----

module attributes {circt.loweringOptions = "alwaysComb"} {
hw.module @test() {
  sv.alwayscomb {
  }
}
}

// DEFAULT: always_comb begin
// DEFAULT: end // always_comb

// CLEAR: always @(*) begin
// CLEAR: end // always @(*)

// ALWAYSCOMB: always_comb begin
// ALWAYSCOMB: end // always_comb
