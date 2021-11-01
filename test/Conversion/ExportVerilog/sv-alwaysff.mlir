// RUN: circt-opt --split-input-file --export-verilog %s | FileCheck %s --check-prefix=DEFAULT
// RUN: circt-opt --lowering-options= --split-input-file --export-verilog %s | FileCheck %s --check-prefix=CLEAR
// RUN: circt-opt --lowering-options=alwaysFF --split-input-file --export-verilog %s | FileCheck %s --check-prefix=ALWAYSFF

hw.module @test(%clock : i1, %cond : i1) {
  sv.alwaysff(posedge %clock) {
  }
}

// DEFAULT: always @(posedge clock) begin
// DEFAULT: end // always @(posedge)

// CLEAR: always @(posedge clock) begin
// CLEAR: end // always @(posedge)

// ALWAYSFF: always_ff @(posedge clock) begin
// ALWAYSFF: end // always_ff @(posedge)

// -----

module attributes {circt.loweringOptions = "alwaysFF"} {
hw.module @test(%clock : i1, %cond : i1) {
  sv.alwaysff(posedge %clock) {
  }
}
}

// DEFAULT: always_ff @(posedge clock) begin
// DEFAULT: end // always_ff @(posedge)

// CLEAR: always @(posedge clock) begin
// CLEAR: end // always @(posedge)

// ALWAYSFF: always_ff @(posedge clock) begin
// ALWAYSFF: end // always_ff @(posedge)
