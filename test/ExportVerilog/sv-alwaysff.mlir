// RUN: circt-translate --split-input-file --export-verilog %s | FileCheck %s --check-prefix=DEFAULT
// RUN: circt-translate --lowering-options= --split-input-file --export-verilog %s | FileCheck %s --check-prefix=CLEAR
// RUN: circt-translate --lowering-options=noAlwaysFF --split-input-file --export-verilog %s | FileCheck %s --check-prefix=NOALWAYSFF

hw.module @test(%clock : i1, %cond : i1) {
  sv.alwaysff(posedge %clock) {
  }
}

// DEFAULT: always_ff @(posedge clock) begin
// DEFAULT: end // always_ff @(posedge)

// CLEAR: always_ff @(posedge clock) begin
// CLEAR: end // always_ff @(posedge)

// NOALWAYSFF: always @(posedge clock) begin
// NOALWAYSFF: end // always @(posedge)

// -----

module attributes {circt.loweringOptions = "noAlwaysFF"} {
hw.module @test(%clock : i1, %cond : i1) {
  sv.alwaysff(posedge %clock) {
  }
}
}

// DEFAULT: always @(posedge clock) begin
// DEFAULT: end // always @(posedge)

// CLEAR: always_ff @(posedge clock) begin
// CLEAR: end // always_ff @(posedge)

// NOALWAYSFF: always @(posedge clock) begin
// NOALWAYSFF: end // always @(posedge)