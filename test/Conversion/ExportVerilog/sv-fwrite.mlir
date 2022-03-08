// RUN: circt-opt %s -export-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

hw.module @top(%clock : i1, %reset: i1) -> () {
  sv.alwaysff(posedge %clock) {
    %0 = sv.fd stdout
    %1 = sv.fd stderr

    // CHECK: $fwrite(32'h80000001, "stdout");
    sv.fwrite %0, "stdout"

    // CHECK: $fwrite(32'h80000002, "stderr once");
    sv.fwrite %1, "stderr once"
    // CHECK: $fwrite(32'h80000002, "stderr twice");
    sv.fwrite %1, "stderr twice"
  }
}
