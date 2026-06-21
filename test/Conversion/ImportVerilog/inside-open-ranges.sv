// RUN: circt-translate --import-verilog %s | FileCheck %s --check-prefix=IMPORT
// RUN: circt-verilog --ir-moore %s | FileCheck %s --check-prefix=MOORE
// RUN: circt-verilog %s | FileCheck %s --check-prefix=FULL
// REQUIRES: slang
// UNSUPPORTED: valgrind

// IMPORT-LABEL: moore.module @InsideOpenRanges
// IMPORT: moore.sle
// IMPORT: moore.sge
// IMPORT: moore.and
// IMPORT: moore.or

// MOORE-LABEL: moore.module @InsideOpenRanges
// MOORE: moore.sle
// MOORE: moore.sge
// MOORE: moore.and
// MOORE: moore.or

// FULL-LABEL: hw.module @InsideOpenRanges
// FULL: comb.icmp slt
// FULL: comb.icmp sgt
// FULL: comb.and
// FULL: comb.or
module InsideOpenRanges
    (input int value,
     output bit found);
  initial begin
    found = value inside {[$ : 3], [7 : $], [10 : 12]};
  end
endmodule
