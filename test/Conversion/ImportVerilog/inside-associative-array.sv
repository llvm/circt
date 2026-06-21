// RUN: circt-translate --import-verilog %s | FileCheck %s --check-prefix=IMPORT
// RUN: circt-verilog --ir-moore %s | FileCheck %s --check-prefix=MOORE
// REQUIRES: slang
// UNSUPPORTED: valgrind

// IMPORT-LABEL: moore.module @InsideAssociativeArray
// IMPORT-DAG: moore.assoc_array.first
// IMPORT-DAG: moore.assoc_array.next
// IMPORT-DAG: moore.assoc_array_extract
// IMPORT-DAG: moore.wildcard_eq
// IMPORT-DAG: moore.or

// MOORE-LABEL: moore.module @InsideAssociativeArray
// MOORE-DAG: moore.assoc_array.first
// MOORE-DAG: moore.assoc_array.next
// MOORE-DAG: moore.assoc_array_extract
// MOORE-DAG: moore.wildcard_eq
// MOORE-DAG: moore.or
module InsideAssociativeArray
    (output bit hit);
  int value;
  int aa[int];

  initial begin
    aa[2] = 5;
    hit = value inside {aa};
  end
endmodule
