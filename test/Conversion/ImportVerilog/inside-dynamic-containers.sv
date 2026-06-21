// RUN: circt-translate --import-verilog %s | FileCheck %s --check-prefix=IMPORT
// RUN: circt-verilog --ir-moore %s | FileCheck %s --check-prefix=MOORE
// REQUIRES: slang
// UNSUPPORTED: valgrind

// IMPORT-LABEL: moore.module @InsideDynamicContainers
// IMPORT-DAG: moore.open_uarray_size
// IMPORT-DAG: moore.builtin.size
// IMPORT-DAG: moore.dyn_extract
// IMPORT-DAG: moore.dyn_queue_extract
// IMPORT-DAG: moore.wildcard_eq
// IMPORT-DAG: moore.or

// MOORE-LABEL: moore.module @InsideDynamicContainers
// MOORE-DAG: moore.open_uarray_size
// MOORE-DAG: moore.builtin.size
// MOORE-DAG: moore.dyn_extract
// MOORE-DAG: moore.dyn_queue_extract
// MOORE-DAG: moore.wildcard_eq
// MOORE-DAG: moore.or
module InsideDynamicContainers
    (output bit dynHit,
     output bit queueHit);
  int value;
  int dyn[];
  int q[$];

  initial begin
    dyn = new[2];
    dynHit = value inside {dyn};
    q.push_back(3);
    queueHit = value inside {q};
  end
endmodule
