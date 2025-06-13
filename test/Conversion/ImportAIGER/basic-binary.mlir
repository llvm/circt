// RUN: circt-translate --import-aiger %S/basic-binary.aig --split-input-file --verify-diagnostics | FileCheck %s
// Test is generated from the following MLIR:
// TODO: After the AIGER exporter is upstreamed, generate AIG file from this MLIR.
// hw.module @aiger_top(in %input_0 : i8196, out output_0 : i1, out output_1 : i1, in %clock: !seq.clock) {
//    %0 = comb.extract %input from 1234 : (i8196) -> i1
//    %1 = comb.extract %input from 5678 : (i8196) -> i1
//    %2 = comb.extract %input from 3 : (i8196) -> i1
//    %3 = comb.extract %input from 8193 : (i8196) -> i1
//    %and1 = and.and_inv not %1, %0 : i1
//    %and2 = and.and_inv %3, not %2 : i1
//    %reg = seq.compreg %clock, %and1 : i1
//    hw.output %and1, %and2, %reg : i1, i1
// }
// CHECK-LABEL: @aiger_top
// CHECK-NEXT:  %[[REG:.+]] = seq.compreg %[[VAL1:.+]], %clock : i1
// CHECK-NEXT:  %[[VAL0:.+]] = aig.and_inv not %input_5678, %input_1234 : i1 
// CHECK-NEXT:  %[[VAL1]] = aig.and_inv %input_8193, not %input_3 : i1
// CHECK-NEXT:  hw.output %[[VAL0]], %[[REG]] : i1, i1
