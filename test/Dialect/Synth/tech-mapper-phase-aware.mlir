// RUN: split-file %s %t
// RUN: circt-opt --pass-pipeline='builtin.module(synth-tech-mapper{strategy=area test=true})' %t/mapping.mlir --split-input-file | FileCheck %s --check-prefixes=CHECK,AREA
// RUN: circt-opt --pass-pipeline='builtin.module(synth-tech-mapper{strategy=timing test=true})' %t/mapping.mlir --split-input-file | FileCheck %s --check-prefixes=CHECK,TIMING
// RUN: circt-opt --pass-pipeline='builtin.module(synth-tech-mapper{strategy=area})' %t/no-inverter.mlir --split-input-file --verify-diagnostics
// RUN: circt-opt --pass-pipeline='builtin.module(synth-tech-mapper{strategy=timing})' %t/no-inverter.mlir --split-input-file --verify-diagnostics

//--- mapping.mlir

// A phase-adjusted candidate contains one library cell and an actual inverter
// for every required input or output phase. INV costs area 2 and delay 3; AND
// costs area 1 and delay 4. Only the final operation receives the cut's arrival
// time annotation, which includes all of these cells.
hw.module @and_gate(in %a: i1, in %b: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<4, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<4, 0, #synth.polarity<positive>>]>} {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}
hw.module @inv_gate(in %a: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 2.0 : f64, arcs = [#synth.linear_timing_arc<3, 0, #synth.polarity<negative>>]>} {
  %0 = synth.aig.and_inv not %a : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @direct_and(
// CHECK-NEXT: %[[DIRECT:.*]] = hw.instance {{.*}} @and_gate(a: %a: i1, b: %b: i1) -> (y: i1) {test.arrival_times = [4]}
// CHECK-NEXT: hw.output %[[DIRECT]] : i1
hw.module @direct_and(in %a: i1, in %b: i1, out y: i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @input_phase(
// CHECK-NEXT: %[[INPUT_INV:.*]] = hw.instance {{.*}} @inv_gate(a: %a: i1) -> (y: i1)
// CHECK-NEXT: %[[INPUT_AND:.*]] = hw.instance {{.*}} @and_gate(a: %b: i1, b: %[[INPUT_INV]]: i1) -> (y: i1) {test.arrival_times = [7]}
// CHECK-NEXT: hw.output %[[INPUT_AND]] : i1
hw.module @input_phase(in %a: i1, in %b: i1, out y: i1) {
  %0 = synth.aig.and_inv not %a, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @output_phase(
// CHECK-NEXT: %[[OUTPUT_AND:.*]] = hw.instance {{.*}} @and_gate(a: %a: i1, b: %b: i1) -> (y: i1)
// CHECK-NEXT: %[[OUTPUT_INV:.*]] = hw.instance {{.*}} @inv_gate(a: %[[OUTPUT_AND]]: i1) -> (y: i1) {test.arrival_times = [7]}
// CHECK-NEXT: hw.output %[[OUTPUT_INV]] : i1
hw.module @output_phase(in %a: i1, in %b: i1, out y: i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv not %0 : i1
  hw.output %1 : i1
}

// CHECK-LABEL: hw.module @both_input_phases(
// CHECK-DAG: %[[BOTH_A:.*]] = hw.instance {{.*}} @inv_gate(a: %a: i1) -> (y: i1)
// CHECK-DAG: %[[BOTH_B:.*]] = hw.instance {{.*}} @inv_gate(a: %b: i1) -> (y: i1)
// CHECK-NEXT: %[[BOTH_AND:.*]] = hw.instance {{.*}} @and_gate(a: %[[BOTH_A]]: i1, b: %[[BOTH_B]]: i1) -> (y: i1) {test.arrival_times = [7]}
// CHECK-NEXT: hw.output %[[BOTH_AND]] : i1
hw.module @both_input_phases(in %a: i1, in %b: i1, out y: i1) {
  %0 = synth.aig.and_inv not %a, not %b : i1
  hw.output %0 : i1
}

// Each candidate pays for and creates its own inverter, even for the same SSA
// input. Cross-candidate sharing would require a different area cost model.
// CHECK-LABEL: hw.module @candidate_local_inverters(
// CHECK-NEXT: %[[LOCAL_A:.*]] = hw.instance {{.*}} @inv_gate(a: %a: i1) -> (y: i1)
// CHECK-NEXT: %[[LOCAL_Y:.*]] = hw.instance {{.*}} @and_gate(a: %b: i1, b: %[[LOCAL_A]]: i1) -> (y: i1) {test.arrival_times = [7]}
// CHECK-NEXT: %[[LOCAL_B:.*]] = hw.instance {{.*}} @inv_gate(a: %a: i1) -> (y: i1)
// CHECK-NEXT: %[[LOCAL_Z:.*]] = hw.instance {{.*}} @and_gate(a: %c: i1, b: %[[LOCAL_B]]: i1) -> (y: i1) {test.arrival_times = [7]}
// CHECK-NEXT: hw.output %[[LOCAL_Y]], %[[LOCAL_Z]] : i1, i1
hw.module @candidate_local_inverters(in %a: i1, in %b: i1, in %c: i1, out y: i1, out z: i1) {
  %0 = synth.aig.and_inv not %a, %b : i1
  %1 = synth.aig.and_inv not %a, %c : i1
  hw.output %0, %1 : i1, i1
}

// An input adapter must also include the nonzero arrival time of its source.
// CHECK-LABEL: hw.module @inverted_internal(
// CHECK-NEXT: %[[INTERNAL:.*]] = hw.instance {{.*}} @and_gate(a: %a: i1, b: %b: i1) -> (y: i1) {test.arrival_times = [4]}
// CHECK-NEXT: %[[INTERNAL_INV:.*]] = hw.instance {{.*}} @inv_gate(a: %[[INTERNAL]]: i1) -> (y: i1)
// CHECK-NEXT: %[[INTERNAL_RESULT:.*]] = hw.instance {{.*}} @and_gate(a: %c: i1, b: %[[INTERNAL_INV]]: i1) -> (y: i1) {test.arrival_times = [11]}
// CHECK-NEXT: hw.output %[[INTERNAL_RESULT]] : i1
hw.module @inverted_internal(in %a: i1, in %b: i1, in %c: i1, out y: i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv not %0, %c : i1
  hw.output %1 : i1
}

// The previously unmatched input phase must not prevent arrival propagation
// to an otherwise directly mappable output inverter.
// CHECK-LABEL: hw.module @downstream_inverter(
// CHECK-NEXT: %[[DOWN_INPUT:.*]] = hw.instance {{.*}} @inv_gate(a: %a: i1) -> (y: i1)
// CHECK-NEXT: %[[DOWN_AND:.*]] = hw.instance {{.*}} @and_gate(a: %b: i1, b: %[[DOWN_INPUT]]: i1) -> (y: i1) {test.arrival_times = [7]}
// CHECK-NEXT: %[[DOWN_OUTPUT:.*]] = hw.instance {{.*}} @inv_gate(a: %[[DOWN_AND]]: i1) -> (y: i1) {test.arrival_times = [10]}
// CHECK-NEXT: hw.output %[[DOWN_OUTPUT]] : i1
hw.module @downstream_inverter(in %a: i1, in %b: i1, out y: i1) {
  %0 = synth.aig.and_inv not %a, %b : i1
  %1 = synth.aig.and_inv not %0 : i1
  hw.output %1 : i1
}

// A cut can absorb both inversions without creating physical INV cells.
// CHECK-LABEL: hw.module @double_inversion(
// CHECK-NEXT: %[[DOUBLE:.*]] = hw.instance {{.*}} @and_gate(a: %a: i1, b: %b: i1) -> (y: i1) {test.arrival_times = [4]}
// CHECK-NEXT: hw.output %[[DOUBLE]] : i1
hw.module @double_inversion(in %a: i1, in %b: i1, out y: i1) {
  %0 = synth.aig.and_inv not %a : i1
  %1 = synth.aig.and_inv not %0, %b : i1
  hw.output %1 : i1
}

// !false & !a reduces to !a; no tie cell or constant-driven INV is needed.
// CHECK-LABEL: hw.module @inverted_constant(
// CHECK: %[[CONSTANT_INV:.*]] = hw.instance {{.*}} @inv_gate(a: %a: i1) -> (y: i1) {test.arrival_times = [3]}
// CHECK-NEXT: hw.output %[[CONSTANT_INV]] : i1
hw.module @inverted_constant(in %a: i1, out y: i1) {
  %false = hw.constant false
  %0 = synth.aig.and_inv not %false, not %a : i1
  hw.output %0 : i1
}

// -----

// A complete mapping must not depend on a positive AND primitive. NAND+INV
// implements AND through output phase, and a mixed-input AND needs both an
// input and an output inverter: total delay 3 + 5 + 3 = 11.
hw.module @nand_gate(in %a: i1, in %b: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<5, 0, #synth.polarity<negative>>, #synth.linear_timing_arc<5, 0, #synth.polarity<negative>>]>} {
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv not %0 : i1
  hw.output %1 : i1
}
hw.module @inv_gate(in %a: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 2.0 : f64, arcs = [#synth.linear_timing_arc<3, 0, #synth.polarity<negative>>]>} {
  %0 = synth.aig.and_inv not %a : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @and_from_nand(
// CHECK-NEXT: %[[NAND:.*]] = hw.instance {{.*}} @nand_gate(a: %a: i1, b: %b: i1) -> (y: i1)
// CHECK-NEXT: %[[NAND_INV:.*]] = hw.instance {{.*}} @inv_gate(a: %[[NAND]]: i1) -> (y: i1) {test.arrival_times = [8]}
// CHECK-NEXT: hw.output %[[NAND_INV]] : i1
hw.module @and_from_nand(in %a: i1, in %b: i1, out y: i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @input_and_output_phase(
// CHECK-NEXT: %[[NAND_INPUT:.*]] = hw.instance {{.*}} @inv_gate(a: %a: i1) -> (y: i1)
// CHECK-NEXT: %[[NAND_CELL:.*]] = hw.instance {{.*}} @nand_gate(a: %b: i1, b: %[[NAND_INPUT]]: i1) -> (y: i1)
// CHECK-NEXT: %[[NAND_OUTPUT:.*]] = hw.instance {{.*}} @inv_gate(a: %[[NAND_CELL]]: i1) -> (y: i1) {test.arrival_times = [11]}
// CHECK-NEXT: hw.output %[[NAND_OUTPUT]] : i1
hw.module @input_and_output_phase(in %a: i1, in %b: i1, out y: i1) {
  %0 = synth.aig.and_inv not %a, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @direct_nand(
// CHECK-NEXT: %[[DIRECT_NAND:.*]] = hw.instance {{.*}} @nand_gate(a: %a: i1, b: %b: i1) -> (y: i1) {test.arrival_times = [5]}
// CHECK-NEXT: hw.output %[[DIRECT_NAND]] : i1
hw.module @direct_nand(in %a: i1, in %b: i1, out y: i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv not %0 : i1
  hw.output %1 : i1
}

// -----

// Inverters are not free. For NOR, AND+2*INV has area 5, delay 4 versus the
// direct NOR's area 4, delay 8. For mixed AND, AND+INV has area 3, delay 4
// versus the direct cell's area 2.5, delay 6. Area and timing must disagree.
hw.module @and_gate(in %a: i1, in %b: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>]>} {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}
hw.module @inv_gate(in %a: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 2.0 : f64, arcs = [#synth.linear_timing_arc<3, 0, #synth.polarity<negative>>]>} {
  %0 = synth.aig.and_inv not %a : i1
  hw.output %0 : i1
}
hw.module @nor_gate(in %a: i1, in %b: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 4.0 : f64, arcs = [#synth.linear_timing_arc<8, 0, #synth.polarity<negative>>, #synth.linear_timing_arc<8, 0, #synth.polarity<negative>>]>} {
  %0 = synth.aig.and_inv not %a, not %b : i1
  hw.output %0 : i1
}
hw.module @and_not_gate(in %a: i1, in %b: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 2.5 : f64, arcs = [#synth.linear_timing_arc<6, 0, #synth.polarity<negative>>, #synth.linear_timing_arc<6, 0, #synth.polarity<positive>>]>} {
  %0 = synth.aig.and_inv not %a, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @charge_two_inverters(
// AREA-NEXT: %[[AREA_NOR:.*]] = hw.instance {{.*}} @nor_gate(a: %a: i1, b: %b: i1) -> (y: i1) {test.arrival_times = [8]}
// AREA-NEXT: hw.output %[[AREA_NOR]] : i1
// TIMING-DAG: %[[TIMING_A:.*]] = hw.instance {{.*}} @inv_gate(a: %a: i1) -> (y: i1)
// TIMING-DAG: %[[TIMING_B:.*]] = hw.instance {{.*}} @inv_gate(a: %b: i1) -> (y: i1)
// TIMING-NEXT: %[[TIMING_NOR:.*]] = hw.instance {{.*}} @and_gate(a: %[[TIMING_A]]: i1, b: %[[TIMING_B]]: i1) -> (y: i1) {test.arrival_times = [4]}
// TIMING-NEXT: hw.output %[[TIMING_NOR]] : i1
hw.module @charge_two_inverters(in %a: i1, in %b: i1, out y: i1) {
  %0 = synth.aig.and_inv not %a, not %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @charge_one_inverter(
// AREA-NEXT: %[[AREA_MIXED:.*]] = hw.instance {{.*}} @and_not_gate(a: %a: i1, b: %b: i1) -> (y: i1) {test.arrival_times = [6]}
// AREA-NEXT: hw.output %[[AREA_MIXED]] : i1
// TIMING-NEXT: %[[TIMING_INV:.*]] = hw.instance {{.*}} @inv_gate(a: %a: i1) -> (y: i1)
// TIMING-NEXT: %[[TIMING_MIXED:.*]] = hw.instance {{.*}} @and_gate(a: %b: i1, b: %[[TIMING_INV]]: i1) -> (y: i1) {test.arrival_times = [4]}
// TIMING-NEXT: hw.output %[[TIMING_MIXED]] : i1
hw.module @charge_one_inverter(in %a: i1, in %b: i1, out y: i1) {
  %0 = synth.aig.and_inv not %a, %b : i1
  hw.output %0 : i1
}

// -----

// Different physical INV implementations must retain their own area/delay.
// Both input and output phase costs use the implementation actually emitted.
hw.module @and_gate(in %a: i1, in %b: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<4, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<4, 0, #synth.polarity<positive>>]>} {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}
hw.module @inv_small(in %a: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<7, 0, #synth.polarity<negative>>]>} {
  %0 = synth.aig.and_inv not %a : i1
  hw.output %0 : i1
}
hw.module @inv_fast(in %a: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 4.0 : f64, arcs = [#synth.linear_timing_arc<2, 0, #synth.polarity<negative>>]>} {
  %0 = synth.aig.and_inv not %a : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @input_inverter_choice(
// AREA-NEXT: %[[SMALL_INPUT:.*]] = hw.instance {{.*}} @inv_small(a: %a: i1) -> (y: i1)
// AREA-NEXT: %[[SMALL_AND:.*]] = hw.instance {{.*}} @and_gate(a: %b: i1, b: %[[SMALL_INPUT]]: i1) -> (y: i1) {test.arrival_times = [11]}
// AREA-NEXT: hw.output %[[SMALL_AND]] : i1
// TIMING-NEXT: %[[FAST_INPUT:.*]] = hw.instance {{.*}} @inv_fast(a: %a: i1) -> (y: i1)
// TIMING-NEXT: %[[FAST_AND:.*]] = hw.instance {{.*}} @and_gate(a: %b: i1, b: %[[FAST_INPUT]]: i1) -> (y: i1) {test.arrival_times = [6]}
// TIMING-NEXT: hw.output %[[FAST_AND]] : i1
hw.module @input_inverter_choice(in %a: i1, in %b: i1, out y: i1) {
  %0 = synth.aig.and_inv not %a, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @output_inverter_choice(
// CHECK-NEXT: %[[DRIVE_AND:.*]] = hw.instance {{.*}} @and_gate(a: %a: i1, b: %b: i1) -> (y: i1)
// AREA-NEXT: %[[SMALL_OUTPUT:.*]] = hw.instance {{.*}} @inv_small(a: %[[DRIVE_AND]]: i1) -> (y: i1) {test.arrival_times = [11]}
// AREA-NEXT: hw.output %[[SMALL_OUTPUT]] : i1
// TIMING-NEXT: %[[FAST_OUTPUT:.*]] = hw.instance {{.*}} @inv_fast(a: %[[DRIVE_AND]]: i1) -> (y: i1) {test.arrival_times = [6]}
// TIMING-NEXT: hw.output %[[FAST_OUTPUT]] : i1
hw.module @output_inverter_choice(in %a: i1, in %b: i1, out y: i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv not %0 : i1
  hw.output %1 : i1
}

// -----

// No inverter is needed for a direct match. Pin delays follow the actual
// permutation: !a&b has a slow negative pin (9) and a fast positive pin (1).
// The root receives the value arriving at time 4 at its negative pin and a
// primary input at its positive pin, so it arrives at max(4+9, 0+1) = 13, not 9.
hw.module @and_gate(in %a: i1, in %b: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<4, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<4, 0, #synth.polarity<positive>>]>} {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}
hw.module @asymmetric_gate(in %a: i1, in %b: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<9, 0, #synth.polarity<negative>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>]>} {
  %0 = synth.aig.and_inv not %a, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @permuted_pin_delays(
// CHECK-NEXT: %[[LATE:.*]] = hw.instance {{.*}} @and_gate(a: %c: i1, b: %d: i1) -> (y: i1) {test.arrival_times = [4]}
// CHECK-NEXT: %[[ASYMMETRIC:.*]] = hw.instance {{.*}} @asymmetric_gate(a: %[[LATE]]: i1, b: %b: i1) -> (y: i1) {test.arrival_times = [13]}
// CHECK-NEXT: hw.output %[[ASYMMETRIC]] : i1
hw.module @permuted_pin_delays(in %b: i1, in %c: i1, in %d: i1, out y: i1) {
  %0 = synth.aig.and_inv %c, %d : i1
  %1 = synth.aig.and_inv not %0, %b : i1
  hw.output %1 : i1
}

// CHECK-LABEL: hw.module @direct_without_inverter(
// CHECK-NEXT: %[[NO_INV:.*]] = hw.instance {{.*}} @asymmetric_gate(a: %b: i1, b: %a: i1) -> (y: i1) {test.arrival_times = [9]}
// CHECK-NEXT: hw.output %[[NO_INV]] : i1
hw.module @direct_without_inverter(in %a: i1, in %b: i1, out y: i1) {
  %0 = synth.aig.and_inv %a, not %b : i1
  hw.output %0 : i1
}

// -----

// Timing chooses the cheapest inverter that preserves the candidate's best
// arrival time on each pin. Only pin a is critical: fast+small reaches time 9
// at total area 6, whereas fast+fast has the same arrival and costs area 9.
// Area instead chooses small+small, total area 3 and arrival time 14.
hw.module @and_gate(in %a: i1, in %b: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<8, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>]>} {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}
hw.module @inv_small(in %a: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<6, 0, #synth.polarity<negative>>]>} {
  %0 = synth.aig.and_inv not %a : i1
  hw.output %0 : i1
}
hw.module @inv_fast(in %a: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 4.0 : f64, arcs = [#synth.linear_timing_arc<1, 0, #synth.polarity<negative>>]>} {
  %0 = synth.aig.and_inv not %a : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @inverter_slack(
// AREA-DAG: %[[SLACK_SMALL_A:.*]] = hw.instance {{.*}} @inv_small(a: %a: i1) -> (y: i1)
// AREA-DAG: %[[SLACK_SMALL_B:.*]] = hw.instance {{.*}} @inv_small(a: %b: i1) -> (y: i1)
// AREA-NEXT: %[[SLACK_AREA:.*]] = hw.instance {{.*}} @and_gate(a: %[[SLACK_SMALL_A]]: i1, b: %[[SLACK_SMALL_B]]: i1) -> (y: i1) {test.arrival_times = [14]}
// AREA-NEXT: hw.output %[[SLACK_AREA]] : i1
// TIMING-DAG: %[[SLACK_FAST_A:.*]] = hw.instance {{.*}} @inv_fast(a: %a: i1) -> (y: i1)
// TIMING-DAG: %[[SLACK_SLOW_B:.*]] = hw.instance {{.*}} @inv_small(a: %b: i1) -> (y: i1)
// TIMING-NEXT: %[[SLACK_TIMING:.*]] = hw.instance {{.*}} @and_gate(a: %[[SLACK_FAST_A]]: i1, b: %[[SLACK_SLOW_B]]: i1) -> (y: i1) {test.arrival_times = [9]}
// TIMING-NEXT: hw.output %[[SLACK_TIMING]] : i1
hw.module @inverter_slack(in %a: i1, in %b: i1, out y: i1) {
  %0 = synth.aig.and_inv not %a, not %b : i1
  hw.output %0 : i1
}

// -----

// Simultaneously compose a nontrivial permutation, two input phases, and the
// output phase. With a as the least significant truth-table input,
// G(a,b,c,d) = (a&b) | (b&d) | (a&c&!d) | (!a&c&d) has table 0xDCA8.
// The target is defined independently as F(a,b,c,d) = !G(!c,a,!d,b), table
// 0x7534. Its pin delays are [2,3,5,7]; each INV adds 11, so the final arrival
// is max(11+2, 3, 11+5, 7) + 11 = 27.
hw.module @four_input_gate(in %a: i1, in %b: i1, in %c: i1, in %d: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<2, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<3, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<5, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<7, 0, #synth.polarity<positive>>]>} {
  %t0 = synth.aig.and_inv %a, %b : i1
  %t1 = synth.aig.and_inv %b, %d : i1
  %ac = synth.aig.and_inv %a, %c : i1
  %t2 = synth.aig.and_inv %ac, not %d : i1
  %nc = synth.aig.and_inv not %a, %c : i1
  %t3 = synth.aig.and_inv %nc, %d : i1
  %nor01 = synth.aig.and_inv not %t0, not %t1 : i1
  %nor23 = synth.aig.and_inv not %t2, not %t3 : i1
  %nor = synth.aig.and_inv %nor01, %nor23 : i1
  %0 = synth.aig.and_inv not %nor : i1
  hw.output %0 : i1
}
hw.module @inv_gate(in %a: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.5 : f64, arcs = [#synth.linear_timing_arc<11, 0, #synth.polarity<negative>>]>} {
  %0 = synth.aig.and_inv not %a : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @composed_phase_and_permutation(
// CHECK-DAG: %[[PHASE_C:.*]] = hw.instance {{.*}} @inv_gate(a: %c: i1) -> (y: i1)
// CHECK-DAG: %[[PHASE_D:.*]] = hw.instance {{.*}} @inv_gate(a: %d: i1) -> (y: i1)
// CHECK-NEXT: %[[FOUR:.*]] = hw.instance {{.*}} @four_input_gate(a: %[[PHASE_C]]: i1, b: %a: i1, c: %[[PHASE_D]]: i1, d: %b: i1) -> (y: i1)
// CHECK-NEXT: %[[FOUR_INV:.*]] = hw.instance {{.*}} @inv_gate(a: %[[FOUR]]: i1) -> (y: i1) {test.arrival_times = [27]}
// CHECK-NEXT: hw.output %[[FOUR_INV]] : i1
hw.module @composed_phase_and_permutation(in %a: i1, in %b: i1, in %c: i1, in %d: i1, out y: i1) {
  %t0 = synth.aig.and_inv not %c, %a : i1
  %t1 = synth.aig.and_inv %a, %b : i1
  %ncd = synth.aig.and_inv not %c, not %d : i1
  %t2 = synth.aig.and_inv %ncd, not %b : i1
  %cd = synth.aig.and_inv %c, not %d : i1
  %t3 = synth.aig.and_inv %cd, %b : i1
  %nor01 = synth.aig.and_inv not %t0, not %t1 : i1
  %nor23 = synth.aig.and_inv not %t2, not %t3 : i1
  %0 = synth.aig.and_inv %nor01, %nor23 : i1
  hw.output %0 : i1
}

// -----

// A larger direct cut can still absorb the inverted edge and beat a smaller
// cell with a local phase adapter under both area and timing strategies.
hw.module @and_gate(in %a: i1, in %b: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<4, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<4, 0, #synth.polarity<positive>>]>} {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}
hw.module @inv_gate(in %a: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 2.0 : f64, arcs = [#synth.linear_timing_arc<3, 0, #synth.polarity<negative>>]>} {
  %0 = synth.aig.and_inv not %a : i1
  hw.output %0 : i1
}
hw.module @fused_gate(in %a: i1, in %b: i1, in %c: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.25 : f64, arcs = [#synth.linear_timing_arc<1, 0, #synth.polarity<negative>>, #synth.linear_timing_arc<1, 0, #synth.polarity<negative>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>]>} {
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv not %0, %c : i1
  hw.output %1 : i1
}

// CHECK-LABEL: hw.module @absorbed_phase(
// CHECK-NEXT: %[[FUSED:.*]] = hw.instance {{.*}} @fused_gate(a: %a: i1, b: %b: i1, c: %c: i1) -> (y: i1) {test.arrival_times = [1]}
// CHECK-NEXT: hw.output %[[FUSED]] : i1
hw.module @absorbed_phase(in %a: i1, in %b: i1, in %c: i1, out y: i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv not %0, %c : i1
  hw.output %1 : i1
}

//--- no-inverter.mlir

// A buffer is not an inverter. Reject input-phase candidates when the library
// has no actual unary NOT implementation.
hw.module @and_gate(in %a: i1, in %b: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>]>} {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}
hw.module @buffer_gate(in %a: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<1, 0, #synth.polarity<positive>>]>} {
  hw.output %a : i1
}
hw.module @missing_input_inverter(in %a: i1, in %b: i1, out y: i1) {
  // expected-error-re@+1 {{No matching cut found for value: {{.*}}}}
  %0 = synth.aig.and_inv not %a, %b : i1
  hw.output %0 : i1
}

// -----

// The same restriction applies to a candidate requiring output inversion.
hw.module @and_gate(in %a: i1, in %b: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>]>} {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}
hw.module @missing_output_inverter(in %a: i1, in %b: i1, out y: i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  // expected-error-re@+1 {{No matching cut found for value: {{.*}}}}
  %1 = synth.aig.and_inv not %0 : i1
  hw.output %1 : i1
}
