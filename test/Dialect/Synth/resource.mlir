// RUN: circt-opt --synth-print-resource-usage-analysis='output-file="-"' %s | FileCheck %s
// RUN: circt-opt --synth-print-resource-usage-analysis='top-module-name=top output-file="-" emit-json=true' %s | FileCheck %s --check-prefix=JSON
// RUN: circt-opt --synth-print-resource-usage-analysis='top-module-name=nested output-file="-" emit-json=true' %s | FileCheck %s --check-prefix=JSON-NESTED

// Test basic single-bit operations
// CHECK:      Resource Usage Analysis for module: top
// CHECK-NEXT: ========================================
// CHECK-NEXT: Total:
// CHECK-NEXT:   comb.and: 2
// CHECK-NEXT:   comb.or: 3
// CHECK-NEXT:   comb.xor: 2
// CHECK-NEXT:   synth.aig.and_inv: 2

// Test JSON output - basic hierarchy
// JSON: [{"instances":[
// JSON-SAME: {"instanceName":"inst1","moduleName":"basic"
// JSON-SAME: {"instanceName":"inst2","moduleName":"basic"
// JSON-SAME: "moduleName":"top","total":{"comb.and":2,"comb.or":3,"comb.xor":2,"synth.aig.and_inv":2}}]

// Test JSON output - nested hierarchy with local and total counts
// JSON-NESTED: [{"instances":[
// JSON-NESTED-SAME: {"instanceName":"mid1","moduleName":"middle","usage":{"instances":[
// JSON-NESTED-SAME: {"instanceName":"leaf1","moduleName":"leaf"
// JSON-NESTED-SAME: "local":{"comb.and":1,"comb.xor":1}
// JSON-NESTED-SAME: "total":{"comb.and":1,"comb.xor":1}
// JSON-NESTED-SAME: {"instanceName":"leaf2","moduleName":"leaf"
// JSON-NESTED-SAME: "local":{"comb.or":1}
// JSON-NESTED-SAME: "moduleName":"middle","total":{"comb.and":2,"comb.or":1,"comb.xor":2}
// JSON-NESTED-SAME: {"instanceName":"mid2","moduleName":"middle","usage":{"instances":[
// JSON-NESTED-SAME: "local":{"comb.or":1}
// JSON-NESTED-SAME: "moduleName":"nested","total":{"comb.and":4,"comb.or":3,"comb.xor":4}}]

hw.module private @basic(in %a : i1, in %b : i1, out x : i1) {
  %p = synth.aig.and_inv not %a, %b : i1
  %q = comb.and %p, %a : i1
  %r = comb.or %q, %a : i1
  %s = comb.xor %r, %a : i1
  hw.output %s : i1
}

hw.module private @top(in %a : i1, in %b : i1, out x : i1) {
  %0 = hw.instance "inst1" @basic(a: %a: i1, b: %b: i1) -> (x: i1)
  %1 = hw.instance "inst2" @basic(a: %a: i1, b: %b: i1) -> (x: i1)
  %s = comb.or %0, %1 : i1
  hw.output %s : i1
}

hw.module private @unrelated(in %a : i1, in %b : i1, out x : i1) {
  %p = synth.aig.and_inv not %a, %b : i1
  hw.output %p : i1
}

// Test multi-bit operations (gate count should be multiplied by bitwidth)
// CHECK:      Resource Usage Analysis for module: multibit
// CHECK-NEXT: ========================================
// CHECK-NEXT: Total:
// CHECK-NEXT:   comb.and: 24
// CHECK-NEXT:   comb.or: 16
// CHECK-NEXT:   comb.xor: 8
// CHECK-NEXT:   synth.aig.and_inv: 8

hw.module private @multibit(in %a : i8, in %b : i8, in %c : i8, out x : i8) {
  // 3-input AND on 8-bit: (3-1) * 8 = 16 gates
  %and3 = comb.and %a, %b, %c : i8
  // 2-input AND on 8-bit: (2-1) * 8 = 8 gates
  %and2 = comb.and %a, %b : i8
  // 3-input OR on 8-bit: (3-1) * 8 = 16 gates
  %or3 = comb.or %a, %b, %c : i8
  // 2-input XOR on 8-bit: (2-1) * 8 = 8 gates
  %xor2 = comb.xor %a, %b : i8
  // AIG on 8-bit: (2-1) * 8 = 8 gates
  %aig = synth.aig.and_inv not %a, %b : i8
  hw.output %aig : i8
}

// Test sequential elements (registers)
// CHECK:      Resource Usage Analysis for module: sequential
// CHECK-NEXT: ========================================
// CHECK-NEXT: Total:
// CHECK-NEXT:   <unknown>: 1
// CHECK-NEXT:   comb.xor: 8
// CHECK-NEXT:   seq.compreg: 16
// CHECK-NEXT:   seq.firreg: 8

hw.module private @sequential(in %clk : !seq.clock, in %a : i8, in %b : i8, out x : i8, out y : i8) {
  // CompReg on 8-bit: 8 DFF bits
  %r1 = seq.compreg %a, %clk : i8
  // CompReg on 8-bit: 8 DFF bits
  %r2 = seq.compreg %b, %clk : i8
  // FirReg on 8-bit: 8 DFF bits
  %r3 = seq.firreg %a clock %clk : i8
  // XOR on 8-bit: (2-1) * 8 = 8 gates
  %xor = comb.xor %r1, %r2 : i8
  // Constant: no resource count
  %cst = hw.constant 0 : i8
  // Unknown operation: count as 1
  %unk = comb.icmp eq %cst, %cst : i8
  hw.output %xor, %r3 : i8, i8
}

// Test nested module hierarchy
// CHECK:      Resource Usage Analysis for module: nested
// CHECK-NEXT: ========================================
// CHECK-NEXT: Total:
// CHECK-NEXT:   comb.and: 4
// CHECK-NEXT:   comb.or: 3
// CHECK-NEXT:   comb.xor: 4

hw.module private @leaf(in %a : i1, in %b : i1, out x : i1) {
  %and = comb.and %a, %b : i1
  %xor = comb.xor %a, %b : i1
  hw.output %xor : i1
}

hw.module private @middle(in %a : i1, in %b : i1, in %c : i1, out x : i1) {
  %0 = hw.instance "leaf1" @leaf(a: %a: i1, b: %b: i1) -> (x: i1)
  %1 = hw.instance "leaf2" @leaf(a: %b: i1, b: %c: i1) -> (x: i1)
  %or = comb.or %0, %1 : i1
  hw.output %or : i1
}

hw.module private @nested(in %a : i1, in %b : i1, in %c : i1, out x : i1) {
  %0 = hw.instance "mid1" @middle(a: %a: i1, b: %b: i1, c: %c: i1) -> (x: i1)
  %1 = hw.instance "mid2" @middle(a: %b: i1, b: %c: i1, c: %a: i1) -> (x: i1)
  %or = comb.or %0, %1 : i1
  hw.output %or : i1
}

// Test MIG (Majority-Inverter Graph) operations
// CHECK:      Resource Usage Analysis for module: mig_test
// CHECK-NEXT: ========================================
// CHECK-NEXT: Total:
// CHECK-NEXT:   synth.mig.maj_inv_3: 4
// CHECK-NEXT:   synth.mig.maj_inv_5: 4

hw.module private @mig_test(in %a : i4, in %b : i4, in %c : i4, out x : i4) {
  %maj1 = synth.mig.maj_inv %a, %b, %c : i4
  %maj2 = synth.mig.maj_inv %a, %b, %c, %a, %b : i4
  hw.output %maj2 : i4
}

// Test truth table operations
// CHECK:      Resource Usage Analysis for module: lut_test
// CHECK-NEXT: ========================================
// CHECK-NEXT: Total:
// CHECK-NEXT:   comb.truth_table_2: 1
// CHECK-NEXT:   comb.truth_table_3: 1

hw.module private @lut_test(in %a : i1, in %b : i1, in %c : i1, out x : i1, out y : i1) {
  // 2-input truth table (LUT2): 1 LUT
  %lut1 = comb.truth_table %a, %b -> [0, 1, 1, 0]
  // 3-input truth table (LUT3): 1 LUT
  %lut2 = comb.truth_table %a, %b, %c -> [0, 1, 1, 0, 1, 0, 0, 1]
  hw.output %lut1, %lut2 : i1, i1
}
