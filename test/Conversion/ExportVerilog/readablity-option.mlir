// RUN: circt-opt --lowering-options=enforceParenthesesToReductionOperators --export-verilog %s | FileCheck %s

hw.module @enforceParenthesesToReductionOperators(%a: i4) -> (o1:i1) {
  // CHECK: assign o1 = (&a) | (|a) | (^a);
  %one4 = hw.constant -1 : i4
  %0 = comb.icmp eq %a, %one4 : i4
  %zero4 = hw.constant 0 : i4
  %1 = comb.icmp ne %a, %zero4 : i4
  %2 = comb.parity %a : i4
  %3 = comb.or %0, %1, %2 : i1
  hw.output %3 : i1
}

