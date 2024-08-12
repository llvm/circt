// RUN: circt-opt %s | circt-opt | FileCheck %s

%true = hw.constant true
%s = unrealized_conversion_cast to !ltl.sequence
%p = unrealized_conversion_cast to !ltl.property

//===----------------------------------------------------------------------===//
// Assertions
//===----------------------------------------------------------------------===//

// CHECK: verif.assert {{%.+}} : i1
// CHECK: verif.assert {{%.+}} label "foo1" : i1
// CHECK: verif.assert {{%.+}} : !ltl.sequence
// CHECK: verif.assert {{%.+}} : !ltl.property
verif.assert %true : i1
verif.assert %true label "foo1" : i1
verif.assert %s : !ltl.sequence
verif.assert %p : !ltl.property

// CHECK: verif.assume {{%.+}} : i1
// CHECK: verif.assume {{%.+}} label "foo2" : i1
// CHECK: verif.assume {{%.+}} : !ltl.sequence
// CHECK: verif.assume {{%.+}} : !ltl.property
verif.assume %true : i1
verif.assume %true label "foo2" : i1
verif.assume %s : !ltl.sequence
verif.assume %p : !ltl.property

// CHECK: verif.cover {{%.+}} : i1
// CHECK: verif.cover {{%.+}} label "foo3" : i1
// CHECK: verif.cover {{%.+}} : !ltl.sequence
// CHECK: verif.cover {{%.+}} : !ltl.property
verif.cover %true : i1
verif.cover %true label "foo3" : i1
verif.cover %s : !ltl.sequence
verif.cover %p : !ltl.property

//===----------------------------------------------------------------------===//
// Formal
//===----------------------------------------------------------------------===//
hw.module @Foo(in %0 "0": i1, in %1 "1": i1, out "" : i1, out "1" : i1) {
  hw.output %0 , %1: i1, i1
 }

// CHECK: verif.formal @formal1(k = 20 : i64) {
verif.formal @formal1(k = 20) {
  // CHECK: %[[C1:.+]] = hw.constant true
  %c1_i1 = hw.constant true
  // CHECK: %[[SYM:.+]] = verif.symbolic_input : i1
  %sym = verif.symbolic_input : i1
  // CHECK: %[[CLK_U:.+]] = comb.xor %8, %[[C1]] : i1
  %clk_update = comb.xor %8, %c1_i1 : i1
  // CHECK: %8 = verif.concrete_input %[[C1]], %[[CLK_U]] : i1
  %8 = verif.concrete_input %c1_i1, %clk_update : i1
  // CHECK: %foo.0, %foo.1 = hw.instance "foo" @Foo("0": %6: i1, "1": %8: i1) -> ("": i1, "1": i1)
  %foo.0, %foo.1 = hw.instance "foo" @Foo("0": %sym: i1, "1": %8 : i1) -> ("" : i1, "1" : i1)
}

// CHECK-LABEL: hw.module @Bar
hw.module @Bar(in %foo : i8, out "" : i8, out "1" : i8) { 
  // CHECK: %[[C1:.+]] = hw.constant
  %c1_8 = hw.constant 1 : i8
  // CHECK: %[[O1:.+]] = comb.add
  %to0 = comb.add bin %foo, %c1_8 : i8
  // CHECK: %[[O2:.+]] = comb.sub
  %to1 = comb.sub bin %foo, %c1_8 : i8

  // CHECK: %[[OUT:.+]]:2 = verif.contract(%[[O1]], %[[O2]]) : (i8, i8) -> (i8, i8) {
  %o0, %o1 = verif.contract (%to0, %to1) : (i8, i8) -> (i8, i8) {
    // CHECK: ^bb0(%[[BAR0:.+]]: i8, %[[BAR1:.+]]: i8):
    ^bb0(%bar.0 : i8, %bar.1 : i8): 
      // CHECK: %[[C0:.+]] = hw.constant 0 : i8
      %c0_8 = hw.constant 0 : i8 
      // CHECK: %[[PREC:.+]] = comb.icmp bin ugt %foo, %[[C0]] : i8
      %prec = comb.icmp bin ugt %foo, %c0_8 : i8
      // CHECK: verif.require %[[PREC]] : i1
      verif.require %prec : i1

      // CHECK: %[[P0:.+]] = comb.icmp bin ugt %[[BAR0]], %foo : i8
      %post = comb.icmp bin ugt %bar.0, %foo : i8
      // CHECK: %[[P1:.+]] = comb.icmp bin ult %[[BAR1]], %foo : i8
      %post1 = comb.icmp bin ult %bar.1, %foo : i8
      // CHECK: verif.ensure %[[P0]] : i1
      verif.ensure %post : i1
      // CHECK: verif.ensure %[[P1]] : i1
      verif.ensure %post1 : i1
      // CHECK: verif.yield %[[BAR0]], %[[BAR1]] : i8, i8
      verif.yield %bar.0, %bar.1 : i8, i8
  } 
  
  // CHECK-LABEL: hw.output
  hw.output %o0, %o1 : i8, i8
}

//===----------------------------------------------------------------------===//
// Print-related
// Must be inside hw.module to ensure that the dialect is loaded.
//===----------------------------------------------------------------------===//

hw.module @foo() {
// CHECK:    %false = hw.constant false
// CHECK:    %[[FSTR:.*]] = verif.format_verilog_string "Hi %x\0A"(%false) : i1
// CHECK:    verif.print %[[FSTR]]
  %false = hw.constant false
  %fstr = verif.format_verilog_string "Hi %x\0A" (%false) : i1
  verif.print %fstr
}

// CHECK-LABEL: hw.module @HasBeenReset
hw.module @HasBeenReset(in %clock: i1, in %reset: i1) {
  // CHECK-NEXT: verif.has_been_reset %clock, async %reset
  // CHECK-NEXT: verif.has_been_reset %clock, sync %reset
  %hbr0 = verif.has_been_reset %clock, async %reset
  %hbr1 = verif.has_been_reset %clock, sync %reset
}

//===----------------------------------------------------------------------===//
// Logical Equivalence Checking related operations
//===----------------------------------------------------------------------===//

// CHECK: verif.lec first {
// CHECK: } second {
// CHECK: }
verif.lec first {
} second {
}

// CHECK: verif.lec {verif.some_attr} first {
// CHECK: ^bb0(%{{.*}}: i32, %{{.*}}: i32):
// CHECK:   verif.yield %{{.*}}, %{{.*}} : i32, i32 {verif.some_attr}
// CHECK: } second {
// CHECK: ^bb0(%{{.*}}: i32, %{{.*}}: i32):
// CHECK:   verif.yield %{{.*}}, %{{.*}} : i32, i32 {verif.some_attr}
// CHECK: }
verif.lec {verif.some_attr} first {
^bb0(%arg0: i32, %arg1: i32):
  verif.yield %arg0, %arg1 : i32, i32 {verif.some_attr}
} second {
^bb0(%arg0: i32, %arg1: i32):
  verif.yield %arg0, %arg1 : i32, i32 {verif.some_attr}
}

//===----------------------------------------------------------------------===//
// Bounded Model Checking related operations
//===----------------------------------------------------------------------===//

// CHECK: verif.bmc bound 10 num_regs 0 attributes {verif.some_attr} init {
// CHECK: } loop {
// CHECK: } circuit {
// CHECK: ^bb0(%{{.*}}):
// CHECK: verif.yield %{{.*}} : i32
// CHECK: }
verif.bmc bound 10 num_regs 0 attributes {verif.some_attr} init {
} loop {
} circuit {
^bb0(%arg0: i32):
  %false = hw.constant false
  // Arbitrary assertion so op verifies
  verif.assert %false : i1
  verif.yield %arg0 : i32
}

//CHECK: verif.bmc bound 10 num_regs 1 attributes {verif.some_attr}
//CHECK: init {
//CHECK:   %{{.*}} = hw.constant false
//CHECK:   %{{.*}} = seq.to_clock %{{.*}}
//CHECK:   verif.yield %{{.*}}, %{{.*}} : !seq.clock, i1
//CHECK: }
//CHECK: loop {
//CHECK:   ^bb0(%{{.*}}: !seq.clock, %{{.*}}: i1):
//CHECK:   %{{.*}} = seq.from_clock %{{.*}}
//CHECK:   %{{.*}} = hw.constant true
//CHECK:   %{{.*}} = comb.xor %{{.*}}, %{{.*}} : i1
//CHECK:   %{{.*}} = comb.xor %{{.*}}, %{{.*}} : i1
//CHECK:   %{{.*}} = seq.to_clock %{{.*}}
//CHECK:   verif.yield %{{.*}}, %{{.*}} : !seq.clock, i1
//CHECK: }
//CHECK: circuit {
//CHECK: ^bb0(%{{.*}}: !seq.clock, %{{.*}}: i32, %{{.*}}: i32):
//CHECK:   %{{.*}} = hw.constant -1 : i32
//CHECK:   %{{.*}} = comb.add %{{.*}}, %{{.*}} : i32
//CHECK:   %{{.*}} = comb.xor %{{.*}}, %{{.*}} : i32
//CHECK:   verif.yield %{{.*}}, %{{.*}} : i32, i32
//CHECK: }
verif.bmc bound 10 num_regs 1 attributes {verif.some_attr}
init {
  %c0_i1 = hw.constant 0 : i1
  %clk = seq.to_clock %c0_i1
  verif.yield %clk, %c0_i1 : !seq.clock, i1
}
loop {
  ^bb0(%clk: !seq.clock, %stateArg: i1):
  %from_clock = seq.from_clock %clk
  %c-1_i1 = hw.constant -1 : i1
  %neg_clock = comb.xor %from_clock, %c-1_i1 : i1
  %newStateArg = comb.xor %stateArg, %c-1_i1 : i1
  %newclk = seq.to_clock %neg_clock
  verif.yield %newclk, %newStateArg : !seq.clock, i1
}
circuit {
^bb0(%clk: !seq.clock, %arg0: i32, %state0: i32):
  %c-1_i32 = hw.constant -1 : i32
  %0 = comb.add %arg0, %state0 : i32
  // %state0 is the result of a seq.compreg taking %0 as input
  %2 = comb.xor %state0, %c-1_i32 : i32
  %false = hw.constant false
  // Arbitrary assertion so op verifies
  verif.assert %false : i1
  verif.yield %2, %0 : i32, i32
}
