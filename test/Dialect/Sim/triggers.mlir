// RUN: circt-opt %s --canonicalize | FileCheck %s
// RUN: circt-opt %s --canonicalize --cse | FileCheck %s

// CHECK-LABEL: hw.module @root_triggers
// CHECK-DAG:   [[PE:%.*]] = sim.on_edge posedge %clock
// CHECK-DAG:   [[NE:%.*]] = sim.on_edge negedge %clock
// CHECK-DAG:   [[BE:%.*]] = sim.on_edge edge %clock
// CHECK-DAG:   [[IT:%.*]] = sim.on_init
// CHECK:       hw.output [[PE]], [[NE]], [[BE]], [[IT]] : !sim.trigger.edge<posedge>, !sim.trigger.edge<negedge>, !sim.trigger.edge<edge>, !sim.trigger.init
hw.module @root_triggers(in %clock : !seq.clock, out o0 : !sim.trigger.edge<posedge>, out o1 : !sim.trigger.edge<negedge>, out o2 : !sim.trigger.edge<edge>, out o3 : !sim.trigger.init) {
  %0 = sim.on_edge posedge %clock
  %1 = sim.on_edge negedge %clock
  %2 = sim.on_edge edge %clock
  %3 = sim.on_init
  hw.output %0, %1, %2, %3 : !sim.trigger.edge<posedge>, !sim.trigger.edge<negedge>, !sim.trigger.edge<edge>, !sim.trigger.init
}

// CHECK-LABEL: hw.module @constant_clock
hw.module @constant_clock(out o0 : !sim.trigger.edge<edge>) {
  // CHECK-NOT: sim.on_edge
  // CHECK: %[[NEVER:.*]] = sim.never : !sim.trigger.edge<edge>
  // CHECK: %[[NEVER]]
  %clock = seq.const_clock low
  %0 = sim.on_edge edge %clock
  hw.output %0 : !sim.trigger.edge<edge>
}

// CHECK-LABEL: hw.module @fold_gates
hw.module @fold_gates(in %trig : !sim.trigger.edge<posedge>, in %en: i1,  out o0 : !sim.trigger.edge<posedge>, out o1 : !sim.trigger.edge<posedge>, out o2 : !sim.trigger.edge<posedge>) {
  %true = hw.constant true
  %false = hw.constant false
  // CHECK: %[[NEVER:.*]] = sim.never : !sim.trigger.edge<posedge>
  %never = sim.never : !sim.trigger.edge<posedge>
  // CHECK-NOT: sim.trigger_gate
  %on = sim.trigger_gate %trig if %true : !sim.trigger.edge<posedge>
  %off = sim.trigger_gate %trig if %false : !sim.trigger.edge<posedge>
  %dyn = sim.trigger_gate %never if %en : !sim.trigger.edge<posedge>
  // CHECK: hw.output %trig, %[[NEVER]], %[[NEVER]]
  hw.output %on, %off, %dyn : !sim.trigger.edge<posedge>, !sim.trigger.edge<posedge>, !sim.trigger.edge<posedge>
}

// CHECK-LABEL: hw.module @fold_triggered
hw.module @fold_triggered(in %a : i8, in %en : i1, in %trig : !sim.trigger.edge<posedge>, out o0 : i8, out o1 : i9, out o2 : i8) {
  // CHECK: %[[CST12:.*]] = hw.constant 12 : i8
  %false = hw.constant false

  // CHECK: %[[GATE:.*]] = sim.trigger_gate %trig if %en
  %enGate = sim.trigger_gate %trig if %en : !sim.trigger.edge<posedge>
  %deadGate = sim.trigger_gate %trig if %false : !sim.trigger.edge<posedge>

  // CHECK: sim.triggered () on (%[[GATE]] : !sim.trigger.edge<posedge>) {
  // CHECK-NEXT:  "Don't touch live process"
  sim.triggered () on (%enGate : !sim.trigger.edge<posedge>) {
    %0 = sim.fmt.lit "Don't touch live process"
    sim.proc.print %0
  } : () -> ()

  // CHECK: %[[RES:.*]]:2 = sim.triggered (%a) on (%[[GATE]] : !sim.trigger.edge<posedge>) tieoff [12 : i8, 33 : i9]  {
  // CHECK: "Don't touch live process with results"
  // CHECK: arith.extui %{{.+}} : i8 to i9
  // CHECK: (i8) -> (i8, i9)

  %res:2 = sim.triggered (%a) on (%enGate : !sim.trigger.edge<posedge>) tieoff [12 : i8, 33 : i9] {
  ^bb0(%arg: i8):
    %0 = sim.fmt.lit "Don't touch live process with results"
    sim.proc.print %0
    %ext = arith.extui %arg : i8 to i9
    sim.yield_seq %arg, %ext : i8, i9
  } : (i8) -> (i8, i9)

  // CHECK-NOT: sim.triggered

  %fold = sim.triggered () on (%deadGate : !sim.trigger.edge<posedge>) tieoff [12 : i8] {
    %cst0_i8 = hw.constant 0 : i8
    %0 = sim.fmt.lit "Fold dead process with result"
    sim.proc.print %0
    sim.yield_seq %cst0_i8 : i8
  } : () -> (i8)

  sim.triggered () on (%deadGate : !sim.trigger.edge<posedge>) {
    %0 = sim.fmt.lit "Remove dead process"
    sim.proc.print %0
  } : () -> ()

  sim.triggered () on (%enGate : !sim.trigger.edge<posedge>) {
  } : () -> ()

   // CHECK: hw.output %[[RES]]#0, %[[RES]]#1, %[[CST12]] : i8, i9, i8
  hw.output %res#0, %res#1, %fold : i8, i9, i8
}

// CHECK-LABEL: hw.module @empty_sequence
// CHECK-NOT:   sim.trigger_sequence
hw.module @empty_sequence(in %trig : !sim.trigger.init) {
  sim.trigger_sequence %trig, 0 : !sim.trigger.init
}

// CHECK-LABEL: hw.module @trivial_sequence
// CHECK-NOT:   sim.trigger_sequence
// CHECK:       hw.output %trig
hw.module @trivial_sequence(in %trig : !sim.trigger.init, out o: !sim.trigger.init) {
  %out = sim.trigger_sequence %trig, 1 : !sim.trigger.init
  hw.output %out : !sim.trigger.init
}

// CHECK-LABEL: hw.module @dead_sequence
// CHECK-NOT:   sim.trigger_sequence
hw.module @dead_sequence(in %trig : !sim.trigger.init) {
  %dead:128 = sim.trigger_sequence %trig, 128 : !sim.trigger.init
}

// CHECK-LABEL: hw.module @mostly_dead_sequence
// CHECK:       %[[RES:.*]]:2 = sim.trigger_sequence %trig, 2 : !sim.trigger.init
// CHECK-NEXT:  hw.output %[[RES]]#1, %[[RES]]#0, %[[RES]]#1
hw.module @mostly_dead_sequence(in %trig : !sim.trigger.init, out o0: !sim.trigger.init, out o1: !sim.trigger.init, out o2: !sim.trigger.init) {
  %notdead:128 = sim.trigger_sequence %trig, 128 : !sim.trigger.init
  hw.output %notdead#50, %notdead#2, %notdead#50 : !sim.trigger.init, !sim.trigger.init, !sim.trigger.init
}

// CHECK-LABEL: hw.module @nested_sequence_0
// CHECK:       [[R:%.*]]:8 = sim.trigger_sequence %trig, 8 : !sim.trigger.edge<negedge>
// CHECK-NEXT:  hw.output [[R]]#0, [[R]]#1, [[R]]#2, [[R]]#3, [[R]]#4, [[R]]#5, [[R]]#6, [[R]]#7
hw.module @nested_sequence_0(in %trig : !sim.trigger.edge<negedge>, out o0: !sim.trigger.edge<negedge>, out o1: !sim.trigger.edge<negedge>, out o2: !sim.trigger.edge<negedge>, out o3: !sim.trigger.edge<negedge>, out o4: !sim.trigger.edge<negedge>, out o5: !sim.trigger.edge<negedge>,  out o6: !sim.trigger.edge<negedge>, out o7: !sim.trigger.edge<negedge>) {
  %a:4 = sim.trigger_sequence %trig, 4 : !sim.trigger.edge<negedge>
  %b:2 = sim.trigger_sequence %a#0, 2 : !sim.trigger.edge<negedge>
  %c:3 = sim.trigger_sequence %a#1, 3 : !sim.trigger.edge<negedge>
  %d:2 = sim.trigger_sequence %a#2, 2 : !sim.trigger.edge<negedge>

  hw.output %b#0, %b#1, %c#0, %c#1, %c#2, %d#0, %d#1, %a#3  : !sim.trigger.edge<negedge>, !sim.trigger.edge<negedge>, !sim.trigger.edge<negedge>, !sim.trigger.edge<negedge>, !sim.trigger.edge<negedge>, !sim.trigger.edge<negedge>, !sim.trigger.edge<negedge>, !sim.trigger.edge<negedge>
}

// CHECK-LABEL: hw.module @nested_sequence_1
// CHECK:       [[R:%.*]]:8 = sim.trigger_sequence %trig, 8 : !sim.trigger.edge<negedge>
// CHECK-NEXT:  hw.output [[R]]#0, [[R]]#1, [[R]]#2, [[R]]#3, [[R]]#4, [[R]]#5, [[R]]#6, [[R]]#7
hw.module @nested_sequence_1(in %trig : !sim.trigger.edge<negedge>, out o0: !sim.trigger.edge<negedge>, out o1: !sim.trigger.edge<negedge>, out o2: !sim.trigger.edge<negedge>, out o3: !sim.trigger.edge<negedge>, out o4: !sim.trigger.edge<negedge>, out o5: !sim.trigger.edge<negedge>,  out o6: !sim.trigger.edge<negedge>, out o7: !sim.trigger.edge<negedge>) {
  %a:2 = sim.trigger_sequence %trig, 2 : !sim.trigger.edge<negedge>

  %b:2 = sim.trigger_sequence %a#0, 2 : !sim.trigger.edge<negedge>
  %c:2 = sim.trigger_sequence %a#1, 2 : !sim.trigger.edge<negedge>

  %d:2 = sim.trigger_sequence %b#0, 2 : !sim.trigger.edge<negedge>
  %e:2 = sim.trigger_sequence %b#1, 2 : !sim.trigger.edge<negedge>
  %f:2 = sim.trigger_sequence %c#0, 2 : !sim.trigger.edge<negedge>
  %g:2 = sim.trigger_sequence %c#1, 2 : !sim.trigger.edge<negedge>

  hw.output %d#0, %d#1, %e#0, %e#1, %f#0, %f#1, %g#0, %g#1  : !sim.trigger.edge<negedge>, !sim.trigger.edge<negedge>, !sim.trigger.edge<negedge>, !sim.trigger.edge<negedge>, !sim.trigger.edge<negedge>, !sim.trigger.edge<negedge>, !sim.trigger.edge<negedge>, !sim.trigger.edge<negedge>
}

// CHECK-LABEL: hw.module @nested_sequence_2
// CHECK:       [[R0:%.*]]:4 = sim.trigger_sequence %trig, 4
// CHECK:       [[R1:%.*]]:2 = sim.trigger_sequence [[R0]]#3, 2
// CHECK:       [[R2:%.*]]:2 = sim.trigger_sequence [[R0]]#3, 2
// CHECK:       hw.output [[R0]]#0, [[R0]]#1, [[R0]]#2, [[R0]]#0, [[R1]]#0, [[R1]]#1, [[R2]]#0, [[R2]]#1
hw.module @nested_sequence_2(in %trig : !sim.trigger.edge<negedge>, out o0: !sim.trigger.edge<negedge>, out o1: !sim.trigger.edge<negedge>, out o2: !sim.trigger.edge<negedge>, out o3: !sim.trigger.edge<negedge>, out o4: !sim.trigger.edge<negedge>, out o5: !sim.trigger.edge<negedge>,  out o6: !sim.trigger.edge<negedge>, out o7: !sim.trigger.edge<negedge>) {
  %a:4 = sim.trigger_sequence %trig, 4 : !sim.trigger.edge<negedge>
  %b:3 = sim.trigger_sequence %a#1, 3 : !sim.trigger.edge<negedge>
  %c:2 = sim.trigger_sequence %b#0, 2 : !sim.trigger.edge<negedge>
  %d:3 = sim.trigger_sequence %a#3, 3 : !sim.trigger.edge<negedge>
  %e:3 = sim.trigger_sequence %a#3, 3 : !sim.trigger.edge<negedge>
  hw.output %b#0, %b#1, %b#2, %c#1, %d#0, %d#2, %e#1, %e#2  : !sim.trigger.edge<negedge>, !sim.trigger.edge<negedge>, !sim.trigger.edge<negedge>, !sim.trigger.edge<negedge>, !sim.trigger.edge<negedge>, !sim.trigger.edge<negedge>, !sim.trigger.edge<negedge>, !sim.trigger.edge<negedge>
}

// CHECK-LABEL: hw.module @skewed_binary_tree
// CHECK:       [[R0:%.*]]:7 = sim.trigger_sequence %trig, 7
// CHECK-NEXT:  hw.output [[R0]]#0, [[R0]]#1, [[R0]]#2, [[R0]]#3, [[R0]]#4, [[R0]]#5, [[R0]]#6
hw.module @skewed_binary_tree(in %trig : !sim.trigger.edge<posedge>, out o0: !sim.trigger.edge<posedge>, out o1: !sim.trigger.edge<posedge>, out o2: !sim.trigger.edge<posedge>, out o3: !sim.trigger.edge<posedge>, out o4: !sim.trigger.edge<posedge>, out o5: !sim.trigger.edge<posedge>,  out o6: !sim.trigger.edge<posedge>) {
  %h:2 = sim.trigger_sequence %g#1, 2 : !sim.trigger.edge<posedge>
  %g:2 = sim.trigger_sequence %f#1, 2 : !sim.trigger.edge<posedge>
  %f:2 = sim.trigger_sequence %e#1, 2 : !sim.trigger.edge<posedge>
  %e:2 = sim.trigger_sequence %d#1, 2 : !sim.trigger.edge<posedge>
  %d:2 = sim.trigger_sequence %c#1, 2 : !sim.trigger.edge<posedge>
  %c:2 = sim.trigger_sequence %b#1, 2 : !sim.trigger.edge<posedge>
  %b:2 = sim.trigger_sequence %a#1, 2 : !sim.trigger.edge<posedge>
  %a:2 = sim.trigger_sequence %trig, 2 : !sim.trigger.edge<posedge>
  hw.output %a#0, %b#0, %c#0, %d#0, %f#0, %g#0, %h#0  : !sim.trigger.edge<posedge>, !sim.trigger.edge<posedge>, !sim.trigger.edge<posedge>, !sim.trigger.edge<posedge>, !sim.trigger.edge<posedge>, !sim.trigger.edge<posedge>, !sim.trigger.edge<posedge>
}

// CHECK-LABEL: hw.module @dead_skewed_binary_tree
// CHECK-NOT:   sim.trigger_sequence %trig,
hw.module @dead_skewed_binary_tree(in %trig : !sim.trigger.edge<posedge>) {
  %a:2 = sim.trigger_sequence %trig, 2 : !sim.trigger.edge<posedge>
  %b:2 = sim.trigger_sequence %a#1, 2 : !sim.trigger.edge<posedge>
  %c:2 = sim.trigger_sequence %b#1, 2 : !sim.trigger.edge<posedge>
  %d:2 = sim.trigger_sequence %c#1, 2 : !sim.trigger.edge<posedge>
}

// CHECK-LABEL: hw.module @always_enabled_gate(in %trig : !sim.trigger.init, out o : !sim.trigger.init)
// CHECK:         hw.output %trig
hw.module @always_enabled_gate(in %trig : !sim.trigger.init, out o : !sim.trigger.init) {
  %true = hw.constant true
  %gate = sim.trigger_gate %trig if %true : !sim.trigger.init
  hw.output %gate : !sim.trigger.init
}

// CHECK-LABEL: hw.module @always_disabled_gate(in %trig : !sim.trigger.init, out o : !sim.trigger.init)
// CHECK:         [[NEVER:%.*]] = sim.never : !sim.trigger.init
// CHECK:         hw.output [[NEVER]] : !sim.trigger.init
hw.module @always_disabled_gate(in %trig : !sim.trigger.init, out o : !sim.trigger.init) {
  %false = hw.constant false
  %gate = sim.trigger_gate %trig if %false : !sim.trigger.init
  hw.output %gate : !sim.trigger.init
}
