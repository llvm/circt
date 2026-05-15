// RUN: circt-opt --firrtl-probes-to-signals --split-input-file %s | FileCheck %s

// Test basic force operation on forceable wire
// CHECK-LABEL: firrtl.circuit "SimpleForce"
firrtl.circuit "SimpleForce" {
  firrtl.module @SimpleForce(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>, in %value: !firrtl.uint<4>) {
    // CHECK: %w = firrtl.wire : !firrtl.uint<4>
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>

    // Force the wire
    firrtl.ref.force %clock, %cond, %w_ref, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>

    // CHECK: %[[FORCED:.+]] = firrtl.reg %clock {{.+}} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %[[FORCED_VALUE:.+]] = firrtl.reg %clock {{.+}} : !firrtl.clock, !firrtl.uint<4>

    // CHECK: firrtl.when {{%.+}} : !firrtl.uint<1> {
    // CHECK:   firrtl.matchingconnect %[[FORCED]]
    // CHECK:   firrtl.matchingconnect %[[FORCED_VALUE]], {{%.+}}

    // CHECK: firrtl.when %[[FORCED]] : !firrtl.uint<1> {
    // CHECK:   firrtl.connect %w, %[[FORCED_VALUE]]
  }
}

// -----

// Test basic release operation
// CHECK-LABEL: firrtl.circuit "SimpleRelease"
firrtl.circuit "SimpleRelease" {
  firrtl.module @SimpleRelease(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    
    firrtl.ref.release %clock, %cond, %w_ref : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>
    
    // CHECK: %[[FORCED:.+]] = firrtl.reg %clock
    // CHECK: firrtl.when {{%.+}} : !firrtl.uint<1> {
    // CHECK:   firrtl.matchingconnect %[[FORCED]]
  }
}

// -----

// Test force and release together
// CHECK-LABEL: firrtl.circuit "ForceAndRelease"
firrtl.circuit "ForceAndRelease" {
  firrtl.module @ForceAndRelease(in %clock: !firrtl.clock, in %forceCond: !firrtl.uint<1>, in %releaseCond: !firrtl.uint<1>, in %value: !firrtl.uint<4>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    
    firrtl.ref.force %clock, %forceCond, %w_ref, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
    firrtl.ref.release %clock, %releaseCond, %w_ref : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>
    
    // Should generate state registers
    // CHECK: %[[FORCED:.+]] = firrtl.reg %clock {{.+}} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %[[FORCED_VALUE:.+]] = firrtl.reg %clock {{.+}} : !firrtl.clock, !firrtl.uint<4>
    
    // Should generate priority mux for forceWins
    // CHECK: firrtl.mux
    
    // Should generate OR gates for enables
    // CHECK: firrtl.or
    
    // Should generate AND for forceActive and releaseActive
    // CHECK: firrtl.and
    
    // Should generate when/else structure
    // CHECK: firrtl.when {{%.+}} : !firrtl.uint<1> {
    // CHECK:   firrtl.matchingconnect %[[FORCED]]
    // CHECK:   firrtl.matchingconnect %[[FORCED_VALUE]]
    // CHECK: } else {
    // CHECK:   firrtl.when {{%.+}} : !firrtl.uint<1> {
    // CHECK:     firrtl.matchingconnect %[[FORCED]]
    
    // Should generate override logic
    // CHECK: firrtl.when %[[FORCED]] : !firrtl.uint<1> {
    // CHECK:   firrtl.connect %w, %[[FORCED_VALUE]]
  }
}

// -----

// Test multiple forces with priority resolution
// CHECK-LABEL: firrtl.circuit "MultipleForces"
firrtl.circuit "MultipleForces" {
  firrtl.module @MultipleForces(in %clock: !firrtl.clock, in %cond1: !firrtl.uint<1>, in %cond2: !firrtl.uint<1>, in %val1: !firrtl.uint<4>, in %val2: !firrtl.uint<4>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    
    firrtl.ref.force %clock, %cond1, %w_ref, %val1 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
    firrtl.ref.force %clock, %cond2, %w_ref, %val2 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
    
    // Should generate priority mux for force value selection
    // CHECK: firrtl.mux
    // CHECK: firrtl.mux
    
    // Should generate OR to combine enables
    // CHECK: firrtl.or
  }
}

// -----

// Test force_initial operation
// CHECK-LABEL: firrtl.circuit "ForceInitial"
firrtl.circuit "ForceInitial" {
  firrtl.module @ForceInitial(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>, in %value: !firrtl.uint<4>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    
    // Mix force and force_initial - force provides the clock
    firrtl.ref.force %clock, %cond, %w_ref, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
    firrtl.ref.force_initial %cond, %w_ref, %value : !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
    
    // Should still generate registers with the clock from force
    // CHECK: firrtl.reg %clock
  }
}

// -----

// Test release_initial operation  
// CHECK-LABEL: firrtl.circuit "ReleaseInitial"
firrtl.circuit "ReleaseInitial" {
  firrtl.module @ReleaseInitial(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    
    firrtl.ref.force %clock, %cond, %w_ref, %w : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
    firrtl.ref.release_initial %cond, %w_ref : !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>
    
    // Should generate registers
    // CHECK: firrtl.reg %clock
  }
}

// -----

// Test RWProbe via rwprobe operation
// CHECK-LABEL: firrtl.circuit "RWProbeTarget"
firrtl.circuit "RWProbeTarget" {
  firrtl.module @RWProbeTarget(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>, in %value: !firrtl.uint<4>) {
    %w = firrtl.wire sym @w : !firrtl.uint<4>
    %rwp = firrtl.ref.rwprobe <@RWProbeTarget::@w> : !firrtl.rwprobe<uint<4>>

    firrtl.ref.force %clock, %cond, %rwp, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>

    // CHECK: firrtl.reg %clock
    // CHECK: firrtl.when
  }
}

// -----

// Test complete generated structure with detailed checks
// CHECK-LABEL: firrtl.circuit "DetailedStructure"
firrtl.circuit "DetailedStructure" {
  firrtl.module @DetailedStructure(in %clock: !firrtl.clock, in %forceCond: !firrtl.uint<1>, in %releaseCond: !firrtl.uint<1>, in %forceValue: !firrtl.uint<8>) {
    %target, %target_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>

    firrtl.ref.force %clock, %forceCond, %target_ref, %forceValue : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.release %clock, %releaseCond, %target_ref : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // Check constants for priority mux
    // CHECK: %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>

    // Check priority mux for forceWins (determines force vs release priority)
    // CHECK: %[[MUX1:.+]] = firrtl.mux(%forceCond, %c1_ui1, %c0_ui1)
    // CHECK: %[[FORCE_WINS:.+]] = firrtl.mux(%releaseCond, %c0_ui1, %[[MUX1]])

    // Check invalid value creation for default force value
    // CHECK: %invalid_ui8 = firrtl.invalidvalue : !firrtl.uint<8>

    // Check priority mux for force value selection
    // CHECK: %[[FORCE_VALUE:.+]] = firrtl.mux(%forceCond, %forceValue, %invalid_ui8)

    // Check state register creation
    // CHECK: %[[FORCED:.+]] = firrtl.reg %clock {name = "forced"} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %[[FORCED_VALUE:.+]] = firrtl.reg %clock {name = "forcedValue"} : !firrtl.clock, !firrtl.uint<8>

    // Check forceActive computation
    // CHECK: %[[FORCE_ACTIVE:.+]] = firrtl.and %forceCond, %[[FORCE_WINS]]

    // Check releaseActive computation
    // CHECK: %[[NOT_FORCE_WINS:.+]] = firrtl.not %[[FORCE_WINS]]
    // CHECK: %[[RELEASE_ACTIVE:.+]] = firrtl.and %releaseCond, %[[NOT_FORCE_WINS]]

    // Check state update logic
    // CHECK: firrtl.when %[[FORCE_ACTIVE]] : !firrtl.uint<1> {
    // CHECK:   firrtl.matchingconnect %[[FORCED]], %c1_ui1
    // CHECK:   firrtl.matchingconnect %[[FORCED_VALUE]], %[[FORCE_VALUE]]
    // CHECK: } else {
    // CHECK:   firrtl.when %[[RELEASE_ACTIVE]] : !firrtl.uint<1> {
    // CHECK:     firrtl.matchingconnect %[[FORCED]], %c0_ui1

    // Check override logic
    // CHECK: firrtl.when %[[FORCED]] : !firrtl.uint<1> {
    // CHECK:   firrtl.connect %target, %[[FORCED_VALUE]]
  }
}

// -----

// Test that original force/release operations are deleted
// CHECK-LABEL: firrtl.circuit "OpsDeleted"
firrtl.circuit "OpsDeleted" {
  firrtl.module @OpsDeleted(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>, in %value: !firrtl.uint<2>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>

    firrtl.ref.force %clock, %cond, %w_ref, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<2>>, !firrtl.uint<2>

    // CHECK-NOT: firrtl.ref.force
    // CHECK-NOT: firrtl.ref.release
    // CHECK-NOT: firrtl.ref.force_initial
    // CHECK-NOT: firrtl.ref.release_initial
  }
}

// -----

//===----------------------------------------------------------------------===//
// Hierarchical Tests
//===----------------------------------------------------------------------===//

// Test simple two-level hierarchy: Parent forces child's probe
// CHECK-LABEL: firrtl.circuit "SimpleHierarchy"
firrtl.circuit "SimpleHierarchy" {
  // CHECK: firrtl.module @Child(out %probe: !firrtl.uint<4>)
  firrtl.module @Child(out %probe: !firrtl.rwprobe<uint<4>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    firrtl.ref.define %probe, %w_ref : !firrtl.rwprobe<uint<4>>
    // CHECK: firrtl.matchingconnect %probe, %w
  }

  firrtl.module @SimpleHierarchy(in %clock: !firrtl.clock, in %forceCond: !firrtl.uint<1>, in %forceValue: !firrtl.uint<4>) {
    // CHECK: %child_probe = firrtl.instance child @Child(out probe: !firrtl.uint<4>)
    %child_probe = firrtl.instance child @Child(out probe: !firrtl.rwprobe<uint<4>>)

    // Parent forces child's probe
    firrtl.ref.force %clock, %forceCond, %child_probe, %forceValue : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>

    // CHECK: %[[FORCED:.+]] = firrtl.reg %clock {name = "forced"} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %[[FORCED_VALUE:.+]] = firrtl.reg %clock {name = "forcedValue"} : !firrtl.clock, !firrtl.uint<4>
    // CHECK: firrtl.when %[[FORCED]] : !firrtl.uint<1> {
    // CHECK:   firrtl.connect %child_probe, %[[FORCED_VALUE]]
  }
}

// -----

// Test three-level hierarchy: Grandparent -> Parent -> Child
// CHECK-LABEL: firrtl.circuit "ThreeLevelHierarchy"
firrtl.circuit "ThreeLevelHierarchy" {
  // Leaf module with forceable wire
  // CHECK: firrtl.module @Leaf(out %p: !firrtl.uint<2>)
  firrtl.module @Leaf(out %p: !firrtl.rwprobe<uint<2>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<2>>
    // CHECK: firrtl.matchingconnect %p, %w
  }

  // Middle module that forwards probe
  // CHECK: firrtl.module @Middle(out %p: !firrtl.uint<2>)
  firrtl.module @Middle(out %p: !firrtl.rwprobe<uint<2>>) {
    %leaf_p = firrtl.instance leaf @Leaf(out p: !firrtl.rwprobe<uint<2>>)
    firrtl.ref.define %p, %leaf_p : !firrtl.rwprobe<uint<2>>
    // CHECK: firrtl.matchingconnect %p, %leaf_p
  }

  // Top module forces the probe from leaf
  firrtl.module @ThreeLevelHierarchy(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>, in %value: !firrtl.uint<2>) {
    %middle_p = firrtl.instance middle @Middle(out p: !firrtl.rwprobe<uint<2>>)

    firrtl.ref.force %clock, %cond, %middle_p, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<2>>, !firrtl.uint<2>

    // CHECK: firrtl.reg %clock
  }
}

// -----

// Test multiple children with independent forces
// CHECK-LABEL: firrtl.circuit "MultipleChildren"
firrtl.circuit "MultipleChildren" {
  // CHECK: firrtl.module @ChildA(out %p: !firrtl.uint<4>)
  firrtl.module @ChildA(out %p: !firrtl.rwprobe<uint<4>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<4>>
  }

  // CHECK: firrtl.module @ChildB(out %p: !firrtl.uint<4>)
  firrtl.module @ChildB(out %p: !firrtl.rwprobe<uint<4>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<4>>
  }

  firrtl.module @MultipleChildren(in %clock: !firrtl.clock, in %condA: !firrtl.uint<1>, in %condB: !firrtl.uint<1>, in %valA: !firrtl.uint<4>, in %valB: !firrtl.uint<4>) {
    %childA_p = firrtl.instance childA @ChildA(out p: !firrtl.rwprobe<uint<4>>)
    %childB_p = firrtl.instance childB @ChildB(out p: !firrtl.rwprobe<uint<4>>)

    // Force both children independently
    firrtl.ref.force %clock, %condA, %childA_p, %valA : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
    firrtl.ref.force %clock, %condB, %childB_p, %valB : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>

    // Should generate separate state registers for each probe
    // CHECK-DAG: firrtl.reg %clock {name = "forced"}
    // CHECK-DAG: firrtl.reg %clock {name = "forcedValue"}
  }
}

// -----

// Test force at multiple hierarchy levels (child and parent both force)
// CHECK-LABEL: firrtl.circuit "ForceAtMultipleLevels"
firrtl.circuit "ForceAtMultipleLevels" {
  // CHECK: firrtl.module @LeafWithForce(in %clock: !firrtl.clock, in %localCond: !firrtl.uint<1>, in %localValue: !firrtl.uint<3>, out %p: !firrtl.uint<3>)
  firrtl.module @LeafWithForce(in %clock: !firrtl.clock, in %localCond: !firrtl.uint<1>, in %localValue: !firrtl.uint<3>, out %p: !firrtl.rwprobe<uint<3>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<3>, !firrtl.rwprobe<uint<3>>

    // Leaf forces its own wire
    firrtl.ref.force %clock, %localCond, %w_ref, %localValue : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<3>>, !firrtl.uint<3>

    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<3>>

    // CHECK: firrtl.reg %clock {name = "forced"}
  }

  firrtl.module @ForceAtMultipleLevels(in %clock: !firrtl.clock, in %parentCond: !firrtl.uint<1>, in %childCond: !firrtl.uint<1>, in %parentValue: !firrtl.uint<3>, in %childValue: !firrtl.uint<3>) {
    %leaf_p = firrtl.instance leaf @LeafWithForce(in clock: !firrtl.clock, in localCond: !firrtl.uint<1>, in localValue: !firrtl.uint<3>, out p: !firrtl.rwprobe<uint<3>>)
    firrtl.matchingconnect %leaf_p#0, %clock : !firrtl.clock
    firrtl.matchingconnect %leaf_p#1, %childCond : !firrtl.uint<1>
    firrtl.matchingconnect %leaf_p#2, %childValue : !firrtl.uint<3>

    // Parent also forces the same probe
    firrtl.ref.force %clock, %parentCond, %leaf_p#3, %parentValue : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<3>>, !firrtl.uint<3>

    // CHECK: firrtl.reg %clock {name = "forced"}
  }
}

// -----

// Test diamond hierarchy: Two paths to same leaf
// CHECK-LABEL: firrtl.circuit "DiamondHierarchy"
firrtl.circuit "DiamondHierarchy" {
  // Shared leaf module
  // CHECK: firrtl.module @SharedLeaf(out %p: !firrtl.uint<2>)
  firrtl.module @SharedLeaf(out %p: !firrtl.rwprobe<uint<2>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<2>>
  }

  // Left path
  // CHECK: firrtl.module @LeftPath(out %p: !firrtl.uint<2>)
  firrtl.module @LeftPath(out %p: !firrtl.rwprobe<uint<2>>) {
    %leaf_p = firrtl.instance leaf @SharedLeaf(out p: !firrtl.rwprobe<uint<2>>)
    firrtl.ref.define %p, %leaf_p : !firrtl.rwprobe<uint<2>>
  }

  // Right path
  // CHECK: firrtl.module @RightPath(out %p: !firrtl.uint<2>)
  firrtl.module @RightPath(out %p: !firrtl.rwprobe<uint<2>>) {
    %leaf_p = firrtl.instance leaf @SharedLeaf(out p: !firrtl.rwprobe<uint<2>>)
    firrtl.ref.define %p, %leaf_p : !firrtl.rwprobe<uint<2>>
  }

  // Top can force through either path
  firrtl.module @DiamondHierarchy(in %clock: !firrtl.clock, in %leftCond: !firrtl.uint<1>, in %rightCond: !firrtl.uint<1>, in %leftVal: !firrtl.uint<2>, in %rightVal: !firrtl.uint<2>) {
    %left_p = firrtl.instance left @LeftPath(out p: !firrtl.rwprobe<uint<2>>)
    %right_p = firrtl.instance right @RightPath(out p: !firrtl.rwprobe<uint<2>>)

    // Force through left path
    firrtl.ref.force %clock, %leftCond, %left_p, %leftVal : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<2>>, !firrtl.uint<2>

    // Force through right path
    firrtl.ref.force %clock, %rightCond, %right_p, %rightVal : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<2>>, !firrtl.uint<2>

    // Should generate registers for each force
    // CHECK: firrtl.reg %clock
  }
}

// -----

// Test deep hierarchy (5 levels)
// CHECK-LABEL: firrtl.circuit "DeepHierarchy"
firrtl.circuit "DeepHierarchy" {
  // CHECK: firrtl.module @Level4(out %p: !firrtl.uint<1>)
  firrtl.module @Level4(out %p: !firrtl.rwprobe<uint<1>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<1>>
  }

  // CHECK: firrtl.module @Level3(out %p: !firrtl.uint<1>)
  firrtl.module @Level3(out %p: !firrtl.rwprobe<uint<1>>) {
    %l4_p = firrtl.instance l4 @Level4(out p: !firrtl.rwprobe<uint<1>>)
    firrtl.ref.define %p, %l4_p : !firrtl.rwprobe<uint<1>>
  }

  // CHECK: firrtl.module @Level2(out %p: !firrtl.uint<1>)
  firrtl.module @Level2(out %p: !firrtl.rwprobe<uint<1>>) {
    %l3_p = firrtl.instance l3 @Level3(out p: !firrtl.rwprobe<uint<1>>)
    firrtl.ref.define %p, %l3_p : !firrtl.rwprobe<uint<1>>
  }

  // CHECK: firrtl.module @Level1(out %p: !firrtl.uint<1>)
  firrtl.module @Level1(out %p: !firrtl.rwprobe<uint<1>>) {
    %l2_p = firrtl.instance l2 @Level2(out p: !firrtl.rwprobe<uint<1>>)
    firrtl.ref.define %p, %l2_p : !firrtl.rwprobe<uint<1>>
  }

  // Top level (Level 0) forces probe from Level 4
  firrtl.module @DeepHierarchy(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>, in %value: !firrtl.uint<1>) {
    %l1_p = firrtl.instance l1 @Level1(out p: !firrtl.rwprobe<uint<1>>)

    firrtl.ref.force %clock, %cond, %l1_p, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>

    // CHECK: firrtl.reg %clock
  }
}

// -----

// Test multiple instances of same module
// CHECK-LABEL: firrtl.circuit "MultipleInstances"
firrtl.circuit "MultipleInstances" {
  // Module that will be instantiated multiple times
  // CHECK: firrtl.module @ReusableModule(out %p: !firrtl.uint<4>)
  firrtl.module @ReusableModule(out %p: !firrtl.rwprobe<uint<4>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<4>>
  }

  firrtl.module @MultipleInstances(in %clock: !firrtl.clock, in %cond1: !firrtl.uint<1>, in %cond2: !firrtl.uint<1>, in %cond3: !firrtl.uint<1>, in %val1: !firrtl.uint<4>, in %val2: !firrtl.uint<4>, in %val3: !firrtl.uint<4>) {
    // Three instances of the same module
    %inst1_p = firrtl.instance inst1 @ReusableModule(out p: !firrtl.rwprobe<uint<4>>)
    %inst2_p = firrtl.instance inst2 @ReusableModule(out p: !firrtl.rwprobe<uint<4>>)
    %inst3_p = firrtl.instance inst3 @ReusableModule(out p: !firrtl.rwprobe<uint<4>>)

    // Force each instance independently
    firrtl.ref.force %clock, %cond1, %inst1_p, %val1 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
    firrtl.ref.force %clock, %cond2, %inst2_p, %val2 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
    firrtl.ref.force %clock, %cond3, %inst3_p, %val3 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>

    // Should generate separate registers for each force
    // CHECK: firrtl.reg %clock
  }
}

// -----

// Test hierarchical force and release
// CHECK-LABEL: firrtl.circuit "HierarchicalForceRelease"
firrtl.circuit "HierarchicalForceRelease" {
  // CHECK: firrtl.module @ChildWithProbe(out %p: !firrtl.uint<3>)
  firrtl.module @ChildWithProbe(out %p: !firrtl.rwprobe<uint<3>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<3>, !firrtl.rwprobe<uint<3>>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<3>>
  }

  firrtl.module @HierarchicalForceRelease(in %clock: !firrtl.clock, in %forceCond: !firrtl.uint<1>, in %releaseCond: !firrtl.uint<1>, in %value: !firrtl.uint<3>) {
    %child_p = firrtl.instance child @ChildWithProbe(out p: !firrtl.rwprobe<uint<3>>)

    // Force and release from parent level
    firrtl.ref.force %clock, %forceCond, %child_p, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<3>>, !firrtl.uint<3>
    firrtl.ref.release %clock, %releaseCond, %child_p : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<3>>

    // Should generate forced and forcedValue registers
    // CHECK: %[[FORCED:.+]] = firrtl.reg %clock {name = "forced"} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %[[FORCED_VALUE:.+]] = firrtl.reg %clock {name = "forcedValue"} : !firrtl.clock, !firrtl.uint<3>

    // Should generate forceActive and releaseActive
    // CHECK: firrtl.and

    // Should generate state update logic
    // CHECK: firrtl.when {{%.+}} : !firrtl.uint<1> {
    // CHECK: } else {
    // CHECK:   firrtl.when {{%.+}} : !firrtl.uint<1> {
  }
}

// -----

// Test complex hierarchy with mixed operations (force, release, resolve)
// CHECK-LABEL: firrtl.circuit "ComplexMixedOps"
firrtl.circuit "ComplexMixedOps" {
  // Leaf with multiple probes
  // CHECK: firrtl.module @LeafMultiProbe(out %p1: !firrtl.uint<2>, out %p2: !firrtl.uint<2>)
  firrtl.module @LeafMultiProbe(out %p1: !firrtl.rwprobe<uint<2>>, out %p2: !firrtl.rwprobe<uint<2>>) {
    %w1, %w1_ref = firrtl.wire forceable : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
    %w2, %w2_ref = firrtl.wire forceable : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
    firrtl.ref.define %p1, %w1_ref : !firrtl.rwprobe<uint<2>>
    firrtl.ref.define %p2, %w2_ref : !firrtl.rwprobe<uint<2>>
  }

  // Middle layer that observes one probe and forwards both
  // CHECK: firrtl.module @MiddleObserver(out %p1: !firrtl.uint<2>, out %p2: !firrtl.uint<2>, out %observed: !firrtl.uint<2>)
  firrtl.module @MiddleObserver(out %p1: !firrtl.rwprobe<uint<2>>, out %p2: !firrtl.rwprobe<uint<2>>, out %observed: !firrtl.uint<2>) {
    %leaf_p1, %leaf_p2 = firrtl.instance leaf @LeafMultiProbe(out p1: !firrtl.rwprobe<uint<2>>, out p2: !firrtl.rwprobe<uint<2>>)

    // Observe p1 by converting to read-only probe
    %ro_probe = firrtl.ref.cast %leaf_p1 : (!firrtl.rwprobe<uint<2>>) -> !firrtl.probe<uint<2>>
    %resolved = firrtl.ref.resolve %ro_probe : !firrtl.probe<uint<2>>
    firrtl.matchingconnect %observed, %resolved : !firrtl.uint<2>

    // Forward both probes
    firrtl.ref.define %p1, %leaf_p1 : !firrtl.rwprobe<uint<2>>
    firrtl.ref.define %p2, %leaf_p2 : !firrtl.rwprobe<uint<2>>
  }

  // Top forces both, releases one
  firrtl.module @ComplexMixedOps(in %clock: !firrtl.clock, in %force1: !firrtl.uint<1>, in %force2: !firrtl.uint<1>, in %release1: !firrtl.uint<1>, in %val1: !firrtl.uint<2>, in %val2: !firrtl.uint<2>) {
    %mid_p1, %mid_p2, %mid_obs = firrtl.instance middle @MiddleObserver(out p1: !firrtl.rwprobe<uint<2>>, out p2: !firrtl.rwprobe<uint<2>>, out observed: !firrtl.uint<2>)

    // Force p1 and release it
    firrtl.ref.force %clock, %force1, %mid_p1, %val1 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<2>>, !firrtl.uint<2>
    firrtl.ref.release %clock, %release1, %mid_p1 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<2>>

    // Force p2 only
    firrtl.ref.force %clock, %force2, %mid_p2, %val2 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<2>>, !firrtl.uint<2>

    // Should generate separate register sets for each probe
    // CHECK-DAG: firrtl.reg %clock {name = "forced"}
    // CHECK-DAG: firrtl.reg %clock {name = "forcedValue"}
  }
}

// -----

// Test hierarchical force with input and output probes
// CHECK-LABEL: firrtl.circuit "BidirectionalProbes"
firrtl.circuit "BidirectionalProbes" {
  // Child accepts probe input and provides probe output
  // CHECK: firrtl.module @BidirChild(in %pin: !firrtl.uint<4>, out %pout: !firrtl.uint<4>)
  firrtl.module @BidirChild(in %pin: !firrtl.rwprobe<uint<4>>, out %pout: !firrtl.rwprobe<uint<4>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    firrtl.ref.define %pout, %w_ref : !firrtl.rwprobe<uint<4>>

    // Child can also force based on input probe
    // (In practice, would do something with %pin here)
  }

  firrtl.module @BidirectionalProbes(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>, in %value: !firrtl.uint<4>) {
    %w_local, %w_local_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    %child_pin, %child_pout = firrtl.instance child @BidirChild(in pin: !firrtl.rwprobe<uint<4>>, out pout: !firrtl.rwprobe<uint<4>>)

    // Connect local probe to child input
    firrtl.ref.define %child_pin, %w_local_ref : !firrtl.rwprobe<uint<4>>

    // Force child's output probe
    firrtl.ref.force %clock, %cond, %child_pout, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>

    // CHECK: firrtl.reg %clock
  }
}

// -----

// Test large hierarchy with force at intermediate level
// CHECK-LABEL: firrtl.circuit "IntermediateForce"
firrtl.circuit "IntermediateForce" {
  // Bottom level
  // CHECK: firrtl.module @Bottom(out %p: !firrtl.uint<2>)
  firrtl.module @Bottom(out %p: !firrtl.rwprobe<uint<2>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<2>>
  }

  // Intermediate level that forces
  // CHECK: firrtl.module @IntermediateForcer(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>, in %value: !firrtl.uint<2>, out %p: !firrtl.uint<2>)
  firrtl.module @IntermediateForcer(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>, in %value: !firrtl.uint<2>, out %p: !firrtl.rwprobe<uint<2>>) {
    %bottom_p = firrtl.instance bottom @Bottom(out p: !firrtl.rwprobe<uint<2>>)

    // Intermediate forces the probe
    firrtl.ref.force %clock, %cond, %bottom_p, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<2>>, !firrtl.uint<2>

    // And forwards it up
    firrtl.ref.define %p, %bottom_p : !firrtl.rwprobe<uint<2>>

    // CHECK: firrtl.reg %clock {name = "forced"}
  }

  // Top level also forces same probe
  firrtl.module @IntermediateForce(in %clock: !firrtl.clock, in %topCond: !firrtl.uint<1>, in %midCond: !firrtl.uint<1>, in %topValue: !firrtl.uint<2>, in %midValue: !firrtl.uint<2>) {
    %inter_p = firrtl.instance inter @IntermediateForcer(in clock: !firrtl.clock, in cond: !firrtl.uint<1>, in value: !firrtl.uint<2>, out p: !firrtl.rwprobe<uint<2>>)
    firrtl.matchingconnect %inter_p#0, %clock : !firrtl.clock
    firrtl.matchingconnect %inter_p#1, %midCond : !firrtl.uint<1>
    firrtl.matchingconnect %inter_p#2, %midValue : !firrtl.uint<2>

    // Top also forces
    firrtl.ref.force %clock, %topCond, %inter_p#3, %topValue : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<2>>, !firrtl.uint<2>

    // CHECK: firrtl.reg %clock {name = "forced"}
  }
}

// -----

// Test tree hierarchy: One parent, multiple levels of children
// CHECK-LABEL: firrtl.circuit "TreeHierarchy"
firrtl.circuit "TreeHierarchy" {
  // Leaf nodes
  // CHECK: firrtl.module @TreeLeaf(out %p: !firrtl.uint<1>)
  firrtl.module @TreeLeaf(out %p: !firrtl.rwprobe<uint<1>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<1>>
  }

  // Branch nodes (each has 2 leaf children)
  // CHECK: firrtl.module @TreeBranch(out %p1: !firrtl.uint<1>, out %p2: !firrtl.uint<1>)
  firrtl.module @TreeBranch(out %p1: !firrtl.rwprobe<uint<1>>, out %p2: !firrtl.rwprobe<uint<1>>) {
    %leaf1_p = firrtl.instance leaf1 @TreeLeaf(out p: !firrtl.rwprobe<uint<1>>)
    %leaf2_p = firrtl.instance leaf2 @TreeLeaf(out p: !firrtl.rwprobe<uint<1>>)
    firrtl.ref.define %p1, %leaf1_p : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %p2, %leaf2_p : !firrtl.rwprobe<uint<1>>
  }

  // Root with 2 branches (total 4 leaf nodes accessible)
  firrtl.module @TreeHierarchy(in %clock: !firrtl.clock, in %c1: !firrtl.uint<1>, in %c2: !firrtl.uint<1>, in %c3: !firrtl.uint<1>, in %c4: !firrtl.uint<1>, in %v: !firrtl.uint<1>) {
    %branchA_p1, %branchA_p2 = firrtl.instance branchA @TreeBranch(out p1: !firrtl.rwprobe<uint<1>>, out p2: !firrtl.rwprobe<uint<1>>)
    %branchB_p1, %branchB_p2 = firrtl.instance branchB @TreeBranch(out p1: !firrtl.rwprobe<uint<1>>, out p2: !firrtl.rwprobe<uint<1>>)

    // Force all 4 leaf probes from root
    firrtl.ref.force %clock, %c1, %branchA_p1, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
    firrtl.ref.force %clock, %c2, %branchA_p2, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
    firrtl.ref.force %clock, %c3, %branchB_p1, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
    firrtl.ref.force %clock, %c4, %branchB_p2, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>

    // Should generate 4 sets of state registers
    // CHECK: firrtl.reg %clock
  }
}

// -----

// Test hierarchical force with bundle types
// CHECK-LABEL: firrtl.circuit "HierarchicalBundle"
firrtl.circuit "HierarchicalBundle" {
  // CHECK: firrtl.module @BundleChild(out %p: !firrtl.bundle<a: uint<2>, b: uint<3>>)
  firrtl.module @BundleChild(out %p: !firrtl.rwprobe<bundle<a: uint<2>, b: uint<3>>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.bundle<a: uint<2>, b: uint<3>>, !firrtl.rwprobe<bundle<a: uint<2>, b: uint<3>>>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<bundle<a: uint<2>, b: uint<3>>>
  }

  firrtl.module @HierarchicalBundle(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>, in %value: !firrtl.bundle<a: uint<2>, b: uint<3>>) {
    %child_p = firrtl.instance child @BundleChild(out p: !firrtl.rwprobe<bundle<a: uint<2>, b: uint<3>>>)

    firrtl.ref.force %clock, %cond, %child_p, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<bundle<a: uint<2>, b: uint<3>>>, !firrtl.bundle<a: uint<2>, b: uint<3>>

    // CHECK: firrtl.reg %clock {name = "forced"} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: firrtl.reg %clock {name = "forcedValue"} : !firrtl.clock, !firrtl.bundle<a: uint<2>, b: uint<3>>
  }
}
