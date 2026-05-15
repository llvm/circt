// RUN: circt-opt --firrtl-probes-to-signals --split-input-file %s | FileCheck %s

// Test force/release with ref.send and ref.define
// CHECK-LABEL: firrtl.circuit "ForceWithRefDefine"
firrtl.circuit "ForceWithRefDefine" {
  firrtl.module @Child(out %p: !firrtl.rwprobe<uint<4>>) {
    // CHECK: @Child(out %p: !firrtl.uint<4>)
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<4>>
    // CHECK: firrtl.matchingconnect %p, %w
  }
  
  firrtl.module @ForceWithRefDefine(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>, in %value: !firrtl.uint<4>) {
    %child_p = firrtl.instance child @Child(out p: !firrtl.rwprobe<uint<4>>)
    // CHECK: %child_p = firrtl.instance child @Child(out p: !firrtl.uint<4>)
    
    firrtl.ref.force %clock, %cond, %child_p, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
    
    // CHECK: %[[FORCED:.+]] = firrtl.reg %clock
    // CHECK: firrtl.when
  }
}

// -----

// Test force on register
// CHECK-LABEL: firrtl.circuit "ForceOnRegister"
firrtl.circuit "ForceOnRegister" {
  firrtl.module @ForceOnRegister(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>, in %value: !firrtl.uint<4>, in %data: !firrtl.uint<4>) {
    %r, %r_ref = firrtl.reg %clock forceable : !firrtl.clock, !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    // CHECK: %r, %r_ref = firrtl.reg %clock forceable
    
    firrtl.matchingconnect %r, %data : !firrtl.uint<4>
    
    firrtl.ref.force %clock, %cond, %r_ref, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
    
    // Should generate forced/forcedValue registers
    // CHECK: %[[FORCED:.+]] = firrtl.reg %clock {{.+}} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %[[FORCED_VALUE:.+]] = firrtl.reg %clock {{.+}} : !firrtl.clock, !firrtl.uint<4>
    
    // Override should connect to the register
    // CHECK: firrtl.when %[[FORCED]] : !firrtl.uint<1> {
    // CHECK:   firrtl.connect %r, %[[FORCED_VALUE]]
  }
}

// -----

// Test force with ref.resolve
// CHECK-LABEL: firrtl.circuit "ForceWithResolve"
firrtl.circuit "ForceWithResolve" {
  firrtl.module @ForceWithResolve(in %clock: !firrtl.clock, in %forceCond: !firrtl.uint<1>, in %value: !firrtl.uint<4>, out %observed: !firrtl.uint<4>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    
    // Send the reference somewhere
    %1 = firrtl.ref.send %w : !firrtl.uint<4>
    
    // Create a read-only probe and resolve it
    %2 = firrtl.ref.cast %w_ref : (!firrtl.rwprobe<uint<4>>) -> !firrtl.probe<uint<4>>
    %resolved = firrtl.ref.resolve %2 : !firrtl.probe<uint<4>>
    // CHECK: firrtl.matchingconnect %observed, %w
    firrtl.matchingconnect %observed, %resolved : !firrtl.uint<4>
    
    // Force the wire
    firrtl.ref.force %clock, %forceCond, %w_ref, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
    
    // CHECK: firrtl.reg %clock
  }
}

// -----

// Test multiple modules with force/release
// CHECK-LABEL: firrtl.circuit "MultiModule"
firrtl.circuit "MultiModule" {
  firrtl.module @Leaf(out %p: !firrtl.rwprobe<uint<2>>) {
    // CHECK: @Leaf(out %p: !firrtl.uint<2>)
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<2>>
  }
  
  firrtl.module @Middle(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>, in %val: !firrtl.uint<2>, out %p: !firrtl.rwprobe<uint<2>>) {
    // CHECK: @Middle(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>, in %val: !firrtl.uint<2>, out %p: !firrtl.uint<2>)
    %leaf_p = firrtl.instance leaf @Leaf(out p: !firrtl.rwprobe<uint<2>>)
    
    // Force in middle module
    firrtl.ref.force %clock, %cond, %leaf_p, %val : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<2>>, !firrtl.uint<2>
    
    // Forward the probe
    firrtl.ref.define %p, %leaf_p : !firrtl.rwprobe<uint<2>>
    
    // CHECK: firrtl.reg %clock
  }
  
  firrtl.module @MultiModule(in %clock: !firrtl.clock, in %cond1: !firrtl.uint<1>, in %cond2: !firrtl.uint<1>, in %val1: !firrtl.uint<2>, in %val2: !firrtl.uint<2>) {
    %middle_p = firrtl.instance middle @Middle(in clock: !firrtl.clock, in cond: !firrtl.uint<1>, in val: !firrtl.uint<2>, out p: !firrtl.rwprobe<uint<2>>)
    firrtl.matchingconnect %middle_p#0, %clock : !firrtl.clock
    firrtl.matchingconnect %middle_p#1, %cond1 : !firrtl.uint<1>
    firrtl.matchingconnect %middle_p#2, %val1 : !firrtl.uint<2>
    
    // Force in top module too
    firrtl.ref.force %clock, %cond2, %middle_p#3, %val2 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<2>>, !firrtl.uint<2>
    
    // CHECK: firrtl.reg %clock
  }
}

// -----

// Test force with aggregate types
// CHECK-LABEL: firrtl.circuit "ForceBundle"
firrtl.circuit "ForceBundle" {
  firrtl.module @ForceBundle(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>, in %value: !firrtl.bundle<a: uint<2>, b: uint<3>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.bundle<a: uint<2>, b: uint<3>>, !firrtl.rwprobe<bundle<a: uint<2>, b: uint<3>>>
    
    firrtl.ref.force %clock, %cond, %w_ref, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<bundle<a: uint<2>, b: uint<3>>>, !firrtl.bundle<a: uint<2>, b: uint<3>>
    
    // Should generate registers for bundle type
    // CHECK: firrtl.reg %clock {name = "forced"} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: firrtl.reg %clock {name = "forcedValue"} : !firrtl.clock, !firrtl.bundle<a: uint<2>, b: uint<3>>
  }
}

// -----

// Test that forceable attribute is removed after processing
// CHECK-LABEL: firrtl.circuit "ForceableRemoved"
firrtl.circuit "ForceableRemoved" {
  firrtl.module @ForceableRemoved(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>, in %value: !firrtl.uint<4>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    // CHECK: %w = firrtl.wire
    // CHECK-NOT: forceable
    
    firrtl.ref.force %clock, %cond, %w_ref, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
  }
}
