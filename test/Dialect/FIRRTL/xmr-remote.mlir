// RUN: circt-opt %s --verify-diagnostics --split-input-file | FileCheck %s

// Test basic xmr.remote operation
// CHECK-LABEL: firrtl.circuit "XMRRemoteBasic"
firrtl.circuit "XMRRemoteBasic" {
  // CHECK: firrtl.module @XMRRemoteBasic
  firrtl.module @XMRRemoteBasic() {
    // CHECK: %w = firrtl.wire sym @myWire : !firrtl.uint<1>
    %w = firrtl.wire sym @myWire : !firrtl.uint<1>
    // CHECK: = firrtl.xmr.remote @XMRRemoteBasic::@myWire : !firrtl.rwprobe<uint<1>>
    %rwProbe = firrtl.xmr.remote @XMRRemoteBasic::@myWire : !firrtl.rwprobe<uint<1>>
  }
}

// -----

// Test xmr.remote with register
// CHECK-LABEL: firrtl.circuit "XMRRemoteReg"
firrtl.circuit "XMRRemoteReg" {
  firrtl.module @XMRRemoteReg(in %clk: !firrtl.clock) {
    // CHECK: %reg = firrtl.reg sym @myReg %clk : !firrtl.clock, !firrtl.uint<8>
    %reg = firrtl.reg sym @myReg %clk : !firrtl.clock, !firrtl.uint<8>
    // CHECK: = firrtl.xmr.remote @XMRRemoteReg::@myReg : !firrtl.rwprobe<uint<8>>
    %rwProbe = firrtl.xmr.remote @XMRRemoteReg::@myReg : !firrtl.rwprobe<uint<8>>
  }
}

// -----

// Test xmr.remote can be used with force operations
// CHECK-LABEL: firrtl.circuit "XMRRemoteForce"
firrtl.circuit "XMRRemoteForce" {
  firrtl.module @XMRRemoteForce(in %clk: !firrtl.clock, in %cond: !firrtl.uint<1>, in %val: !firrtl.uint<4>) {
    %w = firrtl.wire sym @myWire : !firrtl.uint<4>
    %rwProbe = firrtl.xmr.remote @XMRRemoteForce::@myWire : !firrtl.rwprobe<uint<4>>
    // CHECK: firrtl.ref.force %clk, %cond, %{{.*}}, %val
    firrtl.ref.force %clk, %cond, %rwProbe, %val : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
  }
}

// -----

// Test invalid target - target doesn't exist
firrtl.circuit "InvalidTarget" {
  firrtl.module @InvalidTarget() {
    %w = firrtl.wire : !firrtl.uint<1>
    // expected-error @+1 {{has target that cannot be resolved: #hw.innerNameRef<@InvalidTarget::@nonExistent>}}
    %rwProbe = firrtl.xmr.remote @InvalidTarget::@nonExistent : !firrtl.rwprobe<uint<1>>
  }
}

// -----

// Test cross-module xmr.remote
// CHECK-LABEL: firrtl.circuit "CrossModule"
firrtl.circuit "CrossModule" {
  firrtl.module @Child() {
    // CHECK: %w = firrtl.wire sym @childWire : !firrtl.uint<2>
    %w = firrtl.wire sym @childWire : !firrtl.uint<2>
  }

  firrtl.module @CrossModule() {
    firrtl.instance child @Child()
    // CHECK: = firrtl.xmr.remote @Child::@childWire : !firrtl.rwprobe<uint<2>>
    %rwProbe = firrtl.xmr.remote @Child::@childWire : !firrtl.rwprobe<uint<2>>
  }
}
