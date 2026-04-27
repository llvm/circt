// RUN: circt-opt --lower-firrtl-to-hw %s | FileCheck %s

firrtl.circuit "XMRRemoteBasic" {
  // CHECK-LABEL: hw.module @XMRRemoteBasic
  firrtl.module @XMRRemoteBasic() {
    %w = firrtl.wire sym @myWire : !firrtl.uint<1>
    // CHECK: %[[WIRE:.+]] = hw.wire %{{.+}} sym @myWire
    %0 = firrtl.xmr.remote @XMRRemoteBasic::@myWire : !firrtl.rwprobe<uint<1>>
    // CHECK: %[[XMR:.+]] = hw.xmr.remote @XMRRemoteBasic::@myWire : !hw.rwprobe<i1>
  }
}

// -----

firrtl.circuit "XMRRemoteFromExtModule" {
  firrtl.extmodule @Bar(out rwprobe: !firrtl.rwprobe<uint<8>> sym @probePort)

  // CHECK-LABEL: hw.module @XMRRemoteFromExtModule
  firrtl.module @XMRRemoteFromExtModule(in %clk: !firrtl.clock) {
    %bar_rwprobe = firrtl.instance bar @Bar(out rwprobe: !firrtl.rwprobe<uint<8>>)

    %0 = firrtl.xmr.remote @Bar::@probePort : !firrtl.rwprobe<uint<8>>
    // CHECK: hw.xmr.remote @Bar::@probePort : !hw.rwprobe<i8>
  }
}

// -----

firrtl.circuit "XMRRemoteToReg" {
  // CHECK-LABEL: hw.module @XMRRemoteToReg
  firrtl.module @XMRRemoteToReg(in %clk: !firrtl.clock) {
    %reg = firrtl.reg sym @myReg %clk : !firrtl.clock, !firrtl.uint<16>
    // CHECK: %[[REG:.+]] = seq.compreg sym @myReg
    %0 = firrtl.xmr.remote @XMRRemoteToReg::@myReg : !firrtl.rwprobe<uint<16>>
    // CHECK: %[[XMR:.+]] = hw.xmr.remote @XMRRemoteToReg::@myReg : !hw.rwprobe<i16>
  }
}

// -----

firrtl.circuit "XMRRemoteMultiple" {
  // CHECK-LABEL: hw.module @XMRRemoteMultiple
  firrtl.module @XMRRemoteMultiple() {
    %w1 = firrtl.wire sym @wire1 : !firrtl.uint<4>
    %w2 = firrtl.wire sym @wire2 : !firrtl.uint<8>
    // CHECK: hw.wire %{{.+}} sym @wire1
    // CHECK: hw.wire %{{.+}} sym @wire2
    
    %xmr1 = firrtl.xmr.remote @XMRRemoteMultiple::@wire1 : !firrtl.rwprobe<uint<4>>
    %xmr2 = firrtl.xmr.remote @XMRRemoteMultiple::@wire2 : !firrtl.rwprobe<uint<8>>
    // CHECK: hw.xmr.remote @XMRRemoteMultiple::@wire1 : !hw.rwprobe<i4>
    // CHECK: hw.xmr.remote @XMRRemoteMultiple::@wire2 : !hw.rwprobe<i8>
  }
}
