// RUN: circt-opt --lower-firrtl-to-hw -split-input-file %s | FileCheck %s

// Test basic xmr.remote targeting an instance result
firrtl.circuit "XMRRemoteBasic" {
  firrtl.extmodule @Child(out probe: !firrtl.rwprobe<uint<8>>)

  // CHECK: hw.hierpath private @XMRRemoteBasic_childInst_xmr_remote_path [@XMRRemoteBasic::@childInst]
  // CHECK-LABEL: hw.module @XMRRemoteBasic
  firrtl.module @XMRRemoteBasic() {
    %child_probe = firrtl.instance child sym @childInst @Child(out probe: !firrtl.rwprobe<uint<8>>)
    // CHECK: hw.instance "child" sym @childInst @Child

    %0 = firrtl.xmr.remote @XMRRemoteBasic::@childInst, 0 : !firrtl.rwprobe<uint<8>>
    // CHECK: hw.xmr.remote @XMRRemoteBasic_childInst_xmr_remote_path, 0 : !hw.rwprobe<i8>
  }
}

// -----

// Test xmr.remote targeting external module instance with multiple results
firrtl.circuit "XMRRemoteMultiResult" {
  firrtl.extmodule @Bar(out probe1: !firrtl.rwprobe<uint<8>>, out probe2: !firrtl.rwprobe<uint<16>>)

  // Both xmr.remote ops target the same instance, so they share the same hierpath
  // CHECK: hw.hierpath private @XMRRemoteMultiResult_barInst_xmr_remote_path [@XMRRemoteMultiResult::@barInst]
  // CHECK-LABEL: hw.module @XMRRemoteMultiResult
  firrtl.module @XMRRemoteMultiResult() {
    %bar_probe1, %bar_probe2 = firrtl.instance bar sym @barInst @Bar(out probe1: !firrtl.rwprobe<uint<8>>, out probe2: !firrtl.rwprobe<uint<16>>)
    // CHECK: hw.instance "bar" sym @barInst @Bar

    // Target first result (index 0)
    %xmr1 = firrtl.xmr.remote @XMRRemoteMultiResult::@barInst, 0 : !firrtl.rwprobe<uint<8>>
    // CHECK: hw.xmr.remote @XMRRemoteMultiResult_barInst_xmr_remote_path, 0 : !hw.rwprobe<i8>

    // Target second result (index 1)
    %xmr2 = firrtl.xmr.remote @XMRRemoteMultiResult::@barInst, 1 : !firrtl.rwprobe<uint<16>>
    // CHECK: hw.xmr.remote @XMRRemoteMultiResult_barInst_xmr_remote_path, 1 : !hw.rwprobe<i16>
  }
}

// -----

// Test xmr.remote targeting instance with single probe output
firrtl.circuit "XMRRemoteSingleProbe" {
  firrtl.extmodule @Source(out data: !firrtl.rwprobe<uint<32>>)

  // CHECK: hw.hierpath private @XMRRemoteSingleProbe_sourceInst_xmr_remote_path [@XMRRemoteSingleProbe::@sourceInst]
  // CHECK-LABEL: hw.module @XMRRemoteSingleProbe
  firrtl.module @XMRRemoteSingleProbe() {
    %source_data = firrtl.instance source sym @sourceInst @Source(out data: !firrtl.rwprobe<uint<32>>)
    // CHECK: hw.instance "source" sym @sourceInst @Source

    %xmr = firrtl.xmr.remote @XMRRemoteSingleProbe::@sourceInst, 0 : !firrtl.rwprobe<uint<32>>
    // CHECK: hw.xmr.remote @XMRRemoteSingleProbe_sourceInst_xmr_remote_path, 0 : !hw.rwprobe<i32>
  }
}
