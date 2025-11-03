firrtl.circuit "Ref" {
  // CHECK-LABEL: @SubRef
  // CHECK-SAME: out %x: !firrtl.probe<uint<2>>
  // CHECK-SAME: out %y: !firrtl.rwprobe<uint<2>>
  // CHECK-SAME: out %bov_ref: !firrtl.rwprobe<bundle<a: vector<uint<2>, 2>, b: uint<2>>>
  firrtl.module private @SubRef(out %x: !firrtl.probe<uint>, out %y : !firrtl.rwprobe<uint>, out %bov_ref : !firrtl.rwprobe<bundle<a: vector<uint, 2>, b : uint>>) {
    // CHECK: firrtl.wire forceable : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
    %w, %w_rw = firrtl.wire forceable : !firrtl.uint, !firrtl.rwprobe<uint>
    %bov, %bov_rw = firrtl.wire forceable : !firrtl.bundle<a: vector<uint, 2>, b flip: uint>, !firrtl.rwprobe<bundle<a: vector<uint, 2>, b : uint>>
    firrtl.ref.define %bov_ref, %bov_rw : !firrtl.rwprobe<bundle<a: vector<uint, 2>, b : uint>>

    %ref_w = firrtl.ref.send %w : !firrtl.uint
    %cast_ref_w = firrtl.ref.cast %ref_w : (!firrtl.probe<uint>) -> !firrtl.probe<uint>
    firrtl.ref.define %x, %cast_ref_w : !firrtl.probe<uint>
    firrtl.ref.define %y, %w_rw : !firrtl.rwprobe<uint>
    // CHECK: firrtl.ref.cast %w_ref : (!firrtl.rwprobe<uint<2>>) -> !firrtl.probe<uint<2>>
    %cast_w_ro = firrtl.ref.cast %w_rw : (!firrtl.rwprobe<uint>) -> !firrtl.probe<uint>

    %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
    firrtl.connect %w, %c0_ui2 : !firrtl.uint, !firrtl.uint<2>
    
    %bov_a = firrtl.subfield %bov[a] : !firrtl.bundle<a: vector<uint, 2>, b flip: uint>
    %bov_a_1 = firrtl.subindex %bov_a[1] : !firrtl.vector<uint, 2>
    %bov_b = firrtl.subfield %bov[b] : !firrtl.bundle<a: vector<uint, 2>, b flip: uint>

    firrtl.connect %w, %c0_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %bov_a_1, %c0_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %bov_b, %c0_ui2 : !firrtl.uint, !firrtl.uint<2>
  }
  // CHECK-LABEL: @Ref
  // CHECK: out x: !firrtl.probe<uint<2>>
  // CHECK-SAME: out y: !firrtl.rwprobe<uint<2>>
  // CHECK: firrtl.ref.resolve %sub_x : !firrtl.probe<uint<2>>
  // CHECK: firrtl.ref.resolve %sub_y : !firrtl.rwprobe<uint<2>>
  firrtl.module @Ref(out %r : !firrtl.uint, out %s : !firrtl.uint) {
    %sub_x, %sub_y, %sub_bov_ref = firrtl.instance sub @SubRef(out x: !firrtl.probe<uint>, out y: !firrtl.rwprobe<uint>, out bov_ref : !firrtl.rwprobe<bundle<a: vector<uint, 2>, b : uint>>)
    %res_x = firrtl.ref.resolve %sub_x : !firrtl.probe<uint>
    %res_y = firrtl.ref.resolve %sub_y : !firrtl.rwprobe<uint>
    firrtl.connect %r, %res_x : !firrtl.uint, !firrtl.uint
    firrtl.connect %s, %res_y : !firrtl.uint, !firrtl.uint

    // CHECK: !firrtl.rwprobe<bundle<a: vector<uint<2>, 2>, b: uint<2>>>
    %read_bov = firrtl.ref.resolve %sub_bov_ref : !firrtl.rwprobe<bundle<a: vector<uint, 2>, b : uint>>
    // CHECK: !firrtl.rwprobe<bundle<a: vector<uint<2>, 2>, b: uint<2>>>
    %bov_ref_a = firrtl.ref.sub %sub_bov_ref[0] : !firrtl.rwprobe<bundle<a: vector<uint, 2>, b : uint>>
    // CHECK: !firrtl.rwprobe<vector<uint<2>, 2>>
    %bov_ref_a_1 = firrtl.ref.sub %bov_ref_a[1] : !firrtl.rwprobe<vector<uint, 2>>
    // CHECK: !firrtl.rwprobe<bundle<a: vector<uint<2>, 2>, b: uint<2>>>
    %bov_ref_b  = firrtl.ref.sub %sub_bov_ref[1] : !firrtl.rwprobe<bundle<a: vector<uint, 2>, b : uint>>

    // CHECK: !firrtl.rwprobe<vector<uint<2>, 2>>
    %bov_a = firrtl.ref.resolve %bov_ref_a : !firrtl.rwprobe<vector<uint,2>>
    // CHECK: !firrtl.rwprobe<uint<2>>
    %bov_a_1 = firrtl.ref.resolve %bov_ref_a_1 : !firrtl.rwprobe<uint>
    // CHECK: !firrtl.rwprobe<uint<2>>
    %bov_b = firrtl.ref.resolve %bov_ref_b : !firrtl.rwprobe<uint>
  }
}