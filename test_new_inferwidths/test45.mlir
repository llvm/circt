firrtl.circuit "RWProbePort" { 
  // CHECK-LABEL: module @RWProbePort
  // CHECK-SAME: rwprobe<uint<1>>
  // CHECK-SAME: rwprobe<uint<2>>
  firrtl.module @RWProbePort(in %in: !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<2>>,
                             out %p: !firrtl.rwprobe<uint>,
                             out %p2: !firrtl.rwprobe<uint>) {
    // CHECK-NEXT: bundle<a: vector<uint<1>, 2>, b: uint<2>>
    // CHECK-SAME: rwprobe<uint<1>>
    // CHECK-SAME: rwprobe<uint<2>>
    %c_in, %c_p, %c_p2 = firrtl.instance c @RWProbePortChild(in in: !firrtl.bundle<a: vector<uint, 2>, b: uint>, out p: !firrtl.rwprobe<uint>, out p2: !firrtl.rwprobe<uint>)
    // CHECK-NEXT: firrtl.connect %c_in, %in : !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<2>>
   firrtl.connect %c_in, %in : !firrtl.bundle<a: vector<uint, 2>, b: uint>, !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<2>>
    // CHECK-NEXT: rwprobe<uint<1>>
    firrtl.ref.define %p, %c_p : !firrtl.rwprobe<uint>
    // CHECK-NEXT: rwprobe<uint<2>>
    firrtl.ref.define %p2, %c_p2 : !firrtl.rwprobe<uint>
  }
  // CHECK-LABEL: module private @RWProbePortChild(
  // CHECK-SAME: %in: !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<2>>
  // CHECK-SAME: %p: !firrtl.rwprobe<uint<1>>
  // CHECK-SAME: %p2: !firrtl.rwprobe<uint<2>>
  // CHECK-NEXT: ref.rwprobe {{.+}} : !firrtl.rwprobe<uint<1>>
  // CHECK-NEXT: ref.rwprobe {{.+}} : !firrtl.rwprobe<uint<2>>
  firrtl.module private @RWProbePortChild(in %in: !firrtl.bundle<a: vector<uint, 2>, b: uint> sym [<@in_a_1,3,public>,<@in_b,4,public>],
                                          out %p: !firrtl.rwprobe<uint>,
                                          out %p2: !firrtl.rwprobe<uint>) {
    %0 = firrtl.ref.rwprobe <@RWProbePortChild::@in_a_1> : !firrtl.rwprobe<uint>
    %1 = firrtl.ref.rwprobe <@RWProbePortChild::@in_b> : !firrtl.rwprobe<uint>
    firrtl.ref.define %p, %0 : !firrtl.rwprobe<uint>
    firrtl.ref.define %p2, %1 : !firrtl.rwprobe<uint>
  }
}