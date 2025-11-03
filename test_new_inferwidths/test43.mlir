firrtl.circuit "InferConst" { 
  // CHECK-SAME: out %out: !firrtl.const.bundle<a: uint<1>, b: sint<2>, c: analog<3>, d: vector<uint<4>, 2>>
  firrtl.module @InferConst(in %a: !firrtl.const.uint<1>, in %b: !firrtl.const.sint<2>, in %c: !firrtl.const.analog<3>, in %d: !firrtl.const.vector<uint<4>, 2>,
    out %out: !firrtl.const.bundle<a: uint, b: sint, c: analog, d: vector<uint, 2>>) {
    %0 = firrtl.subfield %out[a] : !firrtl.const.bundle<a: uint, b: sint, c: analog, d: vector<uint, 2>>
    %1 = firrtl.subfield %out[b] : !firrtl.const.bundle<a: uint, b: sint, c: analog, d: vector<uint, 2>>
    %2 = firrtl.subfield %out[c] : !firrtl.const.bundle<a: uint, b: sint, c: analog, d: vector<uint, 2>>
    %3 = firrtl.subfield %out[d] : !firrtl.const.bundle<a: uint, b: sint, c: analog, d: vector<uint, 2>>

    firrtl.connect %0, %a : !firrtl.const.uint, !firrtl.const.uint<1>
    firrtl.connect %1, %b : !firrtl.const.sint, !firrtl.const.sint<2>
    firrtl.attach %2, %c : !firrtl.const.analog, !firrtl.const.analog<3>
    firrtl.connect %3, %d : !firrtl.const.vector<uint, 2>, !firrtl.const.vector<uint<4>, 2>
  }
}