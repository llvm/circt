// RUN: circt-opt --arc-resolve-xmr %s --verify-diagnostics --split-input-file

// -----

module {
  hw.hierpath @p [@Top::@mid, @Mid::@leaf, @Leaf::@sig]

  hw.module @Leaf() {
    %c = hw.constant 1 : i1
    %w = hw.wire %c sym @sig : i1
    hw.output
  }

  hw.module @Mid() {
    hw.instance "leaf" sym @leaf @Leaf() -> ()
    hw.output
  }

  hw.module @Top() {
    hw.instance "mid" sym @mid @Mid() -> ()
    // expected-error @below {{unsupported sv.xmr.ref use by 'sv.assign'; only read-only uses (sv.read_inout) are supported}}
    %x = sv.xmr.ref @p : !hw.inout<i1>
    %t = hw.constant true
    sv.assign %x, %t : i1
    hw.output
  }
}

// -----

module {
  hw.hierpath @p [@Top::@mid, @Mid::@leaf, @Leaf::@sig]

  hw.module @Leaf() {
    %c = hw.constant 0 : i1
    %w = hw.wire %c sym @sig : i1
    hw.output
  }

  hw.module @Mid() {
    hw.instance "leaf" sym @leaf @Leaf() -> ()
    hw.output
  }

  hw.module @Top() {
    hw.instance "mid" sym @mid @Mid() -> ()
    // expected-error @below {{sv.xmr.ref has no read uses; write/other uses are not supported yet}}
    %x = sv.xmr.ref @p : !hw.inout<i1>
    hw.output
  }
}

// -----

module {
  hw.hierpath @bbInternal [@Top::@bb, @BlackBox::@hidden]

  hw.module.extern @BlackBox(out out_clk : i1 {hw.exportPort = #hw<innerSym@out_clk>})

  hw.module @Top(out o : i1) {
    hw.instance "bb" sym @bb @BlackBox() -> (out_clk: i1)
    // expected-error @below {{unable to resolve XMR into internal blackbox symbol; rerun with --arc-resolve-xmr=lower-blackbox-internal-to-zero to force zero-lowering}}
    %x = sv.xmr.ref @bbInternal : !hw.inout<i1>
    %r = sv.read_inout %x : !hw.inout<i1>
    hw.output %r : i1
  }
}

// -----

module {
  sv.macro.decl @COND
  hw.hierpath @bindPath [@Host::@src]

  hw.module @Payload(out o : i8) {
    // expected-error @below {{cannot resolve XMR through conditionally guarded sv.bind; only unconditional sv.bind is supported}}
    %x = sv.xmr.ref @bindPath : !hw.inout<i8>
    %r = sv.read_inout %x : !hw.inout<i8>
    hw.output %r : i8
  }

  hw.module @Host(in %src_in : i8 {hw.exportPort = #hw<innerSym@src>}, out out : i8) {
    %src = hw.constant 7 : i8
    hw.instance "payload" sym @payload @Payload() -> (o: i8) {doNotPrint}
    hw.output %src : i8
  }

  sv.ifdef @COND {
    sv.bind <@Host::@payload>
  }
}

// -----

module {
  hw.hierpath @bindPath [@Host::@src]

  hw.module @Payload(out o : i8) {
    // expected-error @below {{payload module has non-bind instances; refusing to resolve bind-context XMR ambiguously}}
    %x = sv.xmr.ref @bindPath : !hw.inout<i8>
    %r = sv.read_inout %x : !hw.inout<i8>
    hw.output %r : i8
  }

  hw.module @Host(in %src_in : i8 {hw.exportPort = #hw<innerSym@src>}, out out : i8) {
    hw.instance "payload" sym @payload @Payload() -> (o: i8) {doNotPrint}
    hw.output %src_in : i8
  }

  hw.module @Top() {
    %src = hw.constant 9 : i8
    hw.instance "host" @Host(src_in: %src: i8) -> (out: i8)
    hw.instance "other" @Other() -> (o: i8)
    hw.output
  }

  hw.module @Other(out o : i8) {
    %p = hw.instance "payload2" sym @payload2 @Payload() -> (o: i8)
    hw.output %p : i8
  }

  sv.bind <@Host::@payload>
}
