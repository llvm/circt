// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lint))' %s

// This test checks that the linter does _not_ error for certain patterns.

firrtl.circuit "XmrNoDut" {
  hw.hierpath private @xmrPath [@XmrNoDut::@sym]
  firrtl.module @XmrNoDut() {
    %a = firrtl.wire sym @sym : !firrtl.uint<1>
    %0 = firrtl.xmr.deref @xmrPath : !firrtl.uint<1>
    %b = firrtl.node %0 : !firrtl.uint<1>
  }
}

firrtl.circuit "XmrInTestHarness" {
  hw.hierpath private @xmrPath [@XmrInTestHarness::@sym]
  firrtl.module @Dut() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {
  }
  firrtl.module @XmrInTestHarness() {
    firrtl.instance dut @Dut()
    %a = firrtl.wire sym @sym : !firrtl.uint<1>
    %0 = firrtl.xmr.deref @xmrPath : !firrtl.uint<1>
    %b = firrtl.node %0 : !firrtl.uint<1>
  }
}

firrtl.circuit "XmrInLayer" {
  hw.hierpath private @xmrPath [@XmrInLayer::@sym]
  firrtl.layer @A bind {}
  firrtl.module @XmrInLayer() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {
    firrtl.layerblock @A {
      %a = firrtl.wire sym @sym : !firrtl.uint<1>
      %0 = firrtl.xmr.deref @xmrPath : !firrtl.uint<1>
      %b = firrtl.node %0 : !firrtl.uint<1>
    }
  }
}

firrtl.circuit "XmrInLayer_lowered_Bind" {
  hw.hierpath private @xmrPath [@Foo::@sym]
  firrtl.module @Foo() {
    %a = firrtl.wire sym @sym : !firrtl.uint<1>
    %0 = firrtl.xmr.deref @xmrPath : !firrtl.uint<1>
    %b = firrtl.node %0 : !firrtl.uint<1>
  }
  firrtl.module @XmrInLayer_lowered_Bind() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {
    firrtl.instance a sym @a {
      doNotPrint
    } @Foo()
  }
}

firrtl.circuit "XmrInLayer_lowered_IfDef" {
  hw.hierpath private @xmrPath [@XmrInLayer_lowered_IfDef::@sym]
  sv.macro.decl @layer
  firrtl.module @XmrInLayer_lowered_IfDef() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {
    sv.ifdef @layer {
      %a = firrtl.wire sym @sym : !firrtl.uint<1>
      %0 = firrtl.xmr.deref @xmrPath : !firrtl.uint<1>
      %b = firrtl.node %0 : !firrtl.uint<1>
    }
  }
}

firrtl.circuit "XMRInDesignNoUsers" {
  hw.hierpath private @xmrPath [@XMRInDesignNoUsers::@sym]
  firrtl.module @XMRInDesignNoUsers() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {
    %a = firrtl.wire sym @sym : !firrtl.uint<1>
    %0 = firrtl.xmr.deref @xmrPath : !firrtl.uint<1>
  }
}
