// RUN: circt-opt %s  --firrtl-lower-xmr -split-input-file |  FileCheck %s

// Test for same module lowering
// CHECK-LABEL: firrtl.circuit "xmr"
firrtl.circuit "xmr" {
  // CHECK-LABEL: firrtl.module @xmr(out %o: !firrtl.uint<2>)
  firrtl.module @xmr(out %o: !firrtl.uint<2>, in %2: !firrtl.ref<uint<0>>) {
    %w = firrtl.wire : !firrtl.uint<2>
    %1 = firrtl.ref.send %w : !firrtl.uint<2>
    %x = firrtl.ref.resolve %1 : !firrtl.ref<uint<2>>
    %x2 = firrtl.ref.resolve %2 : !firrtl.ref<uint<0>>
    // CHECK-NOT: firrtl.ref.resolve
    firrtl.strictconnect %o, %x : !firrtl.uint<2>
    // CHECK:      %w = firrtl.wire sym @[[wSym:[a-zA-Z0-9_]+]] : !firrtl.uint<2>
    // CHECK-NEXT: %[[#xmr:]] = sv.xmr.ref #hw.innerNameRef<@xmr::@[[wSym]]> : !hw.inout<i2>
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]] : !hw.inout<i2> to !firrtl.uint<2>
    // CHECK:      firrtl.strictconnect %o, %[[#cast]] : !firrtl.uint<2>
  }
}

// -----

// Test the correct xmr path is generated
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  // CHECK:      hw.hierpath private @[[path:[a-zA-Z0-9_]+]]
  // CHECK-SAME:   [@Top::@bar, @Bar::@barXMR, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  firrtl.module @XmrSrcMod(out %_a: !firrtl.ref<uint<1>>) {
    // CHECK: firrtl.module @XmrSrcMod() {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK:  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK:  %0 = firrtl.node sym @[[xmrSym]] %c0_ui1  : !firrtl.uint<1>
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.strictconnect %_a, %1 : !firrtl.ref<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
    %xmr   = firrtl.instance bar sym @barXMR @XmrSrcMod(out _a: !firrtl.ref<uint<1>>)
    // CHECK:  firrtl.instance bar sym @barXMR  @XmrSrcMod()
    firrtl.strictconnect %_a, %xmr   : !firrtl.ref<uint<1>>
  }
  firrtl.module @Top() {
    %bar_a = firrtl.instance bar sym @bar  @Bar(out _a: !firrtl.ref<uint<1>>)
    // CHECK:  firrtl.instance bar sym @bar  @Bar()
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]] : !hw.inout<i1>
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]] : !hw.inout<i1> to !firrtl.uint<1>
    // CHECK-NEXT; firrtl.strictconnect %a, %[[#cast]] : !firrtl.uint<1>
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  }
}

// -----

// Test 0-width xmrs are handled
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  firrtl.module @Top(in %bar_a : !firrtl.ref<uint<0>>, in %bar_b : !firrtl.ref<vector<uint<0>,10>>) {
    %a = firrtl.wire : !firrtl.uint<0>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.ref<uint<0>>
    // CHECK:  %[[c0_ui0:.+]] = firrtl.constant 0 : !firrtl.uint<0>
    firrtl.strictconnect %a, %0 : !firrtl.uint<0>
    // CHECK:  firrtl.strictconnect %a, %[[c0_ui0]] : !firrtl.uint<0>
    %b = firrtl.wire : !firrtl.vector<uint<0>,10>
    %1 = firrtl.ref.resolve %bar_b : !firrtl.ref<vector<uint<0>,10>>
    firrtl.strictconnect %b, %1 : !firrtl.vector<uint<0>,10>
    // CHECK:	%[[c0_ui0_0:.+]] = firrtl.constant 0 : !firrtl.uint<0>
    // CHECK:  %[[v2:.+]] = firrtl.bitcast %[[c0_ui0_0]] : (!firrtl.uint<0>) -> !firrtl.vector<uint<0>, 10>
    // CHECK:  firrtl.strictconnect %b, %[[v2]] : !firrtl.vector<uint<0>, 10>
  }
}

// -----

// Test the correct xmr path to port is generated
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@bar, @Bar::@barXMR, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  firrtl.module @XmrSrcMod(in %pa: !firrtl.uint<1>, out %_a: !firrtl.ref<uint<1>>) {
    // CHECK: firrtl.module @XmrSrcMod(in %pa: !firrtl.uint<1> sym @[[xmrSym]]) {
    %1 = firrtl.ref.send %pa : !firrtl.uint<1>
    firrtl.strictconnect %_a, %1 : !firrtl.ref<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
    %pa, %xmr   = firrtl.instance bar sym @barXMR @XmrSrcMod(in pa: !firrtl.uint<1>, out _a: !firrtl.ref<uint<1>>)
    // CHECK: %bar_pa = firrtl.instance bar sym @barXMR  @XmrSrcMod(in pa: !firrtl.uint<1>)
    firrtl.strictconnect %_a, %xmr   : !firrtl.ref<uint<1>>
  }
  firrtl.module @Top() {
    %bar_a = firrtl.instance bar sym @bar  @Bar(out _a: !firrtl.ref<uint<1>>)
    // CHECK:  firrtl.instance bar sym @bar  @Bar()
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]] : !hw.inout<i1>
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]] : !hw.inout<i1> to !firrtl.uint<1>
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %a, %[[#cast]]
  }
}

// -----

// Test for multiple readers and multiple instances
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  // CHECK-DAG: hw.hierpath private @[[path_0:[a-zA-Z0-9_]+]] [@Foo::@fooXMR, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  // CHECK-DAG: hw.hierpath private @[[path_1:[a-zA-Z0-9_]+]] [@Bar::@barXMR, @XmrSrcMod::@[[xmrSym]]]
  // CHECK-DAG: hw.hierpath private @[[path_2:[a-zA-Z0-9_]+]] [@Top::@bar, @Bar::@barXMR, @XmrSrcMod::@[[xmrSym]]]
  // CHECK-DAG: hw.hierpath private @[[path_3:[a-zA-Z0-9_]+]] [@Top::@foo, @Foo::@fooXMR, @XmrSrcMod::@[[xmrSym]]]
  // CHECK-DAG: hw.hierpath private @[[path_4:[a-zA-Z0-9_]+]] [@Top::@xmr, @XmrSrcMod::@[[xmrSym]]]
  firrtl.module @XmrSrcMod(out %_a: !firrtl.ref<uint<1>>) {
    // CHECK: firrtl.module @XmrSrcMod() {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK:   %c0_ui1 = firrtl.constant 0
    // CHECK:  %0 = firrtl.node sym @[[xmrSym]] %c0_ui1  : !firrtl.uint<1>
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.strictconnect %_a, %1 : !firrtl.ref<uint<1>>
  }
  firrtl.module @Foo(out %_a: !firrtl.ref<uint<1>>) {
    %xmr   = firrtl.instance bar sym @fooXMR @XmrSrcMod(out _a: !firrtl.ref<uint<1>>)
    // CHECK:  firrtl.instance bar sym @fooXMR  @XmrSrcMod()
    firrtl.strictconnect %_a, %xmr   : !firrtl.ref<uint<1>>
    %0 = firrtl.ref.resolve %xmr   : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path_0]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %a = firrtl.wire : !firrtl.uint<1>
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK:      firrtl.strictconnect %a, %[[#cast]]
  }
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
    %xmr   = firrtl.instance bar sym @barXMR @XmrSrcMod(out _a: !firrtl.ref<uint<1>>)
    // CHECK:  firrtl.instance bar sym @barXMR  @XmrSrcMod()
    firrtl.strictconnect %_a, %xmr   : !firrtl.ref<uint<1>>
    %0 = firrtl.ref.resolve %xmr   : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path_1]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %a = firrtl.wire : !firrtl.uint<1>
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK:      firrtl.strictconnect %a, %[[#cast]]
  }
  firrtl.module @Top() {
    %bar_a = firrtl.instance bar sym @bar  @Bar(out _a: !firrtl.ref<uint<1>>)
    %foo_a = firrtl.instance foo sym @foo @Foo(out _a: !firrtl.ref<uint<1>>)
    %xmr_a = firrtl.instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.ref<uint<1>>)
    // CHECK:  firrtl.instance bar sym @bar  @Bar()
    // CHECK:  firrtl.instance foo sym @foo  @Foo()
    // CHECK:  firrtl.instance xmr sym @xmr  @XmrSrcMod()
    %a = firrtl.wire : !firrtl.uint<1>
    %b = firrtl.wire : !firrtl.uint<1>
    %c = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path_2]]
    // CHECK-NEXT: %[[#cast_2:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = firrtl.ref.resolve %foo_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path_3]]
    // CHECK-NEXT: %[[#cast_3:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %2 = firrtl.ref.resolve %xmr_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path_4]]
    // CHECK-NEXT: %[[#cast_4:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %a, %[[#cast_2]]
    firrtl.strictconnect %b, %1 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %b, %[[#cast_3]]
    firrtl.strictconnect %c, %2 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %c, %[[#cast_4]]
  }
}

// -----

// Check for downward reference
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@bar, @Bar::@barXMR, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  firrtl.module @XmrSrcMod(out %_a: !firrtl.ref<uint<1>>) {
    // CHECK: firrtl.module @XmrSrcMod() {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK:  %c0_ui1 = firrtl.constant 0
    // CHECK:  %0 = firrtl.node sym @[[xmrSym]] %c0_ui1  : !firrtl.uint<1>
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.strictconnect %_a, %1 : !firrtl.ref<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
    %xmr   = firrtl.instance bar sym @barXMR @XmrSrcMod(out _a: !firrtl.ref<uint<1>>)
    // CHECK:  firrtl.instance bar sym @barXMR  @XmrSrcMod()
    firrtl.strictconnect %_a, %xmr   : !firrtl.ref<uint<1>>
  }
  firrtl.module @Top() {
    %bar_a = firrtl.instance bar sym @bar  @Bar(out _a: !firrtl.ref<uint<1>>)
    // CHECK:  firrtl.instance bar sym @bar  @Bar()
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %a, %[[#cast]]
    %c_a = firrtl.instance child @Child(in  _a: !firrtl.ref<uint<1>>)
    firrtl.strictconnect %c_a, %bar_a : !firrtl.ref<uint<1>>
  }
  firrtl.module @Child(in  %_a: !firrtl.ref<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }

}

// -----

// Check for downward reference to port
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@bar, @Bar::@barXMR, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  firrtl.module @XmrSrcMod(in %pa: !firrtl.uint<1>, out %_a: !firrtl.ref<uint<1>>) {
    // CHECK: firrtl.module @XmrSrcMod(in %pa: !firrtl.uint<1> sym @xmr_sym) {
    %1 = firrtl.ref.send %pa : !firrtl.uint<1>
    firrtl.strictconnect %_a, %1 : !firrtl.ref<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
    %pa, %xmr   = firrtl.instance bar sym @barXMR @XmrSrcMod(in pa: !firrtl.uint<1>, out _a: !firrtl.ref<uint<1>>)
    // CHECK: %bar_pa = firrtl.instance bar sym @barXMR  @XmrSrcMod(in pa: !firrtl.uint<1>)
    firrtl.strictconnect %_a, %xmr   : !firrtl.ref<uint<1>>
  }
  firrtl.module @Top() {
    %bar_a = firrtl.instance bar sym @bar  @Bar(out _a: !firrtl.ref<uint<1>>)
    // CHECK:  firrtl.instance bar sym @bar  @Bar()
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %a, %[[#cast]]
    %c_a = firrtl.instance child @Child(in  _a: !firrtl.ref<uint<1>>)
    firrtl.strictconnect %c_a, %bar_a : !firrtl.ref<uint<1>>
  }
  firrtl.module @Child(in  %_a: !firrtl.ref<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }
}

// -----

// Test for multiple paths and downward reference.
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path_0:[a-zA-Z0-9_]+]] [@Top::@foo, @Foo::@fooXMR, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  // CHECK: hw.hierpath private @[[path_1:[a-zA-Z0-9_]+]] [@Top::@xmr, @XmrSrcMod::@[[xmrSym]]]
  firrtl.module @XmrSrcMod(out %_a: !firrtl.ref<uint<1>>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.strictconnect %_a, %1 : !firrtl.ref<uint<1>>
    // CHECK: firrtl.node sym @[[xmrSym]]
  }
  firrtl.module @Foo(out %_a: !firrtl.ref<uint<1>>) {
    %xmr   = firrtl.instance bar sym @fooXMR @XmrSrcMod(out _a: !firrtl.ref<uint<1>>)
    firrtl.strictconnect %_a, %xmr   : !firrtl.ref<uint<1>>
  }
  firrtl.module @Top() {
    %foo_a = firrtl.instance foo sym @foo @Foo(out _a: !firrtl.ref<uint<1>>)
    %xmr_a = firrtl.instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.ref<uint<1>>)
    %c_a, %c_b = firrtl.instance child @Child2p(in _a: !firrtl.ref<uint<1>>, in _b: !firrtl.ref<uint<1>> )
    // CHECK:  firrtl.instance child  @Child2p()
    firrtl.strictconnect %c_a, %foo_a : !firrtl.ref<uint<1>>
    firrtl.strictconnect %c_b, %xmr_a : !firrtl.ref<uint<1>>
  }
  firrtl.module @Child2p(in  %_a: !firrtl.ref<uint<1>>, in  %_b: !firrtl.ref<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path_0]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = firrtl.ref.resolve %_b : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path_1]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }
}

// -----

// Test for multiple children paths
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@xmr, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  firrtl.module @XmrSrcMod(out %_a: !firrtl.ref<uint<1>>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.strictconnect %_a, %1 : !firrtl.ref<uint<1>>
    // CHECK: firrtl.node sym @[[xmrSym]]
  }
  firrtl.module @Top() {
    %xmr_a = firrtl.instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.ref<uint<1>>)
    %c_a = firrtl.instance child @Child1(in _a: !firrtl.ref<uint<1>>)
    firrtl.strictconnect %c_a, %xmr_a : !firrtl.ref<uint<1>>
  }
  // CHECK-LABEL: firrtl.module @Child1() {
  firrtl.module @Child1(in  %_a: !firrtl.ref<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %c_a, %c_b = firrtl.instance child @Child2(in _a: !firrtl.ref<uint<1>>, in _b: !firrtl.ref<uint<1>> )
    firrtl.strictconnect %c_a, %_a : !firrtl.ref<uint<1>>
    firrtl.strictconnect %c_b, %_a : !firrtl.ref<uint<1>>
    %c3 = firrtl.instance child @Child3(in _a: !firrtl.ref<uint<1>>)
    firrtl.strictconnect %c3 , %_a : !firrtl.ref<uint<1>>
  }
  firrtl.module @Child2(in  %_a: !firrtl.ref<uint<1>>, in  %_b: !firrtl.ref<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = firrtl.ref.resolve %_b : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }
  firrtl.module @Child3(in  %_a: !firrtl.ref<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = firrtl.ref.resolve %_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }
}

// -----

// Test for multiple children paths
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@xmr, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  firrtl.module @XmrSrcMod(out %_a: !firrtl.ref<uint<1>>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.strictconnect %_a, %1 : !firrtl.ref<uint<1>>
    // CHECK: firrtl.node sym @[[xmrSym]]
  }
  firrtl.module @Top() {
    %xmr_a = firrtl.instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.ref<uint<1>>)
    %c_a = firrtl.instance child @Child1(in _a: !firrtl.ref<uint<1>>)
    firrtl.strictconnect %c_a, %xmr_a : !firrtl.ref<uint<1>>
  }
  // CHECK-LABEL: firrtl.module @Child1() {
  firrtl.module @Child1(in  %_a: !firrtl.ref<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %c_a, %c_b = firrtl.instance child @Child2(in _a: !firrtl.ref<uint<1>>, in _b: !firrtl.ref<uint<1>> )
    firrtl.strictconnect %c_a, %_a : !firrtl.ref<uint<1>>
    firrtl.strictconnect %c_b, %_a : !firrtl.ref<uint<1>>
    %c3 = firrtl.instance child @Child3(in _a: !firrtl.ref<uint<1>>)
    firrtl.strictconnect %c3 , %_a : !firrtl.ref<uint<1>>
  }
  firrtl.module @Child2(in  %_a: !firrtl.ref<uint<1>>, in  %_b: !firrtl.ref<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = firrtl.ref.resolve %_b : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }
  firrtl.module @Child3(in  %_a: !firrtl.ref<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = firrtl.ref.resolve %_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }
}

// -----

// Multiply instantiated Top works, because the reference port does not flow through it.
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Dut::@xmr, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  firrtl.module @XmrSrcMod(out %_a: !firrtl.ref<uint<1>>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.strictconnect %_a, %1 : !firrtl.ref<uint<1>>
    // CHECK: firrtl.node sym @[[xmrSym]]
  }
  firrtl.module @Top() {
    firrtl.instance d1 @Dut()
  }
  firrtl.module @Top2() {
    firrtl.instance d2 @Dut()
  }
  firrtl.module @Dut() {
    %xmr_a = firrtl.instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.ref<uint<1>>)
    %c_a = firrtl.instance child @Child1(in _a: !firrtl.ref<uint<1>>)
    firrtl.strictconnect %c_a, %xmr_a : !firrtl.ref<uint<1>>
  }
  // CHECK-LABEL: firrtl.module @Child1() {
  firrtl.module @Child1(in  %_a: !firrtl.ref<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %c_a, %c_b = firrtl.instance child @Child2(in _a: !firrtl.ref<uint<1>>, in _b: !firrtl.ref<uint<1>> )
    firrtl.strictconnect %c_a, %_a : !firrtl.ref<uint<1>>
    firrtl.strictconnect %c_b, %_a : !firrtl.ref<uint<1>>
    %c3 = firrtl.instance child @Child3(in _a: !firrtl.ref<uint<1>>)
    firrtl.strictconnect %c3 , %_a : !firrtl.ref<uint<1>>
  }
  firrtl.module @Child2(in  %_a: !firrtl.ref<uint<1>>, in  %_b: !firrtl.ref<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = firrtl.ref.resolve %_b : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }
  firrtl.module @Child3(in  %_a: !firrtl.ref<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_
    %1 = firrtl.ref.resolve %_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_
  }
}

// -----

firrtl.circuit "Top"  {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@xmr_sym, @DUTModule::@[[xmrSym:[a-zA-Z0-9_]+]]]
  // CHECK-LABEL: firrtl.module private @DUTModule
  // CHECK-SAME: (in %clock: !firrtl.clock, in %io_addr: !firrtl.uint<3>, in %io_dataIn: !firrtl.uint<8>, in %io_wen: !firrtl.uint<1>, out %io_dataOut: !firrtl.uint<8>)
  firrtl.module private @DUTModule(in %clock: !firrtl.clock, in %io_addr: !firrtl.uint<3>, in %io_dataIn: !firrtl.uint<8>, in %io_wen: !firrtl.uint<1>, out %io_dataOut: !firrtl.uint<8>, out %_gen_memTap: !firrtl.ref<vector<uint<8>, 8>>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %rf_memTap, %rf_read, %rf_write = firrtl.mem  Undefined  {depth = 8 : i64, groupID = 1 : ui32, name = "rf", portNames = ["memTap", "read", "write"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.ref<vector<uint<8>, 8>>, !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    // CHECK:  %rf_read, %rf_write = firrtl.mem sym @xmr_sym  Undefined  {depth = 8 : i64, groupID = 1 : ui32, name = "rf", portNames = ["read", "write"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %0 = firrtl.subfield %rf_read[addr] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    %1 = firrtl.subfield %rf_read[en] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    %2 = firrtl.subfield %rf_read[clk] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    %3 = firrtl.subfield %rf_read[data] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    %4 = firrtl.subfield %rf_write[addr] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %5 = firrtl.subfield %rf_write[en] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %6 = firrtl.subfield %rf_write[clk] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %7 = firrtl.subfield %rf_write[data] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %8 = firrtl.subfield %rf_write[mask] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    firrtl.strictconnect %0, %io_addr : !firrtl.uint<3>
    firrtl.strictconnect %1, %c1_ui1 : !firrtl.uint<1>
    firrtl.strictconnect %2, %clock : !firrtl.clock
    firrtl.strictconnect %io_dataOut, %3 : !firrtl.uint<8>
    firrtl.strictconnect %4, %io_addr : !firrtl.uint<3>
    firrtl.strictconnect %5, %io_wen : !firrtl.uint<1>
    firrtl.strictconnect %6, %clock : !firrtl.clock
    firrtl.strictconnect %8, %c1_ui1 : !firrtl.uint<1>
    firrtl.strictconnect %7, %io_dataIn : !firrtl.uint<8>
    firrtl.connect %_gen_memTap, %rf_memTap : !firrtl.ref<vector<uint<8>, 8>>, !firrtl.ref<vector<uint<8>, 8>>
  }
  // CHECK: firrtl.module @Top
  firrtl.module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io_addr: !firrtl.uint<3>, in %io_dataIn: !firrtl.uint<8>, in %io_wen: !firrtl.uint<1>, out %io_dataOut: !firrtl.uint<8>) {
    %dut_clock, %dut_io_addr, %dut_io_dataIn, %dut_io_wen, %dut_io_dataOut, %dut__gen_memTap = firrtl.instance dut  @DUTModule(in clock: !firrtl.clock, in io_addr: !firrtl.uint<3>, in io_dataIn: !firrtl.uint<8>, in io_wen: !firrtl.uint<1>, out io_dataOut: !firrtl.uint<8>, out _gen_memTap: !firrtl.ref<vector<uint<8>, 8>>)
    %0 = firrtl.ref.resolve %dut__gen_memTap : !firrtl.ref<vector<uint<8>, 8>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]] ".Memory"
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    firrtl.strictconnect %dut_clock, %clock : !firrtl.clock
    %memTap_0 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_1 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_2 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_3 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_4 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_5 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_6 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_7 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    firrtl.strictconnect %io_dataOut, %dut_io_dataOut : !firrtl.uint<8>
    firrtl.strictconnect %dut_io_wen, %io_wen : !firrtl.uint<1>
    firrtl.strictconnect %dut_io_dataIn, %io_dataIn : !firrtl.uint<8>
    firrtl.strictconnect %dut_io_addr, %io_addr : !firrtl.uint<3>
    %1 = firrtl.subindex %0[0] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_0:]] = firrtl.subindex %[[#cast]][0]
    firrtl.strictconnect %memTap_0, %1 : !firrtl.uint<8>
    // CHECK:  firrtl.strictconnect %memTap_0, %[[#cast_0]] : !firrtl.uint<8>
    %2 = firrtl.subindex %0[1] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_1:]] = firrtl.subindex %[[#cast]][1]
    firrtl.strictconnect %memTap_1, %2 : !firrtl.uint<8>
    // CHECK:  firrtl.strictconnect %memTap_1, %[[#cast_1]] : !firrtl.uint<8>
    %3 = firrtl.subindex %0[2] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_2:]] = firrtl.subindex %[[#cast]][2]
    firrtl.strictconnect %memTap_2, %3 : !firrtl.uint<8>
    // CHECK:  firrtl.strictconnect %memTap_2, %[[#cast_2]] : !firrtl.uint<8>
    %4 = firrtl.subindex %0[3] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_3:]] = firrtl.subindex %[[#cast]][3]
    firrtl.strictconnect %memTap_3, %4 : !firrtl.uint<8>
    // CHECK:  firrtl.strictconnect %memTap_3, %[[#cast_3]] : !firrtl.uint<8>
    %5 = firrtl.subindex %0[4] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_4:]] = firrtl.subindex %[[#cast]][4]
    firrtl.strictconnect %memTap_4, %5 : !firrtl.uint<8>
    // CHECK:  firrtl.strictconnect %memTap_4, %[[#cast_4]] : !firrtl.uint<8>
    %6 = firrtl.subindex %0[5] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_5:]] = firrtl.subindex %[[#cast]][5]
    firrtl.strictconnect %memTap_5, %6 : !firrtl.uint<8>
    // CHECK:  firrtl.strictconnect %memTap_5, %[[#cast_5]] : !firrtl.uint<8>
    %7 = firrtl.subindex %0[6] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_6:]] = firrtl.subindex %[[#cast]][6]
    firrtl.strictconnect %memTap_6, %7 : !firrtl.uint<8>
    // CHECK:  firrtl.strictconnect %memTap_6, %[[#cast_6]] : !firrtl.uint<8>
    %8 = firrtl.subindex %0[7] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_7:]] = firrtl.subindex %[[#cast]][7]
    firrtl.strictconnect %memTap_7, %8 : !firrtl.uint<8>
    // CHECK:  firrtl.strictconnect %memTap_7, %[[#cast_7]] : !firrtl.uint<8>
    }
}

// -----

firrtl.circuit "Top"  {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@xmr_sym, @DUTModule::@[[xmrSym:[a-zA-Z0-9_]+]]]
  // CHECK-LABEL:  firrtl.module private @DUTModule
  // CHECK-SAME: in %io_wen: !firrtl.uint<1>, out %io_dataOut: !firrtl.uint<8>)
  firrtl.module private @DUTModule(in %clock: !firrtl.clock, in %io_addr: !firrtl.uint<3>, in %io_dataIn: !firrtl.uint<8>, in %io_wen: !firrtl.uint<1>, out %io_dataOut: !firrtl.uint<8>, out %_gen_memTap_0: !firrtl.ref<uint<8>>, out %_gen_memTap_1: !firrtl.ref<uint<8>>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %rf_memTap, %rf_read, %rf_write = firrtl.mem  Undefined  {depth = 2 : i64, groupID = 1 : ui32, name = "rf", portNames = ["memTap", "read", "write"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.ref<vector<uint<8>, 2>>, !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    // CHECK:  %rf_read, %rf_write = firrtl.mem sym @xmr_sym  Undefined  {depth = 2 : i64, groupID = 1 : ui32, name = "rf", portNames = ["read", "write"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %9 = firrtl.ref.sub %rf_memTap[0] : !firrtl.ref<vector<uint<8>, 2>>
    firrtl.strictconnect %_gen_memTap_0, %9 : !firrtl.ref<uint<8>>
    %10 = firrtl.ref.sub %rf_memTap[1] : !firrtl.ref<vector<uint<8>, 2>>
    firrtl.strictconnect %_gen_memTap_1, %10 : !firrtl.ref<uint<8>>
  }
  firrtl.module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io_addr: !firrtl.uint<3>, in %io_dataIn: !firrtl.uint<8>, in %io_wen: !firrtl.uint<1>, out %io_dataOut: !firrtl.uint<8>) {
    %dut_clock, %dut_io_addr, %dut_io_dataIn, %dut_io_wen, %dut_io_dataOut, %dut__gen_memTap_0, %dut__gen_memTap_1 = firrtl.instance dut  @DUTModule(in clock: !firrtl.clock, in io_addr: !firrtl.uint<3>, in io_dataIn: !firrtl.uint<8>, in io_wen: !firrtl.uint<1>, out io_dataOut: !firrtl.uint<8>, out _gen_memTap_0: !firrtl.ref<uint<8>>, out _gen_memTap_1: !firrtl.ref<uint<8>>)
    %0 = firrtl.ref.resolve %dut__gen_memTap_0 : !firrtl.ref<uint<8>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]] ".Memory[0]"
    // CHECK-NEXT: %[[#cast_0:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = firrtl.ref.resolve %dut__gen_memTap_1 : !firrtl.ref<uint<8>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]] ".Memory[1]"
    // CHECK-NEXT: %[[#cast_1:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    firrtl.strictconnect %dut_clock, %clock : !firrtl.clock
    %memTap_0 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_1 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    firrtl.strictconnect %memTap_0, %0 : !firrtl.uint<8>
    // CHECK:      firrtl.strictconnect %memTap_0, %[[#cast_0]]
    firrtl.strictconnect %memTap_1, %1 : !firrtl.uint<8>
    // CHECK:      firrtl.strictconnect %memTap_1, %[[#cast_1]]
  }
}

// -----

// Test lowering of internal path into a module
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@bar, @Bar::@[[xmrSym:[a-zA-Z0-9_]+]]]
  firrtl.module @XmrSrcMod(out %_a: !firrtl.ref<uint<1>>) {
    // CHECK: firrtl.module @XmrSrcMod() {
    // CHECK-NEXT: }
    %z = firrtl.verbatim.expr "internal.path" : () -> !firrtl.uint<1>
    %1 = firrtl.ref.send %z : !firrtl.uint<1>
    firrtl.strictconnect %_a, %1 : !firrtl.ref<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
    %xmr   = firrtl.instance bar sym @barXMR @XmrSrcMod(out _a: !firrtl.ref<uint<1>>)
    // CHECK:  firrtl.instance bar sym @barXMR  @XmrSrcMod()
    firrtl.strictconnect %_a, %xmr   : !firrtl.ref<uint<1>>
  }
  firrtl.module @Top() {
    %bar_a = firrtl.instance bar sym @bar  @Bar(out _a: !firrtl.ref<uint<1>>)
    // CHECK:  firrtl.instance bar sym @bar  @Bar()
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]] ".internal.path"
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %a, %[[#cast]]
  }
}

// -----

// Test lowering of internal path into a module
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@bar, @Bar::@barXMR, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  firrtl.module @XmrSrcMod(out %_a: !firrtl.ref<uint<1>>) {
    // CHECK: firrtl.module @XmrSrcMod() {
    // CHECK{LITERAL}:  firrtl.verbatim.expr "internal.path" : () -> !firrtl.uint<1> {symbols = [@XmrSrcMod]}
    // CHECK:  = firrtl.node sym @xmr_sym  %[[internal:.+]]  : !firrtl.uint<1>
    %z = firrtl.verbatim.expr "internal.path" : () -> !firrtl.uint<1> {symbols = [@XmrSrcMod]}
    %1 = firrtl.ref.send %z : !firrtl.uint<1>
    firrtl.strictconnect %_a, %1 : !firrtl.ref<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
    %xmr   = firrtl.instance bar sym @barXMR @XmrSrcMod(out _a: !firrtl.ref<uint<1>>)
    // CHECK:  firrtl.instance bar sym @barXMR  @XmrSrcMod()
    firrtl.strictconnect %_a, %xmr   : !firrtl.ref<uint<1>>
  }
  firrtl.module @Top() {
    %bar_a = firrtl.instance bar sym @bar  @Bar(out _a: !firrtl.ref<uint<1>>)
    // CHECK:  firrtl.instance bar sym @bar  @Bar()
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.ref<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %a, %[[#cast]]
  }
}

// -----

// Test correct lowering of 0-width ports
firrtl.circuit "Top"  {
  firrtl.module @XmrSrcMod(in %pa: !firrtl.uint<0>, out %_a: !firrtl.ref<uint<0>>) {
  // CHECK-LABEL: firrtl.module @XmrSrcMod(in %pa: !firrtl.uint<0>)
    %0 = firrtl.ref.send %pa : !firrtl.uint<0>
    firrtl.strictconnect %_a, %0 : !firrtl.ref<uint<0>>
  }
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<0>>) {
    %bar_pa, %bar__a = firrtl.instance bar sym @barXMR  @XmrSrcMod(in pa: !firrtl.uint<0>, out _a: !firrtl.ref<uint<0>>)
    firrtl.strictconnect %_a, %bar__a : !firrtl.ref<uint<0>>
  }
  firrtl.module @Top() {
    %bar__a = firrtl.instance bar sym @bar  @Bar(out _a: !firrtl.ref<uint<0>>)
    %a = firrtl.wire   : !firrtl.uint<0>
    %0 = firrtl.ref.resolve %bar__a : !firrtl.ref<uint<0>>
    firrtl.strictconnect %a, %0 : !firrtl.uint<0>
    // CHECK: %c0_ui0 = firrtl.constant 0 : !firrtl.uint<0>
    // CHECK: firrtl.strictconnect %a, %c0_ui0 : !firrtl.uint<0>
  }
}
