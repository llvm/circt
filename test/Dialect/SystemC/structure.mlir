// RUN: circt-opt %s | FileCheck %s

// CHECK: systemc.module @adder (%summand_a: !systemc.in<i32>, %summand_b: !systemc.in<i32>, %sum: !systemc.out<i32>) {
systemc.module @adder (%summand_a: !systemc.in<i32>, %summand_b: !systemc.in<i32>, %sum: !systemc.out<i32>) {
  //CHECK-NEXT: systemc.ctor {
  systemc.ctor {
    // CHECK-NEXT: systemc.method %addFunc
    systemc.method %addFunc
    // CHECK-NEXT: systemc.thread %addFunc
    systemc.thread %addFunc
  //CHECK-NEXT: }
  }
  // CHECK-NEXT: %addFunc = systemc.func {
  %addFunc = systemc.func {
    // CHECK-NEXT: [[SA:%.+]] = systemc.signal.read %summand_a : !systemc.in<i32>
    %0 = systemc.signal.read %summand_a : !systemc.in<i32>
    // CHECK-NEXT: [[SB:%.+]] = systemc.signal.read %summand_b : !systemc.in<i32>
    %1 = systemc.signal.read %summand_b : !systemc.in<i32>
    // CHECK-NEXT: [[RES:%.*]] = comb.add [[SA]], [[SB]] : i32
    %res = comb.add %0, %1 : i32
    // CHECK-NEXT: systemc.signal.write %sum, [[RES]] : !systemc.out<i32>
    systemc.signal.write %sum, %res : !systemc.out<i32>
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: systemc.cpp.destructor {
  systemc.cpp.destructor {
    // CHECK-NEXT: [[PTR:%.+]] = systemc.cpp.new(%summand_a, %summand_b) : (!systemc.in<i32>, !systemc.in<i32>) -> !emitc.ptr<!emitc.opaque<"someclass">>
    %0 = systemc.cpp.new(%summand_a, %summand_b) : (!systemc.in<i32>, !systemc.in<i32>) -> !emitc.ptr<!emitc.opaque<"someclass">>
    // CHECK-NEXT: systemc.cpp.delete [[PTR]] : !emitc.ptr<!emitc.opaque<"someclass">>
    systemc.cpp.delete %0 : !emitc.ptr<!emitc.opaque<"someclass">>
    // CHECK-NEXT: systemc.cpp.new() : () -> !emitc.ptr<!emitc.opaque<"someclass">>
    %1 = systemc.cpp.new() : () -> !emitc.ptr<!emitc.opaque<"someclass">>
  // CHECK-NEXT: }
  }
// CHECK-NEXT: }
}

// CHECK: systemc.module @mixedPorts (%port0: !systemc.out<i4>, %port1: !systemc.in<i32>, %port2: !systemc.out<i4>, %port3: !systemc.inout<i8>) {
systemc.module @mixedPorts (%port0: !systemc.out<i4>, %port1: !systemc.in<i32>, %port2: !systemc.out<i4>, %port3: !systemc.inout<i8>) {}

// CHECK-LABEL: systemc.module @signals
systemc.module @signals () {
  // CHECK-NEXT: %signal0 = systemc.signal : !systemc.signal<i32>
  %signal0 = systemc.signal : !systemc.signal<i32>
  // CHECK-NEXT: %signal1 = systemc.signal : !systemc.signal<i1>
  %signal1 = systemc.signal : !systemc.signal<i1>

  // CHECK-NEXT: systemc.func
  %funchandle = systemc.func {
    // CHECK-NEXT: [[READ:%.*]] = systemc.signal.read %signal0 : !systemc.signal<i32>
    %0 = systemc.signal.read %signal0 : !systemc.signal<i32>
    // CHECK-NEXT: systemc.signal.write %signal0, [[READ:%.*]] : !systemc.signal<i32>
    systemc.signal.write %signal0, %0 : !systemc.signal<i32>
  }
}

// CHECK-LABEL: systemc.module @readWriteInOutPorts
systemc.module @readWriteInOutPorts (%port0 : !systemc.inout<i32>, %port1 : !systemc.inout<i32>) {
  // CHECK-NEXT: systemc.func
  %funchandle = systemc.func {
    // CHECK-NEXT: [[READ:%.*]] = systemc.signal.read %port0 : !systemc.inout<i32>
    %0 = systemc.signal.read %port0 : !systemc.inout<i32>
    // CHECK-NEXT: systemc.signal.write %port1, [[READ:%.*]] : !systemc.inout<i32>
    systemc.signal.write %port1, %0 : !systemc.inout<i32>
  }
}

// CHECK-LABEL: systemc.module private @moduleVisibility
systemc.module private @moduleVisibility () {}

// CHECK-LABEL: systemc.module @argAttrs (%port0: !systemc.in<i32> {hw.attrname = "sometext"}, %port1: !systemc.in<i32>, %out0: !systemc.out<i32>)
systemc.module @argAttrs (%port0: !systemc.in<i32> {hw.attrname = "sometext"}, %port1: !systemc.in<i32>, %out0: !systemc.out<i32>) {}

// CHECK-LABEL: systemc.module @resultAttrs (%port0: !systemc.in<i32>, %out0: !systemc.out<i32> {hw.attrname = "sometext"})
systemc.module @resultAttrs (%port0: !systemc.in<i32>, %out0: !systemc.out<i32> {hw.attrname = "sometext"}) {}

// CHECK-LABEL: systemc.module @instanceDecl
systemc.module @instanceDecl (%input0: !systemc.in<i32>) {
  // CHECK-NEXT: %moduleInstance0 = systemc.instance.decl @adder : !systemc.module<adder(summand_a: !systemc.in<i32>, summand_b: !systemc.in<i32>, sum: !systemc.out<i32>)>
  %moduleInstance0 = systemc.instance.decl @adder : !systemc.module<adder(summand_a: !systemc.in<i32>, summand_b: !systemc.in<i32>, sum: !systemc.out<i32>)>
  // CHECK-NEXT: %moduleInstance1 = systemc.instance.decl @moduleVisibility : !systemc.module<moduleVisibility()>
  %moduleInstance1 = systemc.instance.decl @moduleVisibility : !systemc.module<moduleVisibility()>
  // CHECK-NEXT: systemc.ctor
  systemc.ctor {
    // CHECK-NEXT: systemc.instance.bind_port %moduleInstance0["summand_a"] to %input0 : !systemc.module<adder(summand_a: !systemc.in<i32>, summand_b: !systemc.in<i32>, sum: !systemc.out<i32>)>, !systemc.in<i32>
    systemc.instance.bind_port %moduleInstance0["summand_a"] to %input0 : !systemc.module<adder(summand_a: !systemc.in<i32>, summand_b: !systemc.in<i32>, sum: !systemc.out<i32>)>, !systemc.in<i32>
  }
}

// CHECK-LABEL: systemc.module @attributes
systemc.module @attributes () {
  // CHECK-NEXT: systemc.ctor attributes {systemc.someattr = 0 : i64} {
  systemc.ctor attributes {systemc.someattr = 0 : i64} {}
  // CHECK-NEXT: }
  // CHECK-NEXT: %func = systemc.func attributes {systemc.someattr = 0 : i64} {
  %func = systemc.func attributes {systemc.someattr = 0 : i64} {}
  // CHECK-NEXT: }
  // CHECK-NEXT: systemc.cpp.destructor attributes {systemc.someattr = 0 : i64} {
  systemc.cpp.destructor attributes {systemc.someattr = 0 : i64} {}
  // CHECK-NEXT: }
}

// CHECK-LABEL: systemc.module @variableAndAssign
systemc.module @variableAndAssign () {
  // CHECK-NEXT: %varname = systemc.cpp.variable : i32
  %varname = systemc.cpp.variable : i32
  // CHECK-NEXT: systemc.ctor {
  systemc.ctor {
    // CHECK-NEXT: %c42_i32 = hw.constant 42 : i32
    %c42_i32 = hw.constant 42 : i32
    // CHECK-NEXT: systemc.cpp.assign %varname = %c42_i32 : i32
    systemc.cpp.assign %varname = %c42_i32 : i32
    // CHECK-NEXT: %varwithinit = systemc.cpp.variable %varname : i32
    %varwithinit = systemc.cpp.variable %varname : i32
    // CHECK-NEXT: systemc.cpp.assign %varwithinit = %varname : i32
    systemc.cpp.assign %varwithinit = %varname : i32
  }
}
