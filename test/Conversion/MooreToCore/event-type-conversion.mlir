// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @EventVariable
func.func @EventVariable() {
  // CHECK: [[FALSE:%.+]] = hw.constant false
  // CHECK: llhd.sig [[FALSE]] : i1
  %event = moore.variable : <!moore.event>
  return
}

// CHECK-LABEL: func.func @EventBoolCast
// CHECK-SAME: ([[ARG:%.+]]: i1) -> i1
func.func @EventBoolCast(%arg0: !moore.event) -> !moore.i1 {
  // CHECK: [[FALSE:%.+]] = hw.constant false
  // CHECK: [[RESULT:%.+]] = comb.icmp ne [[ARG]], [[FALSE]] : i1
  // CHECK: return [[RESULT]] : i1
  %0 = moore.bool_cast %arg0 : !moore.event -> !moore.i1
  return %0 : !moore.i1
}

// CHECK-LABEL: func.func @EventRefSignature
// CHECK-SAME: ([[ARG:%.+]]: !llhd.ref<i1>) -> !llhd.ref<i1>
func.func @EventRefSignature(%event: !moore.ref<event>) -> !moore.ref<event> {
  // CHECK: return [[ARG]] : !llhd.ref<i1>
  return %event : !moore.ref<event>
}

// CHECK-LABEL: func.func @EventConditional
// CHECK-SAME: ([[COND:%.+]]: i1, [[LHS:%.+]]: i1, [[RHS:%.+]]: i1)
func.func @EventConditional(%cond: !moore.i1, %lhs: !moore.event, %rhs: !moore.event) {
  // CHECK: comb.mux [[COND]], [[LHS]], [[RHS]] : i1
  %0 = moore.conditional %cond : i1 -> event {
    moore.yield %lhs : event
  } {
    moore.yield %rhs : event
  }
  return
}

// CHECK-LABEL: func.func @UArrayEventSignature
// CHECK-SAME: ([[ARG:%.+]]: !hw.array<2xi1>) -> !hw.array<2xi1>
func.func @UArrayEventSignature(%arg0: !moore.uarray<2 x event>) -> !moore.uarray<2 x event> {
  // CHECK: return [[ARG]] : !hw.array<2xi1>
  return %arg0 : !moore.uarray<2 x event>
}

// CHECK-LABEL: func.func @UStructEventSignature
// CHECK-SAME: ([[ARG:%.+]]: !hw.struct<e: i1, bits: i2>) -> !hw.struct<e: i1, bits: i2>
func.func @UStructEventSignature(%arg0: !moore.ustruct<{e: event, bits: i2}>) -> !moore.ustruct<{e: event, bits: i2}> {
  // CHECK: return [[ARG]] : !hw.struct<e: i1, bits: i2>
  return %arg0 : !moore.ustruct<{e: event, bits: i2}>
}

// CHECK-LABEL: func.func @UUnionEventSignature
// CHECK-SAME: ([[ARG:%.+]]: !hw.union<e: i1, bits: i1>) -> !hw.union<e: i1, bits: i1>
func.func @UUnionEventSignature(%arg0: !moore.uunion<{e: event, bits: i1}>) -> !moore.uunion<{e: event, bits: i1}> {
  // CHECK: return [[ARG]] : !hw.union<e: i1, bits: i1>
  return %arg0 : !moore.uunion<{e: event, bits: i1}>
}

moore.class.classdecl @EventClass {
  moore.class.propertydecl @e : !moore.event
  moore.class.propertydecl @bits : !moore.i8
}

// CHECK-LABEL: func.func @EventClassNew() -> !llvm.ptr
func.func @EventClassNew() -> !moore.class<@EventClass> {
  // CHECK: call @malloc
  // CHECK: return {{%.+}} : !llvm.ptr
  %h = moore.class.new : <@EventClass>
  return %h : !moore.class<@EventClass>
}

// CHECK-LABEL: func.func @EventClassProperty
// CHECK-SAME: ([[ARG:%.+]]: !llvm.ptr) -> !llhd.ref<i1>
func.func @EventClassProperty(%h: !moore.class<@EventClass>) -> !moore.ref<event> {
  // CHECK: llvm.getelementptr [[ARG]]
  // CHECK: builtin.unrealized_conversion_cast {{%.+}} : !llvm.ptr to !llhd.ref<i1>
  %ref = moore.class.property_ref %h[@e] : <@EventClass> -> !moore.ref<event>
  return %ref : !moore.ref<event>
}

// CHECK-LABEL: func.func @EventUnionConditional
// CHECK-SAME: ([[COND:%.+]]: i1, [[LHS:%.+]]: !hw.union<e: i1, bits: i1>, [[RHS:%.+]]: !hw.union<e: i1, bits: i1>)
func.func @EventUnionConditional(%cond: !moore.i1, %lhs: !moore.uunion<{e: event, bits: i1}>, %rhs: !moore.uunion<{e: event, bits: i1}>) -> !moore.uunion<{e: event, bits: i1}> {
  // CHECK: comb.mux [[COND]], [[LHS]], [[RHS]] : !hw.union<e: i1, bits: i1>
  %result = moore.conditional %cond : i1 -> uunion<{e: event, bits: i1}> {
    moore.yield %lhs : uunion<{e: event, bits: i1}>
  } {
    moore.yield %rhs : uunion<{e: event, bits: i1}>
  }
  return %result : !moore.uunion<{e: event, bits: i1}>
}

// CHECK-LABEL: func.func @EventNetAndAssigned
// CHECK-SAME: ([[ARG:%.+]]: i1)
func.func @EventNetAndAssigned(%input: !moore.event) {
  // CHECK: [[FALSE:%.+]] = hw.constant false
  // CHECK: [[NET:%.+]] = llhd.sig [[FALSE]] : i1
  // CHECK: llhd.drv [[NET]], [[ARG]] after
  %net = moore.net wire %input : <!moore.event>
  // CHECK: hw.wire [[ARG]]  : i1
  %assigned = moore.assigned_variable %input : event
  return
}

// CHECK-LABEL: hw.module @EventWaitControls
moore.module @EventWaitControls() {
  %event = moore.variable : <event>
  %gate = moore.variable : <i1>

  // CHECK: llhd.process
  // CHECK: llhd.wait
  moore.procedure initial {
    moore.wait_event {
      %0 = moore.read %event : <event>
      moore.detect_event any %0 : event
    }
    moore.return
  }

  // CHECK: llhd.process
  // CHECK: llhd.wait
  // CHECK: comb.icmp bin ne
  // CHECK: comb.and bin
  moore.procedure initial {
    moore.wait_event {
      %0 = moore.read %event : <event>
      %1 = moore.read %gate : <i1>
      moore.detect_event any %0 if %1 : event
    }
    moore.return
  }
}

// CHECK-LABEL: llhd.global_signal @GlobalEvent : i1
moore.global_variable @GlobalEvent : !moore.event

// CHECK-LABEL: hw.module @EventGlobalValues
moore.module @EventGlobalValues() {
  // CHECK: llhd.process
  // CHECK: llhd.get_global_signal @GlobalEvent : <i1>
  // CHECK: llhd.prb
  // CHECK: hw.wire
  moore.procedure initial {
    %event = moore.get_global_variable @GlobalEvent : <!moore.event>
    %value = moore.read %event : <event>
    %assigned = moore.assigned_variable %value : event
    moore.return
  }
}

// CHECK-LABEL: hw.module @EventModulePorts
// CHECK-SAME: in %event_in : i1
// CHECK-SAME: out event_out : i1
moore.module @EventModulePorts(in %event_in : !moore.event, out event_out : !moore.event) {
  // CHECK: hw.output %event_in : i1
  moore.output %event_in : !moore.event
}

// CHECK-LABEL: hw.module @EventInstTop
// CHECK-SAME: in %event_in : i1
// CHECK-SAME: out event_out : i1
moore.module @EventInstTop(in %event_in : !moore.event, out event_out : !moore.event) {
  // CHECK: hw.instance "child" @EventInstChild(event_in: %event_in: i1) -> (event_out: i1)
  %child.event_out = moore.instance "child" @EventInstChild(event_in: %event_in : !moore.event) -> (event_out: !moore.event)
  // CHECK: hw.output %child.event_out : i1
  moore.output %child.event_out : !moore.event
}

// CHECK-LABEL: hw.module private @EventInstChild
moore.module private @EventInstChild(in %event_in : !moore.event, out event_out : !moore.event) {
  // CHECK: hw.output %event_in : i1
  moore.output %event_in : !moore.event
}

// CHECK-LABEL: hw.module @EventRefModulePorts
// CHECK-SAME: in %event_ref : !llhd.ref<i1>
moore.module @EventRefModulePorts(in %event_ref : !moore.ref<event>) {
  // CHECK: hw.output
  moore.output
}
