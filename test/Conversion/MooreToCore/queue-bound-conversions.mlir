// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

moore.class.classdecl @QueueBoundClass {
}

// CHECK-LABEL: hw.module @QueueBoundConversions
moore.module @QueueBoundConversions() {
  moore.procedure initial {
    // CHECK: [[EMPTY_BOUND:%.+]] = sim.queue.empty : <i32, 4>
    // CHECK: [[BOUND_REF:%.+]] = llhd.sig [[EMPTY_BOUND]] : !sim.queue<i32, 4>
    %bound = moore.variable : <!moore.queue<i32, 4>>
    // CHECK: [[EMPTY_UNBOUND:%.+]] = sim.queue.empty : <i32, 0>
    // CHECK: [[UNBOUND_REF:%.+]] = llhd.sig [[EMPTY_UNBOUND]] : !sim.queue<i32, 0>
    %unbound = moore.variable : <!moore.queue<i32, 0>>
    // CHECK: [[EMPTY_STRUCT_BOUND:%.+]] = sim.queue.empty : <!hw.struct<a: i3, b: i2>, 4>
    // CHECK: [[STRUCT_BOUND_REF:%.+]] = llhd.sig [[EMPTY_STRUCT_BOUND]] : !sim.queue<!hw.struct<a: i3, b: i2>, 4>
    %struct_bound = moore.variable : <!moore.queue<ustruct<{a: i3, b: i2}>, 4>>
    // CHECK: [[EMPTY_STRUCT_UNBOUND:%.+]] = sim.queue.empty : <!hw.struct<a: i3, b: i2>, 0>
    // CHECK: [[STRUCT_UNBOUND_REF:%.+]] = llhd.sig [[EMPTY_STRUCT_UNBOUND]] : !sim.queue<!hw.struct<a: i3, b: i2>, 0>
    %struct_unbound = moore.variable : <!moore.queue<ustruct<{a: i3, b: i2}>, 0>>
    // CHECK: [[EMPTY_CLASS_BOUND:%.+]] = sim.queue.empty : <!llvm.ptr, 4>
    // CHECK: [[CLASS_BOUND_REF:%.+]] = llhd.sig [[EMPTY_CLASS_BOUND]] : !sim.queue<!llvm.ptr, 4>
    %class_bound = moore.variable : <!moore.queue<class<@QueueBoundClass>, 4>>
    // CHECK: [[EMPTY_CLASS_UNBOUND:%.+]] = sim.queue.empty : <!llvm.ptr, 0>
    // CHECK: [[CLASS_UNBOUND_REF:%.+]] = llhd.sig [[EMPTY_CLASS_UNBOUND]] : !sim.queue<!llvm.ptr, 0>
    %class_unbound = moore.variable : <!moore.queue<class<@QueueBoundClass>, 0>>
    // CHECK: [[EMPTY_REAL_BOUND:%.+]] = sim.queue.empty : <f64, 4>
    // CHECK: [[REAL_BOUND_REF:%.+]] = llhd.sig [[EMPTY_REAL_BOUND]] : !sim.queue<f64, 4>
    %real_bound = moore.variable : <!moore.queue<f64, 4>>
    // CHECK: [[EMPTY_REAL_UNBOUND:%.+]] = sim.queue.empty : <f64, 0>
    // CHECK: [[REAL_UNBOUND_REF:%.+]] = llhd.sig [[EMPTY_REAL_UNBOUND]] : !sim.queue<f64, 0>
    %real_unbound = moore.variable : <!moore.queue<f64, 0>>
    // CHECK: [[EMPTY_TIME_BOUND:%.+]] = sim.queue.empty : <!llhd.time, 4>
    // CHECK: [[TIME_BOUND_REF:%.+]] = llhd.sig [[EMPTY_TIME_BOUND]] : !sim.queue<!llhd.time, 4>
    %time_bound = moore.variable : <!moore.queue<time, 4>>
    // CHECK: [[EMPTY_TIME_UNBOUND:%.+]] = sim.queue.empty : <!llhd.time, 0>
    // CHECK: [[TIME_UNBOUND_REF:%.+]] = llhd.sig [[EMPTY_TIME_UNBOUND]] : !sim.queue<!llhd.time, 0>
    %time_unbound = moore.variable : <!moore.queue<time, 0>>

    // CHECK: [[BOUND:%.+]] = llhd.prb [[BOUND_REF]] : !sim.queue<i32, 4>
    %bound_v = moore.read %bound : <!moore.queue<i32, 4>>
    // CHECK: [[UNBOUND:%.+]] = llhd.prb [[UNBOUND_REF]] : !sim.queue<i32, 0>
    %unbound_v = moore.read %unbound : <!moore.queue<i32, 0>>
    // CHECK: [[STRUCT_BOUND:%.+]] = llhd.prb [[STRUCT_BOUND_REF]] : !sim.queue<!hw.struct<a: i3, b: i2>, 4>
    %struct_bound_v = moore.read %struct_bound : <!moore.queue<ustruct<{a: i3, b: i2}>, 4>>
    // CHECK: [[CLASS_BOUND:%.+]] = llhd.prb [[CLASS_BOUND_REF]] : !sim.queue<!llvm.ptr, 4>
    %class_bound_v = moore.read %class_bound : <!moore.queue<class<@QueueBoundClass>, 4>>
    // CHECK: [[REAL_BOUND:%.+]] = llhd.prb [[REAL_BOUND_REF]] : !sim.queue<f64, 4>
    %real_bound_v = moore.read %real_bound : <!moore.queue<f64, 4>>
    // CHECK: [[TIME_BOUND:%.+]] = llhd.prb [[TIME_BOUND_REF]] : !sim.queue<!llhd.time, 4>
    %time_bound_v = moore.read %time_bound : <!moore.queue<time, 4>>

    // CHECK: [[TO_UNBOUND:%.+]] = sim.queue.resize [[BOUND]] : <i32, 4> -> <i32, 0>
    %to_unbound = moore.conversion %bound_v : !moore.queue<i32, 4> -> !moore.queue<i32, 0>
    // CHECK: [[TO_BOUND:%.+]] = sim.queue.resize [[UNBOUND]] : <i32, 0> -> <i32, 4>
    %to_bound = moore.conversion %unbound_v : !moore.queue<i32, 0> -> !moore.queue<i32, 4>
    // CHECK: [[STRUCT_TO_UNBOUND:%.+]] = sim.queue.resize [[STRUCT_BOUND]] : <!hw.struct<a: i3, b: i2>, 4> -> <!hw.struct<a: i3, b: i2>, 0>
    %struct_to_unbound = moore.conversion %struct_bound_v : !moore.queue<ustruct<{a: i3, b: i2}>, 4> -> !moore.queue<ustruct<{a: i3, b: i2}>, 0>
    // CHECK: [[CLASS_TO_UNBOUND:%.+]] = sim.queue.resize [[CLASS_BOUND]] : <!llvm.ptr, 4> -> <!llvm.ptr, 0>
    %class_to_unbound = moore.conversion %class_bound_v : !moore.queue<class<@QueueBoundClass>, 4> -> !moore.queue<class<@QueueBoundClass>, 0>
    // CHECK: [[REAL_TO_UNBOUND:%.+]] = sim.queue.resize [[REAL_BOUND]] : <f64, 4> -> <f64, 0>
    %real_to_unbound = moore.conversion %real_bound_v : !moore.queue<f64, 4> -> !moore.queue<f64, 0>
    // CHECK: [[TIME_TO_UNBOUND:%.+]] = sim.queue.resize [[TIME_BOUND]] : <!llhd.time, 4> -> <!llhd.time, 0>
    %time_to_unbound = moore.conversion %time_bound_v : !moore.queue<time, 4> -> !moore.queue<time, 0>

    // CHECK: llhd.drv [[UNBOUND_REF]], [[TO_UNBOUND]]
    moore.blocking_assign %unbound, %to_unbound : !moore.queue<i32, 0>
    // CHECK: llhd.drv [[BOUND_REF]], [[TO_BOUND]]
    moore.blocking_assign %bound, %to_bound : !moore.queue<i32, 4>
    // CHECK: llhd.drv [[STRUCT_UNBOUND_REF]], [[STRUCT_TO_UNBOUND]]
    moore.blocking_assign %struct_unbound, %struct_to_unbound : !moore.queue<ustruct<{a: i3, b: i2}>, 0>
    // CHECK: llhd.drv [[CLASS_UNBOUND_REF]], [[CLASS_TO_UNBOUND]]
    moore.blocking_assign %class_unbound, %class_to_unbound : !moore.queue<class<@QueueBoundClass>, 0>
    // CHECK: llhd.drv [[REAL_UNBOUND_REF]], [[REAL_TO_UNBOUND]]
    moore.blocking_assign %real_unbound, %real_to_unbound : !moore.queue<f64, 0>
    // CHECK: llhd.drv [[TIME_UNBOUND_REF]], [[TIME_TO_UNBOUND]]
    moore.blocking_assign %time_unbound, %time_to_unbound : !moore.queue<time, 0>
    moore.return
  }
}

// CHECK-NOT: moore.conversion
