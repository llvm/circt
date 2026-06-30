// RUN: circt-opt %s --arc-lower-coroutines | FileCheck %s

// Empty coroutine with an immediate return. No resume blocks, so the state
// degenerates to an empty struct and the PC is dispatched by an unconditional
// branch.
// CHECK-LABEL: func.func @Empty
// CHECK-SAME: (%arg0: !hw.struct<>, %arg1: i2) -> (!hw.struct<>, i2)
arc.coroutine.define @Empty() {
  // CHECK: cf.br ^[[BB:.+]]
  // CHECK: ^[[BB]]:
  // CHECK: [[STATE:%.+]] = ub.poison : !hw.struct<>
  // CHECK: [[PC:%.+]] = hw.constant -2 : i2
  // CHECK: return [[STATE]], [[PC]] : !hw.struct<>, i2
  arc.coroutine.return
}

// CHECK-LABEL: func.func @HaltOnly
// CHECK-SAME: (%arg0: !hw.struct<>, %arg1: i2, %arg2: i42) -> (!hw.struct<>, i2)
arc.coroutine.define @HaltOnly(%arg0: i42) {
  // CHECK: [[STATE:%.+]] = ub.poison : !hw.struct<>
  // CHECK: [[PC:%.+]] = hw.constant -1 : i2
  // CHECK: return [[STATE]], [[PC]] : !hw.struct<>, i2
  arc.coroutine.halt
}

// CHECK-LABEL: func.func @ReturnOnly
// CHECK-SAME: (%arg0: !hw.struct<>, %arg1: i2, %arg2: i42) -> (!hw.struct<>, i2, i42)
arc.coroutine.define @ReturnOnly(%arg0: i42) -> i42 {
  // CHECK: cf.br ^[[BB:.+]](%arg2 : i42)
  // CHECK: ^[[BB]]([[ARG:%.+]]: i42):
  // CHECK: [[STATE:%.+]] = ub.poison : !hw.struct<>
  // CHECK: [[PC:%.+]] = hw.constant -2 : i2
  // CHECK: return [[STATE]], [[PC]], [[ARG]] : !hw.struct<>, i2, i42
  arc.coroutine.return %arg0 : i42
}

// A single yield whose resume block persists no state. No union variant is
// allocated; the yield returns a poison state and the trampoline passes only
// the fresh caller-supplied arguments.
// CHECK-LABEL: func.func @YieldStateless
// CHECK-SAME: (%arg0: !hw.struct<>, %arg1: i2, %arg2: i42) -> (!hw.struct<>, i2, i42)
arc.coroutine.define @YieldStateless(%arg0: i42) -> i42 {
  // CHECK: cf.switch %arg1 : i2, [
  // CHECK-NEXT: default: ^[[ENTRY:.+]](%arg2 : i42),
  // CHECK-NEXT: 1: ^[[TRAMPOLINE:.+]]
  // CHECK-NEXT: ]
  // CHECK: ^[[ENTRY]]([[ARG:%.+]]: i42):
  // CHECK: [[STATE:%.+]] = ub.poison : !hw.struct<>
  // CHECK: [[PC:%.+]] = hw.constant 1 : i2
  // CHECK: return [[STATE]], [[PC]], [[ARG]]
  arc.coroutine.yield (%arg0 : i42), ^bb1
  // CHECK: ^[[TRAMPOLINE]]:
  // CHECK: cf.br ^[[RESUME:.+]](%arg2 : i42)
  // CHECK: ^[[RESUME]]([[FRESH:%.+]]: i42):
  // CHECK: hw.constant -2 : i2
  // CHECK: return {{%.+}}, {{%.+}}, [[FRESH]]
^bb1(%arg1: i42):
  arc.coroutine.return %arg1 : i42
}

// A yield with an explicit destination operand; the value is persisted in the
// resume block's union variant.
// CHECK-LABEL: func.func @YieldPersist
// CHECK-SAME: (%arg0: !hw.union<r1: !hw.struct<f0: i42>>, %arg1: i2, %arg2: i42)
// CHECK-SAME: -> (!hw.union<r1: !hw.struct<f0: i42>>, i2, i42)
arc.coroutine.define @YieldPersist(%arg0: i42) -> i42 {
  // CHECK: cf.switch %arg1 : i2, [
  // CHECK-NEXT: default: ^[[ENTRY:.+]](%arg2 : i42),
  // CHECK-NEXT: 1: ^[[TRAMPOLINE:.+]]
  // CHECK-NEXT: ]
  // CHECK: ^[[ENTRY]]([[ARG:%.+]]: i42):
  // CHECK: [[VARIANT:%.+]] = hw.struct_create ([[ARG]]) : !hw.struct<f0: i42>
  // CHECK: [[STATE:%.+]] = hw.union_create "r1", [[VARIANT]]
  // CHECK: [[PC:%.+]] = hw.constant 1 : i2
  // CHECK: return [[STATE]], [[PC]], [[ARG]]
  arc.coroutine.yield (%arg0 : i42), ^bb1(%arg0 : i42)
  // CHECK: ^[[TRAMPOLINE]]:
  // CHECK: [[VARIANT2:%.+]] = hw.union_extract %arg0["r1"]
  // CHECK: [[FIELD:%.+]] = hw.struct_explode [[VARIANT2]]
  // CHECK: cf.br ^[[RESUME:.+]](%arg2, [[FIELD]] : i42, i42)
  // CHECK: ^[[RESUME]]([[FRESH:%.+]]: i42, [[OLD:%.+]]: i42):
  // CHECK: [[SUM:%.+]] = comb.add [[FRESH]], [[OLD]]
  // CHECK: return {{%.+}}, {{%.+}}, [[SUM]]
^bb1(%fresh: i42, %old: i42):
  %sum = comb.add %fresh, %old : i42
  arc.coroutine.return %sum : i42
}

// A value that is live across a yield without being a destination operand. It
// must be captured as a trailing block argument of the resume block and
// persisted.
// CHECK-LABEL: func.func @LiveAcrossYield
// CHECK-SAME: (%arg0: !hw.union<r1: !hw.struct<f0: i42>>, %arg1: i2, %arg2: i42)
arc.coroutine.define @LiveAcrossYield(%arg0: i42) -> i42 {
  // CHECK: ^[[ENTRY:.+]]([[ARG:%.+]]: i42):
  // CHECK: [[MUL:%.+]] = comb.mul [[ARG]], [[ARG]]
  // CHECK: [[VARIANT:%.+]] = hw.struct_create ([[MUL]]) : !hw.struct<f0: i42>
  // CHECK: hw.union_create "r1", [[VARIANT]]
  %0 = comb.mul %arg0, %arg0 : i42
  arc.coroutine.yield (%arg0 : i42), ^bb1
  // CHECK: [[VARIANT2:%.+]] = hw.union_extract %arg0["r1"]
  // CHECK: [[FIELD:%.+]] = hw.struct_explode [[VARIANT2]]
  // CHECK: cf.br ^[[RESUME:.+]](%arg2, [[FIELD]] : i42, i42)
  // CHECK: ^[[RESUME]]([[FRESH:%.+]]: i42, [[OLD:%.+]]: i42):
  // CHECK: comb.add [[OLD]], [[FRESH]]
^bb1(%fresh: i42):
  %1 = comb.add %0, %fresh : i42
  arc.coroutine.return %1 : i42
}

// A constant that is live across a yield is rematerialized in the resume
// block instead of being persisted; the coroutine ends up with no state.
// CHECK-LABEL: func.func @ConstAcrossYield
// CHECK-SAME: (%arg0: !hw.struct<>, %arg1: i2) -> (!hw.struct<>, i2, i42)
arc.coroutine.define @ConstAcrossYield() -> i42 {
  // CHECK: [[CONST:%.+]] = hw.constant 9001 : i42
  // CHECK: return {{%.+}}, {{%.+}}, [[CONST]]
  %0 = hw.constant 9001 : i42
  arc.coroutine.yield (%0 : i42), ^bb1
  // CHECK: [[CONST2:%.+]] = hw.constant 9001 : i42
  // CHECK: return {{%.+}}, {{%.+}}, [[CONST2]]
^bb1:
  arc.coroutine.return %0 : i42
}

// A value used across an ordinary branch without crossing a suspension point
// is not captured; the using block keeps referring to the dominating
// definition instead of receiving a block argument.
// CHECK-LABEL: func.func @DominatedUseAcrossBranch
// CHECK-SAME: (%arg0: !hw.struct<>, %arg1: i2, %arg2: i42)
arc.coroutine.define @DominatedUseAcrossBranch(%arg0: i42) -> i42 {
  // CHECK: ^{{.+}}([[ARG:%.+]]: i42):
  // CHECK: [[MUL:%.+]] = comb.mul [[ARG]], [[ARG]]
  // CHECK-NEXT: cf.br ^[[BB:.+]]{{$}}
  %0 = comb.mul %arg0, %arg0 : i42
  cf.br ^bb1
  // CHECK: ^[[BB]]:
  // CHECK: comb.add [[MUL]], [[ARG]]
^bb1:
  %1 = comb.add %0, %arg0 : i42
  arc.coroutine.yield (%1 : i42), ^bb2
^bb2(%fresh: i42):
  arc.coroutine.return %fresh : i42
}

// A value live across a yield is captured as a trailing block argument of the
// resume block only. Blocks downstream of the resume block use the captured
// value through dominance and receive no arguments of their own.
// CHECK-LABEL: func.func @CaptureOnlyAtResume
// CHECK-SAME: (%arg0: !hw.union<r1: !hw.struct<f0: i42>>, %arg1: i2, %arg2: i42)
arc.coroutine.define @CaptureOnlyAtResume(%arg0: i42) -> i42 {
  // CHECK: ^{{.+}}([[ARG:%.+]]: i42):
  // CHECK: [[MUL:%.+]] = comb.mul [[ARG]], [[ARG]]
  // CHECK: hw.struct_create ([[MUL]])
  %0 = comb.mul %arg0, %arg0 : i42
  arc.coroutine.yield (%arg0 : i42), ^bb1
  // CHECK: ^[[RESUME:.+]]([[FRESH:%.+]]: i42, [[OLD:%.+]]: i42):
  // CHECK-NEXT: cf.br ^[[TAIL:.+]]{{$}}
^bb1(%fresh: i42):
  cf.br ^bb2
  // CHECK: ^[[TAIL]]:
  // CHECK: [[ADD:%.+]] = comb.add [[OLD]], [[FRESH]]
  // CHECK: return {{%.+}}, {{%.+}}, [[ADD]]
^bb2:
  %1 = comb.add %0, %fresh : i42
  arc.coroutine.return %1 : i42
}

// Where a path through a resume block rejoins a path carrying the original
// definition, the join block receives a merging block argument. Only the
// resume block and the join block are touched.
// CHECK-LABEL: func.func @RejoinAfterResume
// CHECK-SAME: (%arg0: !hw.union<r1: !hw.struct<f0: i42>>, %arg1: i2, %arg2: i1, %arg3: i42)
arc.coroutine.define @RejoinAfterResume(%arg0: i1, %arg1: i42) -> i42 {
  // CHECK: ^{{.+}}([[COND:%.+]]: i1, [[INIT:%.+]]: i42):
  // CHECK: [[MUL:%.+]] = comb.mul [[INIT]], [[INIT]]
  // CHECK: cf.cond_br [[COND]], ^[[SUSPEND:.+]], ^[[JOIN:.+]]([[MUL]] : i42)
  %0 = comb.mul %arg1, %arg1 : i42
  cf.cond_br %arg0, ^suspend, ^join
  // CHECK: ^[[SUSPEND]]:
  // CHECK: hw.struct_create ([[MUL]])
^suspend:
  arc.coroutine.yield (%arg1 : i42), ^resume
  // CHECK: ^{{.+}}({{%.+}}: i1, {{%.+}}: i42, [[OLD:%.+]]: i42):
  // CHECK-NEXT: cf.br ^[[JOIN]]([[OLD]] : i42)
^resume(%f0: i1, %f1: i42):
  cf.br ^join
  // CHECK: ^[[JOIN]]([[MERGED:%.+]]: i42):
  // CHECK: return {{%.+}}, {{%.+}}, [[MERGED]]
^join:
  arc.coroutine.return %0 : i42
}

// A merge block whose predecessors all carry the original definition does not
// receive a block argument; only the resume block captures the value.
// CHECK-LABEL: func.func @MergeBeforeYield
// CHECK-SAME: (%arg0: !hw.union<r1: !hw.struct<f0: i42>>, %arg1: i2, %arg2: i1, %arg3: i42)
arc.coroutine.define @MergeBeforeYield(%arg0: i1, %arg1: i42) -> i42 {
  // CHECK: ^{{.+}}([[COND:%.+]]: i1, [[INIT:%.+]]: i42):
  // CHECK: [[MUL:%.+]] = comb.mul [[INIT]], [[INIT]]
  // CHECK: cf.cond_br [[COND]], ^[[A:.+]], ^[[B:.+]]{{$}}
  %0 = comb.mul %arg1, %arg1 : i42
  cf.cond_br %arg0, ^a, ^b
  // CHECK: ^[[A]]:
  // CHECK-NEXT: cf.br ^[[MERGE:.+]]{{$}}
^a:
  cf.br ^merge
  // CHECK: ^[[B]]:
  // CHECK-NEXT: cf.br ^[[MERGE]]{{$}}
^b:
  cf.br ^merge
  // CHECK: ^[[MERGE]]:
  // CHECK: hw.struct_create ([[MUL]])
^merge:
  arc.coroutine.yield (%arg1 : i42), ^resume
  // CHECK: ^{{.+}}({{%.+}}: i1, {{%.+}}: i42, [[OLD:%.+]]: i42):
  // CHECK: return {{%.+}}, {{%.+}}, [[OLD]]
^resume(%f0: i1, %f1: i42):
  arc.coroutine.return %0 : i42
}

// All three terminators in one coroutine.
// CHECK-LABEL: func.func @AllTerminators
arc.coroutine.define @AllTerminators(%arg0: i1) -> i8 {
  %c0 = hw.constant 0 : i8
  cf.cond_br %arg0, ^suspend, ^stop
  // CHECK: hw.constant 1 : i2
  // CHECK: return
^suspend:
  arc.coroutine.yield (%c0 : i8), ^resume
  // CHECK: hw.constant -2 : i2
  // CHECK: return
^resume(%fresh: i1):
  %c1 = hw.constant 1 : i8
  arc.coroutine.return %c1 : i8
  // CHECK: hw.constant -1 : i2
  // CHECK: return
^stop:
  arc.coroutine.halt %c0 : i8
}

// Two yields targeting the same resume block share the same PC and union
// variant.
// CHECK-LABEL: func.func @SharedResume
// CHECK-SAME: (%arg0: !hw.union<r1: !hw.struct<f0: i32>>, %arg1: i2, %arg2: i1)
arc.coroutine.define @SharedResume(%arg0: i1) -> i32 {
  // CHECK: cf.switch %arg1 : i2, [
  // CHECK-NEXT: default: ^{{.+}}(%arg2 : i1),
  // CHECK-NEXT: 1: ^{{.+}}
  // CHECK-NEXT: ]
  %c0 = hw.constant 0 : i32
  %c1 = hw.constant 1 : i32
  cf.cond_br %arg0, ^a, ^b
  // CHECK: hw.union_create "r1"
  // CHECK: hw.constant 1 : i2
  // CHECK: return
^a:
  arc.coroutine.yield (%c0 : i32), ^resume(%c0 : i32)
  // CHECK: hw.union_create "r1"
  // CHECK: hw.constant 1 : i2
  // CHECK: return
^b:
  arc.coroutine.yield (%c1 : i32), ^resume(%c1 : i32)
  // CHECK: hw.union_extract %arg0["r1"]
  // CHECK: hw.struct_explode
^resume(%fresh: i1, %old: i32):
  arc.coroutine.return %old : i32
}

// Child coroutine used by the nesting tests below.
// CHECK-LABEL: func.func @Child
// CHECK-SAME: (%arg0: !hw.union<r1: !hw.struct<f0: i8>>, %arg1: i2)
// CHECK-SAME: -> (!hw.union<r1: !hw.struct<f0: i8>>, i2, i8)
arc.coroutine.define @Child() -> i8 {
  %c1 = hw.constant 1 : i8
  arc.coroutine.yield (%c1 : i8), ^bb1(%c1 : i8)
^bb1(%old: i8):
  arc.coroutine.return %old : i8
}

// A nested coroutine call. The call lowers to a regular function call, and
// the sentinel value ops lower to constants and comparisons.
// CHECK-LABEL: func.func @NestedCall
arc.coroutine.define @NestedCall() -> i1 {
  // CHECK: [[STATE:%.+]] = ub.poison : !hw.union<r1: !hw.struct<f0: i8>>
  // CHECK: [[PC:%.+]] = hw.constant 0 : i2
  // CHECK: [[CALL:%.+]]:3 = call @Child([[STATE]], [[PC]])
  // CHECK-SAME: (!hw.union<r1: !hw.struct<f0: i8>>, i2) -> (!hw.union<r1: !hw.struct<f0: i8>>, i2, i8)
  // CHECK: [[SENTINEL:%.+]] = hw.constant -2 : i2
  // CHECK: [[DONE:%.+]] = comb.icmp eq [[CALL]]#1, [[SENTINEL]] : i2
  // CHECK: return {{%.+}}, {{%.+}}, [[DONE]]
  %state = arc.coroutine.undefined_state : !arc.coroutine_state<@Child>
  %pc = arc.coroutine.start_pc : !arc.coroutine_pc<@Child>
  %s, %p, %r = arc.coroutine.call @Child(%state, %pc) : (!arc.coroutine_state<@Child>, !arc.coroutine_pc<@Child>) -> (!arc.coroutine_state<@Child>, !arc.coroutine_pc<@Child>, i8)
  %done = arc.coroutine.pc_is_return %p : !arc.coroutine_pc<@Child>
  arc.coroutine.return %done : i1
}

// A nested coroutine call carried across a yield. The child's state and PC
// are embedded in the parent's persisted state variant.
// CHECK-LABEL: func.func @NestedCarried
// CHECK-SAME: (%arg0: !hw.union<r1: !hw.struct<f0: !hw.union<r1: !hw.struct<f0: i8>>, f1: i2>>, %arg1: i2)
arc.coroutine.define @NestedCarried() -> i8 {
  %state = arc.coroutine.undefined_state : !arc.coroutine_state<@Child>
  %pc = arc.coroutine.start_pc : !arc.coroutine_pc<@Child>
  cf.br ^loop(%state, %pc : !arc.coroutine_state<@Child>, !arc.coroutine_pc<@Child>)
  // CHECK: [[CALL:%.+]]:3 = call @Child
  // CHECK: [[SENTINEL:%.+]] = hw.constant -1 : i2
  // CHECK: comb.icmp eq [[CALL]]#1, [[SENTINEL]] : i2
^loop(%s: !arc.coroutine_state<@Child>, %p: !arc.coroutine_pc<@Child>):
  %s2, %p2, %r = arc.coroutine.call @Child(%s, %p) : (!arc.coroutine_state<@Child>, !arc.coroutine_pc<@Child>) -> (!arc.coroutine_state<@Child>, !arc.coroutine_pc<@Child>, i8)
  %halted = arc.coroutine.pc_is_halt %p2 : !arc.coroutine_pc<@Child>
  cf.cond_br %halted, ^done, ^suspend
  // CHECK: [[VARIANT:%.+]] = hw.struct_create ({{%.+}}, {{%.+}}) : !hw.struct<f0: !hw.union<r1: !hw.struct<f0: i8>>, f1: i2>
  // CHECK: hw.union_create "r1", [[VARIANT]]
  // CHECK: hw.constant 1 : i2
  // CHECK: return
^suspend:
  arc.coroutine.yield (%r : i8), ^resume(%s2, %p2 : !arc.coroutine_state<@Child>, !arc.coroutine_pc<@Child>)
  // CHECK: [[VARIANT2:%.+]] = hw.union_extract %arg0["r1"]
  // CHECK: [[F0:%.+]], [[F1:%.+]] = hw.struct_explode [[VARIANT2]]
^resume(%s3: !arc.coroutine_state<@Child>, %p3: !arc.coroutine_pc<@Child>):
  cf.br ^loop(%s3, %p3 : !arc.coroutine_state<@Child>, !arc.coroutine_pc<@Child>)
^done:
  arc.coroutine.return %r : i8
}

// A non-coroutine driver function polling a coroutine in a loop. The state
// and PC flow through loop-carried block arguments, which must be retyped.
// CHECK-LABEL: func.func @Driver
func.func @Driver() -> i8 {
  // CHECK: [[STATE:%.+]] = ub.poison : !hw.union<r1: !hw.struct<f0: i8>>
  // CHECK: [[PC:%.+]] = hw.constant 0 : i2
  // CHECK: cf.br ^[[LOOP:.+]]([[STATE]], [[PC]] : !hw.union<r1: !hw.struct<f0: i8>>, i2)
  %state = arc.coroutine.undefined_state : !arc.coroutine_state<@Child>
  %pc = arc.coroutine.start_pc : !arc.coroutine_pc<@Child>
  cf.br ^loop(%state, %pc : !arc.coroutine_state<@Child>, !arc.coroutine_pc<@Child>)
  // CHECK: ^[[LOOP]]([[S:%.+]]: !hw.union<r1: !hw.struct<f0: i8>>, [[P:%.+]]: i2):
  // CHECK: [[CALL:%.+]]:3 = call @Child([[S]], [[P]])
  // CHECK: [[DONE:%.+]] = comb.icmp eq [[CALL]]#1
  // CHECK: cf.cond_br [[DONE]], ^[[EXIT:.+]], ^[[LOOP]]([[CALL]]#0, [[CALL]]#1 : !hw.union<r1: !hw.struct<f0: i8>>, i2)
^loop(%s: !arc.coroutine_state<@Child>, %p: !arc.coroutine_pc<@Child>):
  %s2, %p2, %r = arc.coroutine.call @Child(%s, %p) : (!arc.coroutine_state<@Child>, !arc.coroutine_pc<@Child>) -> (!arc.coroutine_state<@Child>, !arc.coroutine_pc<@Child>, i8)
  %done = arc.coroutine.pc_is_return %p2 : !arc.coroutine_pc<@Child>
  cf.cond_br %done, ^exit, ^loop(%s2, %p2 : !arc.coroutine_state<@Child>, !arc.coroutine_pc<@Child>)
  // CHECK: ^[[EXIT]]:
  // CHECK: return [[CALL]]#2 : i8
^exit:
  return %r : i8
}

// Unrelated ops referring to the opaque types directly. The function
// signatures and call types must be concretized.
// CHECK-LABEL: func.func private @Helper(!hw.union<r1: !hw.struct<f0: i8>>) -> i2
func.func private @Helper(!arc.coroutine_state<@Child>) -> !arc.coroutine_pc<@Child>

// CHECK-LABEL: func.func @CallsHelper
// CHECK-SAME: (%arg0: !hw.union<r1: !hw.struct<f0: i8>>) -> i2
func.func @CallsHelper(%arg0: !arc.coroutine_state<@Child>) -> !arc.coroutine_pc<@Child> {
  // CHECK: [[RESULT:%.+]] = call @Helper(%arg0) : (!hw.union<r1: !hw.struct<f0: i8>>) -> i2
  // CHECK: return [[RESULT]] : i2
  %0 = func.call @Helper(%arg0) : (!arc.coroutine_state<@Child>) -> !arc.coroutine_pc<@Child>
  return %0 : !arc.coroutine_pc<@Child>
}

// Opaque types nested within aggregate types on unrelated ops.
// CHECK-LABEL: func.func private @AggregateHelper
// CHECK-SAME: (!hw.struct<s: !hw.union<r1: !hw.struct<f0: i8>>, p: !hw.union<a: i2, b: i1>>)
// CHECK-SAME: -> !hw.array<2xstruct<x: i2>>
func.func private @AggregateHelper(!hw.struct<s: !arc.coroutine_state<@Child>, p: !hw.union<a: !arc.coroutine_pc<@Child>, b: i1>>) -> !hw.array<2xstruct<x: !arc.coroutine_pc<@Child>>>
