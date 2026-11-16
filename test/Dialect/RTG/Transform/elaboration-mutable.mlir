// RUN: circt-opt --rtg-elaborate=seed=0 --split-input-file --verify-diagnostics %s | FileCheck %s

// -----

// Test: mutable cells are collapsed during elaboration

func.func @consume(%arg0: index) {
  %text = rtg.int_format %arg0
  rtg.comment %text
  return
}

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @mut_ops
rtg.test @mut_ops() {
  %zero = index.constant 0
  %ref = rtg.mut_create %zero : index
  %v = rtg.mut_read %ref : !rtg.mut<index>
  %one = index.constant 1
  %v2 = index.add %v, %one
  rtg.mut_write %ref, %v2 : !rtg.mut<index>
  %v3 = rtg.mut_read %ref : !rtg.mut<index>
  func.call @consume(%v3) : (index) -> ()
}
// CHECK-NOT: rtg.mut_create
// CHECK-NOT: rtg.mut_read
// CHECK-NOT: rtg.mut_write
// The second read sees the written value (zero + one = 1).
// CHECK: [[C1:%.+]] = rtg.constant "1" : !rtg.string
// CHECK: rtg.comment [[C1]]

// -----

// Test: multi-shot with mutable state — the handler mutates shared state between
// resumes; each shot of the continuation observes the accumulated value.

rtg.effect @tick : () -> ()
func.func @log_val(%arg0: index) {
  %text = rtg.int_format %arg0
  rtg.comment %text
  return
}

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @multi_shot_mut_state
rtg.test @multi_shot_mut_state() {
  %zero = index.constant 0
  %ref  = rtg.mut_create %zero : index
  rtg.with_handlers {
    handle @tick(%k: !rtg.continuation<none>) {
      rtg.resume %k : !rtg.continuation<none>
      %one = index.constant 1
      rtg.mut_write %ref, %one : !rtg.mut<index>
      rtg.resume %k : !rtg.continuation<none>
      rtg.yield
    }
    do {
      rtg.perform @tick() : () -> none
      %v = rtg.mut_read %ref : !rtg.mut<index>
      func.call @log_val(%v) : (index) -> ()
      rtg.yield
    }
  }
}
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK-NOT: rtg.mut_create
// CHECK-NOT: rtg.mut_read
// CHECK-NOT: rtg.mut_write
// First shot sees initial value (0); second shot sees mutated value (1).
// CHECK: [[C0:%.+]] = rtg.constant "0" : !rtg.string
// CHECK: rtg.comment [[C0]]
// CHECK: [[C1:%.+]] = rtg.constant "1" : !rtg.string
// CHECK: rtg.comment [[C1]]

// -----

// Test: deep handler semantics — two sequential `perform` calls in the same
// body are both routed to the same handler. After the first perform's
// continuation runs, the handler must still be installed for the second
// perform to find. (Shallow handlers would handle only the first; the second
// would error as unhandled.) The handler increments a Mut on each call so we
// can observe the call count.

rtg.effect @ask : () -> index
func.func @record(%arg0: index) {
  %text = rtg.int_format %arg0
  rtg.comment %text
  return
}

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @deep_two_performs
rtg.test @deep_two_performs() {
  %zero = index.constant 0
  %ref  = rtg.mut_create %zero : index
  rtg.with_handlers {
    handle @ask(%k: !rtg.continuation<index>) {
      %v   = rtg.mut_read %ref : !rtg.mut<index>
      %one = index.constant 1
      %v1  = index.add %v, %one
      rtg.mut_write %ref, %v1 : !rtg.mut<index>
      rtg.resume %k, %v : !rtg.continuation<index>, index
      rtg.yield
    }
    do {
      %a = rtg.perform @ask() : () -> index
      func.call @record(%a) : (index) -> ()
      %b = rtg.perform @ask() : () -> index
      func.call @record(%b) : (index) -> ()
      rtg.yield
    }
  }
}
// Both performs are handled (no orphan-perform error). Handler ran twice;
// counter values 0 and 1 reach @record in order.
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK-NOT: rtg.mut_create
// CHECK: [[C0:%.+]] = rtg.constant "0" : !rtg.string
// CHECK: rtg.comment [[C0]]
// CHECK: [[C1:%.+]] = rtg.constant "1" : !rtg.string
// CHECK: rtg.comment [[C1]]

// -----

// Test: state visibility through an abort. With a multi-handler "state +
// abort" composition modeled by a Mut for state and a no-resume handler for
// abort, lexical ordering of writes matters: pre-perform writes from the
// body persist; the handler's own writes persist; post-perform writes from
// the body are skipped because the continuation was dropped. This pins the
// observable effect order so a future change cannot accidentally swap the
// handler-vs-body sequencing.

rtg.effect @abort_s : () -> ()
func.func @log_state(%arg0: index) {
  %text = rtg.int_format %arg0
  rtg.comment %text
  return
}

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @abort_state_ordering
rtg.test @abort_state_ordering() {
  %z   = index.constant 0
  %ref = rtg.mut_create %z : index
  rtg.with_handlers {
    handle @abort_s(%k: !rtg.continuation<none>) {
      // Handler runs after the body's pre-perform write, so this overrides.
      %c1 = index.constant 1
      rtg.mut_write %ref, %c1 : !rtg.mut<index>
      // No resume: body's post-perform write must be skipped.
      rtg.yield
    }
    do {
      %c2 = index.constant 2
      rtg.mut_write %ref, %c2 : !rtg.mut<index>
      rtg.perform @abort_s() : () -> none
      // Skipped:
      %c3 = index.constant 3
      rtg.mut_write %ref, %c3 : !rtg.mut<index>
      rtg.yield
    }
  }
  %final = rtg.mut_read %ref : !rtg.mut<index>
  func.call @log_state(%final) : (index) -> ()
}
// Final state is 1 (handler's write); body's `2` was overwritten by the
// handler, and the post-perform `3` write never ran.
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK-NOT: rtg.mut_create
// CHECK-NOT: rtg.mut_write
// CHECK: [[C1:%.+]] = rtg.constant "1" : !rtg.string
// CHECK: rtg.comment [[C1]]

// -----

// Test: continuation escape via empty set + Mut. An empty
// !rtg.set<!rtg.continuation<T>> requires no initial continuation value and
// can be used as the initial value for a !rtg.mut. The handler stores %k in
// the set (without resuming), and the outer code retrieves and resumes it
// after with_handlers exits. This is "deferred continuation" semantics:
// after_handlers runs first, then after_perform runs when %k is resumed.

rtg.effect @deferred_e : () -> ()
rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @cont_escape_deferred
rtg.test @cont_escape_deferred() {
  %empty = rtg.set_create : !rtg.continuation<none>
  %ref   = rtg.mut_create %empty : !rtg.set<!rtg.continuation<none>>

  rtg.with_handlers {
    handle @deferred_e(%k: !rtg.continuation<none>) {
      // Stash %k without resuming (abort the body).
      %s = rtg.set_create %k : !rtg.continuation<none>
      rtg.mut_write %ref, %s : !rtg.mut<!rtg.set<!rtg.continuation<none>>>
      rtg.yield
    }
    do {
      rtg.perform @deferred_e() : () -> none
      // Skipped by abort — becomes the continuation's remaining ops.
      %after_perform = rtg.constant "after_perform" : !rtg.string
      rtg.comment %after_perform
      rtg.yield
    }
  }

  // Runs before the deferred continuation.
  %after_handlers = rtg.constant "after_handlers" : !rtg.string
  rtg.comment %after_handlers

  // Retrieve and resume the escaped continuation.
  %s_out = rtg.mut_read %ref : !rtg.mut<!rtg.set<!rtg.continuation<none>>>
  %k_out = rtg.set_select_random %s_out : !rtg.set<!rtg.continuation<none>>
  rtg.resume %k_out : !rtg.continuation<none>
  rtg.yield
}
// after_handlers_d runs first; after_perform_d runs when %k_out is resumed.
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK: [[AFTER_HANDLERS:%.+]] = rtg.constant "after_handlers" : !rtg.string
// CHECK: rtg.comment [[AFTER_HANDLERS]]
// CHECK: [[AFTER_PERFORM:%.+]] = rtg.constant "after_perform" : !rtg.string
// CHECK: rtg.comment [[AFTER_PERFORM]]

// -----

// Recursive entry uses a fresh lexical frame for each handler invocation, but
// each cell still uses shared mutable storage.

rtg.effect @ask_local : (index) -> index
func.func @record_local(%value: index) {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @recursive_handler_mut_locals
rtg.test @recursive_handler_mut_locals() {
  %one = index.constant 1
  %two = index.constant 2
  rtg.with_handlers {
    handle @ask_local(%input: index, %k: !rtg.continuation<index>) {
      %local = rtg.mut_create %input : index
      rtg.resume %k, %input : !rtg.continuation<index>, index
      %value = rtg.mut_read %local : !rtg.mut<index>
      func.call @record_local(%value) : (index) -> ()
      rtg.yield
    }
    do {
      %first = rtg.perform @ask_local(%one) : (index) -> index
      %second = rtg.perform @ask_local(%two) : (index) -> index
      rtg.yield
    }
  }
}
// The nested handler invocation records 2 first, then the outer invocation
// resumes with its own local cell and records 1.
// CHECK: [[TWO:%.+]] = rtg.constant "2" : !rtg.string
// CHECK: rtg.comment [[TWO]]
// CHECK: [[ONE:%.+]] = rtg.constant "1" : !rtg.string
// CHECK: rtg.comment [[ONE]]

// -----

// A continuation captured in an opaque external region cannot be resumed after
// that shell has completed.

rtg.effect @opaque_continuation : () -> ()

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

rtg.test @continuation_escapes_external_region() {
  %empty = rtg.set_create : !rtg.continuation<none>
  %ref = rtg.mut_create %empty : !rtg.set<!rtg.continuation<none>>
  scf.execute_region {
    rtg.with_handlers {
      handle @opaque_continuation(%k: !rtg.continuation<none>) {
        %set = rtg.set_create %k : !rtg.continuation<none>
        // expected-error @below {{continuation cannot escape an opaque external-region shell}}
        rtg.mut_write %ref, %set : !rtg.mut<!rtg.set<!rtg.continuation<none>>>
        rtg.yield
      }
      do {
        rtg.perform @opaque_continuation() : () -> none
        rtg.yield
      }
    }
    scf.yield
  }
  %set = rtg.mut_read %ref : !rtg.mut<!rtg.set<!rtg.continuation<none>>>
  %k = rtg.set_select_random %set : !rtg.set<!rtg.continuation<none>>
  rtg.resume %k : !rtg.continuation<none>
}

// -----

// A continuation stored through a sequence input cannot be resumed outside
// the sequence invocation that captured it.

rtg.effect @sequence_continuation : () -> ()
rtg.sequence @stash_sequence_continuation(%ref: !rtg.mut<!rtg.set<!rtg.continuation<none>>>) {
  rtg.with_handlers {
    handle @sequence_continuation(%k: !rtg.continuation<none>) {
      %set = rtg.set_create %k : !rtg.continuation<none>
      // expected-error @below {{continuation cannot escape a sequence boundary}}
      rtg.mut_write %ref, %set : !rtg.mut<!rtg.set<!rtg.continuation<none>>>
      rtg.yield
    }
    do {
      rtg.perform @sequence_continuation() : () -> none
      rtg.yield
    }
  }
}

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

rtg.test @continuation_escapes_sequence() {
  %empty = rtg.set_create : !rtg.continuation<none>
  %ref = rtg.mut_create %empty : !rtg.set<!rtg.continuation<none>>
  %seq = rtg.get_sequence @stash_sequence_continuation : !rtg.sequence<!rtg.mut<!rtg.set<!rtg.continuation<none>>>>
  %bound = rtg.substitute_sequence %seq(%ref) : !rtg.sequence<!rtg.mut<!rtg.set<!rtg.continuation<none>>>>
  %randomized = rtg.randomize_sequence %bound
  rtg.embed_sequence %randomized
  %set = rtg.mut_read %ref : !rtg.mut<!rtg.set<!rtg.continuation<none>>>
  %k = rtg.set_select_random %set : !rtg.set<!rtg.continuation<none>>
  rtg.resume %k : !rtg.continuation<none>
}

// -----

// Test: perform inside scf.if then-branch. The continuation extends through
// the rest of the protected with_handlers body, including the scf.if
// completion and the outer-block suffix.

rtg.effect @ev_if : () -> index
func.func @record_if(%v: index) {
  %text = rtg.int_format %v
  rtg.comment %text
  return
}

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @scfif_handler_delimited_continuation
rtg.test @scfif_handler_delimited_continuation() {
  %zero = index.constant 0
  %one = index.constant 1
  %ref  = rtg.mut_create %zero : index
  %true = arith.constant true
  rtg.with_handlers {
    handle @ev_if(%k: !rtg.continuation<index>) {
      %c7 = index.constant 7
      rtg.resume %k, %c7 : !rtg.continuation<index>, index
      rtg.yield
    }
    do {
      // Resumption continues through this branch and then the outer suffix.
      scf.if %true {
        %v = rtg.perform @ev_if() : () -> index
        func.call @record_if(%v) : (index) -> ()
      }
      rtg.mut_write %ref, %one : !rtg.mut<index>
      %final = rtg.mut_read %ref : !rtg.mut<index>
      func.call @record_if(%final) : (index) -> ()
      rtg.yield
    }
  }
}
// Then-branch: record_if(7). The outer suffix observes the write of 1.
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK-NOT: rtg.mut_create
// CHECK: [[C7:%.+]] = rtg.constant "7" : !rtg.string
// CHECK: rtg.comment [[C7]]
// CHECK: [[C1:%.+]] = rtg.constant "1" : !rtg.string
// CHECK: rtg.comment [[C1]]

// -----

// Test: continuation passed as an effect INPUT argument.
// An escaped !rtg.continuation<index> (captured via set/mut) is forwarded as
// an operand to rtg.perform. The handler receives it as a block arg and
// resumes it directly with value 42 — proving continuations are valid effect
// operands and that the elaborator threads them through state correctly.

rtg.effect @relay_ci  : (!rtg.continuation<index>) -> ()
rtg.effect @supply_ci : () -> index
func.func @record_ci(%v: index) {
  %text = rtg.int_format %v
  rtg.comment %text
  return
}

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @cont_as_effect_input
rtg.test @cont_as_effect_input() {
  // Escape a !rtg.continuation<index> via abort + set/mut.
  %empty_ci = rtg.set_create : !rtg.continuation<index>
  %ref_ci   = rtg.mut_create %empty_ci : !rtg.set<!rtg.continuation<index>>
  rtg.with_handlers {
    handle @supply_ci(%k_s: !rtg.continuation<index>) {
      %s = rtg.set_create %k_s : !rtg.continuation<index>
      rtg.mut_write %ref_ci, %s : !rtg.mut<!rtg.set<!rtg.continuation<index>>>
      rtg.yield
    }
    do {
      %val = rtg.perform @supply_ci() : () -> index
      func.call @record_ci(%val) : (index) -> ()
      rtg.yield
    }
  }
  %s_ci  = rtg.mut_read %ref_ci : !rtg.mut<!rtg.set<!rtg.continuation<index>>>
  %k_held = rtg.set_select_random %s_ci : !rtg.set<!rtg.continuation<index>>

  // Pass the continuation as an effect input operand.
  rtg.with_handlers {
    handle @relay_ci(%cont_arg: !rtg.continuation<index>,
                     %k_r: !rtg.continuation<none>) {
      %c42 = index.constant 42
      rtg.resume %cont_arg, %c42 : !rtg.continuation<index>, index
      rtg.resume %k_r : !rtg.continuation<none>
      rtg.yield
    }
    do {
      rtg.perform @relay_ci(%k_held) : (!rtg.continuation<index>) -> none
      rtg.yield
    }
  }
}
// relay handler resumes the passed continuation with 42 → record_ci(42).
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK: [[C42:%.+]] = rtg.constant "42" : !rtg.string
// CHECK: rtg.comment [[C42]]

// -----

// Test: effect that RETURNS a continuation (!rtg.continuation<T> as effect
// result type). The handler for @give_cont resumes its own %k with a
// captured continuation value; the test body receives that continuation
// and resumes it with 99 — a second-order "give me a handle to resume later"
// pattern.

rtg.effect @give_cont_co : () -> !rtg.continuation<index>
rtg.effect @make_val_co  : () -> index
func.func @record_co(%v: index) {
  %text = rtg.int_format %v
  rtg.comment %text
  return
}

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @effect_returns_continuation
rtg.test @effect_returns_continuation() {
  // Escape a !rtg.continuation<index> to hand to @give_cont_co's handler.
  %empty_co = rtg.set_create : !rtg.continuation<index>
  %ref_co   = rtg.mut_create %empty_co : !rtg.set<!rtg.continuation<index>>
  rtg.with_handlers {
    handle @make_val_co(%k_mv: !rtg.continuation<index>) {
      %s = rtg.set_create %k_mv : !rtg.continuation<index>
      rtg.mut_write %ref_co, %s : !rtg.mut<!rtg.set<!rtg.continuation<index>>>
      rtg.yield
    }
    do {
      %v = rtg.perform @make_val_co() : () -> index
      func.call @record_co(%v) : (index) -> ()
      rtg.yield
    }
  }
  %s_co   = rtg.mut_read %ref_co : !rtg.mut<!rtg.set<!rtg.continuation<index>>>
  %k_held_co = rtg.set_select_random %s_co : !rtg.set<!rtg.continuation<index>>

  // @give_cont_co's continuation arg expects !rtg.continuation<index> as the
  // resume value (since that is the declared effect result type).
  rtg.with_handlers {
    handle @give_cont_co(%k: !rtg.continuation<!rtg.continuation<index>>) {
      rtg.resume %k, %k_held_co : !rtg.continuation<!rtg.continuation<index>>,
                                   !rtg.continuation<index>
      rtg.yield
    }
    do {
      %k_got = rtg.perform @give_cont_co() : () -> !rtg.continuation<index>
      %c99 = index.constant 99
      rtg.resume %k_got, %c99 : !rtg.continuation<index>, index
      rtg.yield
    }
  }
}
// Body resumes the returned continuation with 99 → record_co(99).
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK: [[C99:%.+]] = rtg.constant "99" : !rtg.string
// CHECK: rtg.comment [[C99]]

// -----

// Symbolic values used to initialize and overwrite a mutable cell must still
// pass through the mutable-cell interpreter.

rtg.target @symbolic_mut_target : !rtg.dict<reg: !rtgtest.ireg> {
  %reg = rtg.constant #rtgtest.t0
  rtg.yield %reg : !rtgtest.ireg
}

func.func @consume_symbolic_mut(%value: i32) {
  %text = rtg.immediate_format %value : i32
  rtg.comment %text
  return
}

// CHECK-LABEL: rtg.test @symbolic_mut_values
rtg.test @symbolic_mut_values(reg = %reg: !rtgtest.ireg) {
  %zero = rtg.constant 0 : i32
  %one = rtg.constant 1 : i32
  %initial = rtg.validate %reg, %zero, "mut_initial" : !rtgtest.ireg -> i32
  %ref = rtg.mut_create %initial : i32
  %written = rtg.validate %reg, %one, "mut_written" : !rtgtest.ireg -> i32
  rtg.mut_write %ref, %written : !rtg.mut<i32>
  %observed = rtg.mut_read %ref : !rtg.mut<i32>
  func.call @consume_symbolic_mut(%observed) : (i32) -> ()
}
// CHECK-NOT: rtg.mut_create
// CHECK-NOT: rtg.mut_write
// CHECK-NOT: rtg.mut_read
// CHECK: rtg.validate {{.*}}, [[ONE:%.+]], "mut_written" : !rtgtest.ireg -> i32
// CHECK: [[WRITTEN_TEXT:%.+]] = rtg.immediate_format {{.*}} : i32
// CHECK: rtg.comment [[WRITTEN_TEXT]]

// -----

// A perform handled by an outer installation removes the matching and inner
// frames while its clause runs. A further effect therefore reaches the
// surviving outer handler, and its mutable write remains visible.

rtg.effect @outer_mut_effect : () -> ()
rtg.effect @side_mut_effect : () -> ()
func.func @record_handler_cut(%value: index) {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @handler_cut_target : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @mutable_handler_stack_cut
rtg.test @mutable_handler_stack_cut() {
  %zero = index.constant 0
  %one = index.constant 1
  %two = index.constant 2
  %three = index.constant 3
  %ref = rtg.mut_create %zero : index
  rtg.with_handlers {
    handle @side_mut_effect(%root_k: !rtg.continuation<none>) {
      rtg.mut_write %ref, %three : !rtg.mut<index>
      rtg.resume %root_k : !rtg.continuation<none>
      rtg.yield
    }
    do {
      rtg.with_handlers {
        handle @outer_mut_effect(%outer_k: !rtg.continuation<none>) {
          rtg.mut_write %ref, %one : !rtg.mut<index>
          rtg.perform @side_mut_effect() : () -> none
          rtg.resume %outer_k : !rtg.continuation<none>
          rtg.yield
        }
        handle @side_mut_effect(%middle_k: !rtg.continuation<none>) {
          rtg.mut_write %ref, %two : !rtg.mut<index>
          rtg.resume %middle_k : !rtg.continuation<none>
          rtg.yield
        }
        do {
          rtg.with_handlers {
            handle @side_mut_effect(%inner_k: !rtg.continuation<none>) {
              rtg.mut_write %ref, %two : !rtg.mut<index>
              rtg.resume %inner_k : !rtg.continuation<none>
              rtg.yield
            }
            do {
              rtg.perform @outer_mut_effect() : () -> none
              rtg.yield
            }
          }
          rtg.yield
        }
      }
      %value = rtg.mut_read %ref : !rtg.mut<index>
      func.call @record_handler_cut(%value) : (index) -> ()
      rtg.yield
    }
  }
}
// The outer side-effect handler runs, rather than either handler in the
// matching or inner installations.
// CHECK: [[THREE:%.+]] = rtg.constant "3" : !rtg.string
// CHECK: rtg.comment [[THREE]]

// -----

// Dropping a continuation from a perform nested in scf.if must abort the
// remainder of the enclosing with_handlers body, not just the branch block.
rtg.effect @abort_from_if : () -> ()
func.func @record_after_abort(%value: index) {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @abort_skips_outer_if_suffix
rtg.test @abort_skips_outer_if_suffix() {
  %zero = index.constant 0
  %one = index.constant 1
  %ref = rtg.mut_create %zero : index
  %true = arith.constant true
  rtg.with_handlers {
    handle @abort_from_if(%k: !rtg.continuation<none>) {
      // Deliberately drop %k.
      rtg.yield
    }
    do {
      scf.if %true {
        rtg.perform @abort_from_if() : () -> none
      }
      // This write is in the handler's protected body but outside the scf.if.
      rtg.mut_write %ref, %one : !rtg.mut<index>
      rtg.yield
    }
  }
  %observed = rtg.mut_read %ref : !rtg.mut<index>
  func.call @record_after_abort(%observed) : (index) -> ()
}
// Handler-delimited semantics require the write to be skipped, leaving 0.
// CHECK: [[ZERO:%.+]] = rtg.constant "0" : !rtg.string
// CHECK: rtg.comment [[ZERO]]

// -----

// Resuming from inside both scf.if and scf.for must finish the branch, complete
// the current iteration, run later iterations, and then execute the body
// suffix. Only the first iteration performs the effect.
rtg.effect @choose_first_iteration : () -> index
func.func @record_loop_sum(%value: index) {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @resume_through_scffor_and_if
rtg.test @resume_through_scffor_and_if() {
  %zero = index.constant 0
  %one = index.constant 1
  %upper = index.constant 3
  rtg.with_handlers {
    handle @choose_first_iteration(%k: !rtg.continuation<index>) {
      %five = index.constant 5
      rtg.resume %k, %five : !rtg.continuation<index>, index
      rtg.yield
    }
    do {
      %sum = scf.for %i = %zero to %upper step %one
          iter_args(%acc = %zero) -> (index) {
        %first = index.cmp eq(%i, %zero)
        %term = scf.if %first -> index {
          %value = rtg.perform @choose_first_iteration() : () -> index
          scf.yield %value : index
        } else {
          scf.yield %one : index
        }
        %next = index.add %acc, %term
        scf.yield %next : index
      }
      func.call @record_loop_sum(%sum) : (index) -> ()
      rtg.yield
    }
  }
}
// The resumed continuation carries 5 through the first iteration and adds 1
// in each remaining iteration, so the protected-body suffix observes 7.
// CHECK: [[SEVEN:%.+]] = rtg.constant "7" : !rtg.string
// CHECK: rtg.comment [[SEVEN]]

// -----

// A continuation crossing a random_scope must run its scope completion before
// the protected-body suffix consumes the random_scope result.
rtg.effect @choose_in_random_scope : () -> index
func.func @record_random_scope(%value: index) {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @resume_through_random_scope
rtg.test @resume_through_random_scope() {
  %one = index.constant 1
  rtg.with_handlers {
    handle @choose_in_random_scope(%k: !rtg.continuation<index>) {
      %five = index.constant 5
      rtg.resume %k, %five : !rtg.continuation<index>, index
      rtg.yield
    }
    do {
      %value = rtg.random_scope seed 17 : index {
        %choice = rtg.perform @choose_in_random_scope() : () -> index
        %sum = index.add %choice, %one
        rtg.yield %sum : index
      }
      func.call @record_random_scope(%value) : (index) -> ()
      rtg.yield
    }
  }
}
// CHECK: [[SIX:%.+]] = rtg.constant "6" : !rtg.string
// CHECK: rtg.comment [[SIX]]

// -----

// Dropping from a loop body abandons the remaining iterations and the suffix
// of the protected handler body.
rtg.effect @abort_from_for : () -> ()
func.func @record_after_for_abort(%value: index) {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @abort_skips_scffor_suffix
rtg.test @abort_skips_scffor_suffix() {
  %zero = index.constant 0
  %one = index.constant 1
  %upper = index.constant 3
  %ref = rtg.mut_create %zero : index
  rtg.with_handlers {
    handle @abort_from_for(%k: !rtg.continuation<none>) {
      rtg.yield
    }
    do {
      scf.for %i = %zero to %upper step %one {
        %first = index.cmp eq(%i, %zero)
        scf.if %first {
          rtg.perform @abort_from_for() : () -> none
        }
      }
      rtg.mut_write %ref, %one : !rtg.mut<index>
      rtg.yield
    }
  }
  %observed = rtg.mut_read %ref : !rtg.mut<index>
  func.call @record_after_for_abort(%observed) : (index) -> ()
}
// Neither the rest of the loop nor the body suffix writes 1.
// CHECK: [[ZERO:%.+]] = rtg.constant "0" : !rtg.string
// CHECK: rtg.comment [[ZERO]]

// -----

// Each resume gets an independent copy of the loop-control and lexical state,
// while both shots materialize their body suffix in order.
rtg.effect @choose_loop_shot : () -> index
func.func @record_loop_shot(%value: index) {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @multi_shot_resume_through_scffor
rtg.test @multi_shot_resume_through_scffor() {
  %zero = index.constant 0
  %one = index.constant 1
  %upper = index.constant 3
  rtg.with_handlers {
    handle @choose_loop_shot(%k: !rtg.continuation<index>) {
      %five = index.constant 5
      rtg.resume %k, %five : !rtg.continuation<index>, index
      %seven = index.constant 7
      rtg.resume %k, %seven : !rtg.continuation<index>, index
      rtg.yield
    }
    do {
      %sum = scf.for %i = %zero to %upper step %one
          iter_args(%acc = %zero) -> (index) {
        %first = index.cmp eq(%i, %zero)
        %term = scf.if %first -> index {
          %value = rtg.perform @choose_loop_shot() : () -> index
          scf.yield %value : index
        } else {
          scf.yield %one : index
        }
        %next = index.add %acc, %term
        scf.yield %next : index
      }
      func.call @record_loop_shot(%sum) : (index) -> ()
      rtg.yield
    }
  }
}
// The two replayed loop results are 7 and 9, respectively.
// CHECK: [[SEVEN:%.+]] = rtg.constant "7" : !rtg.string
// CHECK: rtg.comment [[SEVEN]]
// CHECK: [[NINE:%.+]] = rtg.constant "9" : !rtg.string
// CHECK: rtg.comment [[NINE]]

// -----

// Separate cells with equal initial values must retain independent identity.
func.func @record_mutable_pair(%first: index, %second: index) -> () {
  %first_text = rtg.int_format %first
  rtg.comment %first_text
  %second_text = rtg.int_format %second
  rtg.comment %second_text
  return
}

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @distinct_mutable_cells
rtg.test @distinct_mutable_cells() {
  %zero = index.constant 0
  %one = index.constant 1
  %first = rtg.mut_create %zero : index
  %second = rtg.mut_create %zero : index
  rtg.mut_write %first, %one : !rtg.mut<index>
  %first_value = rtg.mut_read %first : !rtg.mut<index>
  %second_value = rtg.mut_read %second : !rtg.mut<index>
  func.call @record_mutable_pair(%first_value, %second_value) :
      (index, index) -> ()
}
// CHECK: [[ONE:%.+]] = rtg.constant "1" : !rtg.string
// CHECK: rtg.comment [[ONE]]
// CHECK: [[ZERO:%.+]] = rtg.constant "0" : !rtg.string
// CHECK: rtg.comment [[ZERO]]

// -----

// Mutable identity remains usable after passing through a tuple to a helper.
func.func @observe_mut_tuple(%tuple: !rtg.tuple<!rtg.mut<index>>) {
  %cell = rtg.tuple_extract %tuple at 0 : !rtg.tuple<!rtg.mut<index>>
  %value = rtg.mut_read %cell : !rtg.mut<index>
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

rtg.test @mutable_cell_passed_through_tuple() {
  %zero = index.constant 0
  %ref = rtg.mut_create %zero : index
  %tuple = rtg.tuple_create %ref : !rtg.mut<index>
  func.call @observe_mut_tuple(%tuple) : (!rtg.tuple<!rtg.mut<index>>) -> ()
}
// CHECK-LABEL: rtg.test @mutable_cell_passed_through_tuple
// CHECK: [[ZERO:%.+]] = rtg.constant "0" : !rtg.string
// CHECK: rtg.comment [[ZERO]]

// -----

rtg.effect @escape_tuple_continuation : () -> ()
func.func @resume_continuation_tuple(
    %tuple: !rtg.tuple<!rtg.continuation<none>>) {
  %continuation = rtg.tuple_extract %tuple at 0 :
      !rtg.tuple<!rtg.continuation<none>>
  rtg.resume %continuation : !rtg.continuation<none>
  return
}

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

rtg.test @continuation_passed_through_tuple_to_helper() {
  rtg.with_handlers {
    handle @escape_tuple_continuation(%k: !rtg.continuation<none>) {
      %tuple = rtg.tuple_create %k : !rtg.continuation<none>
      func.call @resume_continuation_tuple(%tuple) :
          (!rtg.tuple<!rtg.continuation<none>>) -> ()
      rtg.yield
    }
    do {
      rtg.perform @escape_tuple_continuation() : () -> none
      %text = rtg.constant "resumed" : !rtg.string
      rtg.comment %text
      rtg.yield
    }
  }
}
// The helper resumes the continuation carried through the tuple.
// CHECK-LABEL: rtg.test @continuation_passed_through_tuple_to_helper
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK: [[RESUMED:%.+]] = rtg.constant "resumed" : !rtg.string
// CHECK: rtg.comment [[RESUMED]]

// -----

// A mutable cell passed through an inlined helper as an effect operand remains
// shared with the handler and across multi-shot continuation replays.
rtg.effect @update_cell : (!rtg.mut<index>) -> index

func.func @helper_with_mutable_effect(%cell: !rtg.mut<index>) -> index {
  %value = rtg.perform @update_cell(%cell) : (!rtg.mut<index>) -> index
  %before = rtg.mut_read %cell : !rtg.mut<index>
  %one = index.constant 1
  %after = index.add %before, %one
  rtg.mut_write %cell, %after : !rtg.mut<index>
  %result = index.add %value, %before
  return %result : index
}

// Emit a trace comment so each replayed result is visible in the output.
func.func @record_mutable_effect(%value: index) -> () attributes {no_inline} {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @mutable_state_across_inlined_multishot_effect
rtg.test @mutable_state_across_inlined_multishot_effect() {
  %zero = index.constant 0
  %cell = rtg.mut_create %zero : index
  rtg.with_handlers {
    handle @update_cell(%handled_cell: !rtg.mut<index>,
                        %k: !rtg.continuation<index>) {
      %before = rtg.mut_read %handled_cell : !rtg.mut<index>
      %one = index.constant 1
      %updated = index.add %before, %one
      rtg.mut_write %handled_cell, %updated : !rtg.mut<index>
      %ten = index.constant 10
      rtg.resume %k, %ten : !rtg.continuation<index>, index

      %before_again = rtg.mut_read %handled_cell : !rtg.mut<index>
      %updated_again = index.add %before_again, %one
      rtg.mut_write %handled_cell, %updated_again : !rtg.mut<index>
      %twenty = index.constant 20
      rtg.resume %k, %twenty : !rtg.continuation<index>, index
      rtg.yield
    }
    do {
      %result = func.call @helper_with_mutable_effect(%cell) :
          (!rtg.mut<index>) -> index
      func.call @record_mutable_effect(%result) : (index) -> ()
      %final = rtg.mut_read %cell : !rtg.mut<index>
      func.call @record_mutable_effect(%final) : (index) -> ()
      rtg.yield
    }
  }
}
// Each handler write is visible to the replayed helper, whose own write is
// then visible to the next handler shot and to the caller suffix.
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK-NOT: rtg.mut_create
// CHECK-NOT: rtg.mut_read
// CHECK-NOT: rtg.mut_write
// CHECK-NOT: func.call @helper_with_mutable_effect
// CHECK: [[ELEVEN:%.+]] = rtg.constant "11" : !rtg.string
// CHECK: rtg.comment [[ELEVEN]]
// CHECK: [[TWO:%.+]] = rtg.constant "2" : !rtg.string
// CHECK: rtg.comment [[TWO]]
// CHECK: [[TWENTY_THREE:%.+]] = rtg.constant "23" : !rtg.string
// CHECK: rtg.comment [[TWENTY_THREE]]
// CHECK: [[FOUR:%.+]] = rtg.constant "4" : !rtg.string
// CHECK: rtg.comment [[FOUR]]

// -----

// Dropping a continuation captured inside a helper skips both the helper
// suffix and the caller suffix, but preserves the handler's mutation.
rtg.effect @abort_mutable_helper : (!rtg.mut<index>) -> ()

func.func @helper_with_aborted_effect(%cell: !rtg.mut<index>) {
  %two = index.constant 2
  rtg.perform @abort_mutable_helper(%cell) : (!rtg.mut<index>) -> none
  rtg.mut_write %cell, %two : !rtg.mut<index>
  return
}

// Emit a trace comment so the handler's final state is visible.
func.func @record_mutable_abort(%value: index) -> () attributes {no_inline} {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @abort_across_helper_preserves_mutable_effect
rtg.test @abort_across_helper_preserves_mutable_effect() {
  %zero = index.constant 0
  %cell = rtg.mut_create %zero : index
  rtg.with_handlers {
    handle @abort_mutable_helper(%handled_cell: !rtg.mut<index>,
                                 %k: !rtg.continuation<none>) {
      %one = index.constant 1
      rtg.mut_write %handled_cell, %one : !rtg.mut<index>
      rtg.yield
    }
    do {
      func.call @helper_with_aborted_effect(%cell) : (!rtg.mut<index>) -> ()
      %three = index.constant 3
      rtg.mut_write %cell, %three : !rtg.mut<index>
      rtg.yield
    }
  }
  %final = rtg.mut_read %cell : !rtg.mut<index>
  func.call @record_mutable_abort(%final) : (index) -> ()
}
// Neither the helper's write of 2 nor the caller's write of 3 runs.
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK-NOT: rtg.mut_create
// CHECK-NOT: rtg.mut_read
// CHECK-NOT: rtg.mut_write
// CHECK-NOT: func.call @helper_with_aborted_effect
// CHECK: [[ONE:%.+]] = rtg.constant "1" : !rtg.string
// CHECK: rtg.comment [[ONE]]
