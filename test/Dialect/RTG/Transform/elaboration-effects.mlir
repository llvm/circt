// RUN: circt-opt --rtg-elaborate=seed=0 --split-input-file --verify-diagnostics %s | FileCheck %s

// -----

// Test: simple handler (auto-resume path) — choose effect

rtg.effect @choose : (!rtg.set<index>) -> index
func.func @use_index(%arg0: index) -> () { return }

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @simple_choose
rtg.test @simple_choose() {
  %c2 = index.constant 2
  %c3 = index.constant 3
  %set = rtg.set_create %c2, %c3 : index
  rtg.with_handlers {
    handle @choose(%s: !rtg.set<index>, %k: !rtg.continuation<index>) {
      %sel = rtg.set_select_random %s : !rtg.set<index>
      rtg.resume %k, %sel : !rtg.continuation<index>, index
      rtg.yield
    }
    do {
      %result = rtg.perform @choose(%set) : (!rtg.set<index>) -> index
      func.call @use_index(%result) : (index) -> ()
      rtg.yield
    }
  }
}
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK: func.call @use_index

// -----

// Test: void handler — log effect

rtg.effect @log : (index) -> ()
func.func @log_sink(%arg0: index) -> () { return }

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @void_handler
rtg.test @void_handler() {
  rtg.with_handlers {
    handle @log(%n: index, %k: !rtg.continuation<none>) {
      func.call @log_sink(%n) : (index) -> ()
      rtg.resume %k : !rtg.continuation<none>
      rtg.yield
    }
    do {
      %c42 = index.constant 42
      rtg.perform @log(%c42) : (index) -> none
      rtg.yield
    }
  }
}
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK: func.call @log_sink

// -----

// Test: multi-shot (void) — resume the same continuation twice; body runs both times.

rtg.effect @signal : () -> ()
func.func @on_signal() -> () { return }

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @multi_shot_void
rtg.test @multi_shot_void() {
  rtg.with_handlers {
    handle @signal(%k: !rtg.continuation<none>) {
      rtg.resume %k : !rtg.continuation<none>
      rtg.resume %k : !rtg.continuation<none>
      rtg.yield
    }
    do {
      rtg.perform @signal() : () -> none
      func.call @on_signal() : () -> ()
      rtg.yield
    }
  }
}
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// Body runs twice — two calls to @on_signal.
// CHECK: func.call @on_signal
// CHECK: func.call @on_signal

// -----

// Test: multi-shot (valued) — each resume injects a distinct value into the continuation body.

rtg.effect @choose_val : () -> index
func.func @record(%arg0: index) -> () { return }

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @multi_shot_values
rtg.test @multi_shot_values() {
  rtg.with_handlers {
    handle @choose_val(%k: !rtg.continuation<index>) {
      %c7  = index.constant 7
      rtg.resume %k, %c7  : !rtg.continuation<index>, index
      %c13 = index.constant 13
      rtg.resume %k, %c13 : !rtg.continuation<index>, index
      rtg.yield
    }
    do {
      %v = rtg.perform @choose_val() : () -> index
      func.call @record(%v) : (index) -> ()
      rtg.yield
    }
  }
}
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// First shot: record(7), second shot: record(13), in order.
// CHECK: [[C7:%.+]] = rtg.constant 7 : index
// CHECK: func.call @record([[C7]])
// CHECK: [[C13:%.+]] = rtg.constant 13 : index
// CHECK: func.call @record([[C13]])

// -----

// Test: code after rtg.resume executes after the continuation body returns.

rtg.effect @step : () -> ()
func.func @body_work() -> () { return }
func.func @post_work() -> () { return }

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @code_after_resume
rtg.test @code_after_resume() {
  rtg.with_handlers {
    handle @step(%k: !rtg.continuation<none>) {
      rtg.resume %k : !rtg.continuation<none>
      func.call @post_work() : () -> ()
      rtg.yield
    }
    do {
      rtg.perform @step() : () -> none
      func.call @body_work() : () -> ()
      rtg.yield
    }
  }
}
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// body_work (continuation) runs first, then post_work (after the resume call).
// CHECK: func.call @body_work
// CHECK: func.call @post_work

// -----

// Test: sequences (get_sequence / substitute_sequence / randomize_sequence /
// embed_sequence) work inside a handler region.

rtg.effect @emit_item : (index) -> ()
func.func @item_func(%arg0: index) -> () { return }

rtg.sequence @item_seq(%v: index) {
  func.call @item_func(%v) : (index) -> ()
}

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @seq_in_handler
rtg.test @seq_in_handler() {
  %c1 = index.constant 1
  %c2 = index.constant 2
  %set = rtg.set_create %c1, %c2 : index
  rtg.with_handlers {
    handle @emit_item(%n: index, %k: !rtg.continuation<none>) {
      %seq   = rtg.get_sequence @item_seq : !rtg.sequence<index>
      %bound = rtg.substitute_sequence %seq(%n) : !rtg.sequence<index>
      %rand  = rtg.randomize_sequence %bound
      rtg.embed_sequence %rand
      rtg.resume %k : !rtg.continuation<none>
      rtg.yield
    }
    do {
      %sel = rtg.set_select_random %set : !rtg.set<index>
      rtg.perform @emit_item(%sel) : (index) -> none
      rtg.yield
    }
  }
}
// Handler-region sequence ops are elaborated: with_handlers/perform/resume gone,
// substitute_sequence resolved to a concrete instantiation.
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK-NOT: rtg.substitute_sequence
// The instantiated sequence is embedded in the test body.
// CHECK: rtg.embed_sequence

// -----

// Test: randomization ops (set_select_random on a set passed as an effect
// operand) combined with sequence embedding both work inside a handler region.

rtg.effect @choose_and_emit : (!rtg.set<index>) -> ()
func.func @work_func(%arg0: index) -> () { return }

rtg.sequence @work_seq(%v: index) {
  func.call @work_func(%v) : (index) -> ()
}

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @rand_and_seq_in_handler
rtg.test @rand_and_seq_in_handler() {
  %c10 = index.constant 10
  %c20 = index.constant 20
  %set  = rtg.set_create %c10, %c20 : index
  rtg.with_handlers {
    handle @choose_and_emit(%s: !rtg.set<index>, %k: !rtg.continuation<none>) {
      %sel   = rtg.set_select_random %s : !rtg.set<index>
      %seq   = rtg.get_sequence @work_seq : !rtg.sequence<index>
      %bound = rtg.substitute_sequence %seq(%sel) : !rtg.sequence<index>
      %rand  = rtg.randomize_sequence %bound
      rtg.embed_sequence %rand
      rtg.resume %k : !rtg.continuation<none>
      rtg.yield
    }
    do {
      rtg.perform @choose_and_emit(%set) : (!rtg.set<index>) -> none
      rtg.yield
    }
  }
}
// with_handlers/perform/resume gone; randomization and substitute_sequence resolved.
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK-NOT: rtg.set_select_random
// CHECK-NOT: rtg.substitute_sequence
// The concretely-selected sequence is embedded in the test body.
// CHECK: rtg.embed_sequence

// -----

// Test: deep handler — handler region itself can `perform` an outer effect.
// `@inner` is handled by the inner with_handlers; its handler body performs
// `@outer`, which must reach the outer with_handlers (the handler stack
// retains outer frames even while running an inner handler). This proves
// the `inner -> outer` dispatch path of the deep stack.

rtg.effect @outer : () -> index
rtg.effect @inner : () -> ()
func.func @sink(%arg0: index) -> () { return }

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @deep_handler_calls_outer_effect
rtg.test @deep_handler_calls_outer_effect() {
  rtg.with_handlers {
    handle @outer(%k: !rtg.continuation<index>) {
      %c7 = index.constant 7
      rtg.resume %k, %c7 : !rtg.continuation<index>, index
      rtg.yield
    }
    do {
      rtg.with_handlers {
        handle @inner(%k: !rtg.continuation<none>) {
          // Inner handler reaches up through the stack to perform @outer.
          %v = rtg.perform @outer() : () -> index
          func.call @sink(%v) : (index) -> ()
          rtg.resume %k : !rtg.continuation<none>
          rtg.yield
        }
        do {
          rtg.perform @inner() : () -> none
          rtg.yield
        }
      }
      rtg.yield
    }
  }
}
// All effect ops elaborated; @outer's handler ran for the inner's perform,
// emitting sink(7).
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK: [[C7:%.+]] = rtg.constant 7 : index
// CHECK: func.call @sink([[C7]])

// -----

// Test: deep handler — outer effect performed *after* an inner with_handlers
// scope has exited. The inner scope must not consume or shadow the outer
// frame; subsequent performs of @outer in the outer body still find the
// outer handler.

rtg.effect @outer2 : () -> index
rtg.effect @inner2 : () -> ()
func.func @sink2(%arg0: index) -> () { return }

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @deep_outer_survives_inner_scope
rtg.test @deep_outer_survives_inner_scope() {
  rtg.with_handlers {
    handle @outer2(%k: !rtg.continuation<index>) {
      %c5 = index.constant 5
      rtg.resume %k, %c5 : !rtg.continuation<index>, index
      rtg.yield
    }
    do {
      rtg.with_handlers {
        handle @inner2(%k: !rtg.continuation<none>) {
          rtg.resume %k : !rtg.continuation<none>
          rtg.yield
        }
        do {
          rtg.perform @inner2() : () -> none
          rtg.yield
        }
      }
      // Inner scope exited; outer must still be installed.
      %v = rtg.perform @outer2() : () -> index
      func.call @sink2(%v) : (index) -> ()
      rtg.yield
    }
  }
}
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK: [[C5:%.+]] = rtg.constant 5 : index
// CHECK: func.call @sink2([[C5]])

// -----

// Test: multi-shot continuation whose body re-performs the *same* effect.
// During the continuation invoked by an outer multi-shot resume, the body
// runs another `rtg.perform` of the handled effect, which recursively
// re-enters the same handler region. This stresses block-arg state across
// the re-entry: the outer handler's `%k` SSA value would otherwise be
// rebound to the inner frame's continuation, causing the outer handler's
// next `rtg.resume %k, ...` to resume the wrong frame. The full
// cross-product (outer 2 shots x inner 2 shots = 4 inner runs, plus 2
// outer-shot record(a) calls) must materialize, six records total.

rtg.effect @e_xshot : () -> index
func.func @record(%arg0: index) -> () { return }

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @multi_shot_reperform_same_effect
rtg.test @multi_shot_reperform_same_effect() {
  rtg.with_handlers {
    handle @e_xshot(%k: !rtg.continuation<index>) {
      %c1 = index.constant 1
      rtg.resume %k, %c1 : !rtg.continuation<index>, index
      %c2 = index.constant 2
      rtg.resume %k, %c2 : !rtg.continuation<index>, index
      rtg.yield
    }
    do {
      %a = rtg.perform @e_xshot() : () -> index
      func.call @record(%a) : (index) -> ()
      %b = rtg.perform @e_xshot() : () -> index
      func.call @record(%b) : (index) -> ()
      rtg.yield
    }
  }
}
// All effect ops elaborated. The two distinct constants are CSE'd; the
// cross-product order is: outer shot1 (a=1) -> inner b=1, b=2; outer shot2
// (a=2) -> inner b=1, b=2. Six records total, in this exact sequence.
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK: [[C1:%.+]] = rtg.constant 1 : index
// CHECK: func.call @record([[C1]])
// CHECK: func.call @record([[C1]])
// CHECK: [[C2:%.+]] = rtg.constant 2 : index
// CHECK: func.call @record([[C2]])
// CHECK: func.call @record([[C2]])
// CHECK: func.call @record([[C1]])
// CHECK: func.call @record([[C2]])

// -----

// Test: a handler region installs a *new* with_handlers scope inside its own
// body and performs an effect handled by that nested scope. This exercises
// installation of fresh handler frames during handler execution (distinct
// from the prior tests, which only nest with_handlers in the outer `do`
// block). The handler stack must accept the new frame, dispatch to it for
// the locally-performed effect, then pop it before resuming the outer
// continuation.

rtg.effect @outer3 : () -> ()
rtg.effect @local : () -> index
func.func @sink3(%arg0: index) -> () { return }

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @handler_installs_inner_handler
rtg.test @handler_installs_inner_handler() {
  rtg.with_handlers {
    handle @outer3(%k: !rtg.continuation<none>) {
      // Install a fresh handler scope inside this handler body.
      rtg.with_handlers {
        handle @local(%lk: !rtg.continuation<index>) {
          %c9 = index.constant 9
          rtg.resume %lk, %c9 : !rtg.continuation<index>, index
          rtg.yield
        }
        do {
          // Performed inside the handler region; must dispatch to the
          // freshly-installed @local frame, not escape outward.
          %v = rtg.perform @local() : () -> index
          func.call @sink3(%v) : (index) -> ()
          rtg.yield
        }
      }
      rtg.resume %k : !rtg.continuation<none>
      rtg.yield
    }
    do {
      rtg.perform @outer3() : () -> none
      rtg.yield
    }
  }
}
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK: [[C9:%.+]] = rtg.constant 9 : index
// CHECK: func.call @sink3([[C9]])

// -----

// Test: a handler that omits rtg.resume aborts the continuation, modeling an
// "exception"-like effect. Body ops following the perform are skipped, but
// ops outside the with_handlers still execute. This exercises the case where
// the continuation is dropped rather than invoked.

rtg.effect @abort_e : () -> ()
func.func @before_perform() -> () { return }
func.func @after_perform_skipped() -> () { return }
func.func @after_with_handlers() -> () { return }

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @handler_no_resume_aborts
rtg.test @handler_no_resume_aborts() {
  rtg.with_handlers {
    handle @abort_e(%k: !rtg.continuation<none>) {
      // Intentionally no rtg.resume - drop the continuation.
      rtg.yield
    }
    do {
      func.call @before_perform() : () -> ()
      rtg.perform @abort_e() : () -> none
      // Unreachable: handler dropped %k.
      func.call @after_perform_skipped() : () -> ()
      rtg.yield
    }
  }
  // After abort, control returns to the enclosing scope.
  func.call @after_with_handlers() : () -> ()
}
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK-NOT: func.call @after_perform_skipped
// CHECK: func.call @before_perform
// CHECK: func.call @after_with_handlers

// -----

// Error test: rtg.perform without an installed handler must fail at elaboration
// time with a clear diagnostic.

rtg.effect @orphan : () -> ()

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

rtg.test @unhandled_perform() {
  // expected-error @below {{no handler for effect @orphan}}
  rtg.perform @orphan() : () -> none
  rtg.yield
}

// -----

// Test: multi-shot handler that creates a label_unique_decl on each shot
// produces N distinct labels (one per resume), not a shared identity.
// UniqueLabelStorage is created via create<> (not internalize), so each
// visit of the same LabelUniqueDeclOp yields a fresh pointer.

rtg.effect @shot_lbl : () -> ()

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @multishot_distinct_labels
rtg.test @multishot_distinct_labels() {
  %pfx = rtg.constant "item" : !rtg.string
  rtg.with_handlers {
    handle @shot_lbl(%k: !rtg.continuation<none>) {
      %lbl = rtg.label_unique_decl %pfx
      rtg.label local %lbl
      rtg.resume %k : !rtg.continuation<none>
      rtg.yield
    }
    do {
      rtg.perform @shot_lbl() : () -> none
      rtg.perform @shot_lbl() : () -> none
      rtg.yield
    }
  }
}
// Two performs → handler runs twice → two distinct label_unique_decl ops.
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK: [[PFX:%.+]] = rtg.constant "item"
// CHECK: [[L1:%.+]] = rtg.label_unique_decl [[PFX]]
// CHECK: rtg.label local [[L1]]
// CHECK: [[L2:%.+]] = rtg.label_unique_decl [[PFX]]
// CHECK: rtg.label local [[L2]]

// -----

// Test: scf.for in handler body drives multi-shot fan-out. Handler iterates
// [0,3) and resumes the continuation once per iteration value.

rtg.effect @pick_for : () -> index
func.func @record_for(%v: index) -> () { return }

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @multishot_via_scffor
rtg.test @multishot_via_scffor() {
  %lb   = index.constant 0
  %ub   = index.constant 3
  %step = index.constant 1
  rtg.with_handlers {
    handle @pick_for(%k: !rtg.continuation<index>) {
      scf.for %i = %lb to %ub step %step {
        rtg.resume %k, %i : !rtg.continuation<index>, index
      }
      rtg.yield
    }
    do {
      %v = rtg.perform @pick_for() : () -> index
      func.call @record_for(%v) : (index) -> ()
      rtg.yield
    }
  }
}
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK: [[C0:%.+]] = rtg.constant 0 : index
// CHECK: func.call @record_for([[C0]])
// CHECK: [[C1:%.+]] = rtg.constant 1 : index
// CHECK: func.call @record_for([[C1]])
// CHECK: [[C2:%.+]] = rtg.constant 2 : index
// CHECK: func.call @record_for([[C2]])

// -----

// Test: cross-handler resume. Inner handler for @inner_xh resumes the OUTER
// continuation %k_outer, bypassing its own %k_inner ("throw" semantics).
// The inner handler's continuation is implicitly dropped; the outer body
// receives value 99.

rtg.effect @outer_xh : () -> index
rtg.effect @inner_xh : () -> ()
func.func @record_xh(%v: index) -> () { return }

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @cross_handler_resume
rtg.test @cross_handler_resume() {
  rtg.with_handlers {
    handle @outer_xh(%k_outer: !rtg.continuation<index>) {
      rtg.with_handlers {
        handle @inner_xh(%k_inner: !rtg.continuation<none>) {
          // Resumes outer continuation directly; k_inner is dropped.
          %c99 = index.constant 99
          rtg.resume %k_outer, %c99 : !rtg.continuation<index>, index
          rtg.yield
        }
        do {
          rtg.perform @inner_xh() : () -> none
          rtg.yield
        }
      }
      rtg.yield
    }
    do {
      %v = rtg.perform @outer_xh() : () -> index
      func.call @record_xh(%v) : (index) -> ()
      rtg.yield
    }
  }
}
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK: [[C99:%.+]] = rtg.constant 99 : index
// CHECK: func.call @record_xh([[C99]])

// -----

// Test: deep-capture / shallow-resume. Outer perform captures %k_deep with
// only the outer frame in its snapshot. Inner handler for @shallow_dr resumes
// %k_deep (providing value 5), then resumes its own %k_inner to complete.
// The outer body receives 5 from the outer perform.

rtg.effect @deep_dr   : () -> index
rtg.effect @shallow_dr : () -> ()
func.func @record_dr(%v: index) -> () { return }

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @deep_capture_shallow_resume
rtg.test @deep_capture_shallow_resume() {
  rtg.with_handlers {
    handle @deep_dr(%k_deep: !rtg.continuation<index>) {
      rtg.with_handlers {
        handle @shallow_dr(%k_sh: !rtg.continuation<none>) {
          %c5 = index.constant 5
          rtg.resume %k_deep, %c5 : !rtg.continuation<index>, index
          rtg.resume %k_sh : !rtg.continuation<none>
          rtg.yield
        }
        do {
          rtg.perform @shallow_dr() : () -> none
          rtg.yield
        }
      }
      rtg.yield
    }
    do {
      %v = rtg.perform @deep_dr() : () -> index
      func.call @record_dr(%v) : (index) -> ()
      rtg.yield
    }
  }
}
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK: [[C5:%.+]] = rtg.constant 5 : index
// CHECK: func.call @record_dr([[C5]])

// -----

// Error test: duplicate handle clause for the same effect in one
// rtg.with_handlers must be rejected by the verifier.

rtg.effect @dup_e : () -> index

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// expected-error @below {{duplicate handler for effect 'dup_e'}}
rtg.with_handlers {
  handle @dup_e(%k: !rtg.continuation<index>) {
    %c1 = index.constant 1
    rtg.resume %k, %c1 : !rtg.continuation<index>, index
    rtg.yield
  }
  handle @dup_e(%k: !rtg.continuation<index>) {
    %c2 = index.constant 2
    rtg.resume %k, %c2 : !rtg.continuation<index>, index
    rtg.yield
  }
  do {
    rtg.yield
  }
}

