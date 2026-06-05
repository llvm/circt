// RUN: circt-opt %s --arc-lower-coroutines --split-input-file --verify-diagnostics

// expected-error @below {{recursive coroutines are not supported}}
arc.coroutine.define @Recursive() {
  %state = arc.coroutine.undefined_state : !arc.coroutine_state<@Recursive>
  arc.coroutine.yield ^bb1(%state : !arc.coroutine_state<@Recursive>)
^bb1(%s: !arc.coroutine_state<@Recursive>):
  arc.coroutine.return
}

// -----

// expected-error @below {{recursive coroutines are not supported}}
arc.coroutine.define @MutualA() {
  %state = arc.coroutine.undefined_state : !arc.coroutine_state<@MutualB>
  arc.coroutine.yield ^bb1(%state : !arc.coroutine_state<@MutualB>)
^bb1(%s: !arc.coroutine_state<@MutualB>):
  arc.coroutine.return
}

arc.coroutine.define @MutualB() {
  %state = arc.coroutine.undefined_state : !arc.coroutine_state<@MutualA>
  arc.coroutine.yield ^bb1(%state : !arc.coroutine_state<@MutualA>)
^bb1(%s: !arc.coroutine_state<@MutualA>):
  arc.coroutine.return
}

// -----

arc.coroutine.define @WithWakeup() -> i64 {
  %c0_i64 = hw.constant 0 : i64
  arc.coroutine.return %c0_i64 : i64
}

hw.module @Module() {
  // expected-error @below {{must be lowered before LowerCoroutines}}
  arc.coroutine.instance @WithWakeup() : () -> ()
}

// -----

// expected-error @below {{coroutine type references unknown coroutine @DoesNotExist}}
module {
  func.func private @Unknown(!arc.coroutine_state<@DoesNotExist>)
}
