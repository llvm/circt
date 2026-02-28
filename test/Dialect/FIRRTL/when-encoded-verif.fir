; RUN: circt-translate -import-firrtl -verify-diagnostics %s
FIRRTL version 4.0.0
circuit WhenEncodedVerification:
  ; expected-error @below {{module contains 23 printf-encoded verification operation(s), which are no longer supported.}}
  ; expected-note @below {{For more information, see https://github.com/llvm/circt/issues/6970}}
  public module WhenEncodedVerification:
    input clock: Clock
    input cond: UInt<1>
    input enable: UInt<1>
    input not_reset: UInt<1>
    input value: UInt<42>

    ; expected-note @+3 {{example printf here, this is now just a printf and nothing more}}
    ; rocket-chip properties
    when cond:
      printf(clock, enable, "assert:foo 0", value)

    when cond:
      printf(clock, enable, "assume:foo 1", value)

    when cond:
      printf(clock, enable, "cover:foo 2", value)

    when cond:
      printf(clock, enable, "assert:foo_0:", value)

    when cond:
      printf(clock, enable, "assume:foo_1:", value)

    when cond:
      printf(clock, enable, "cover:foo_2:", value)

    when cond:
      printf(clock, enable, "assert:custom label 0:foo 3", value)

    when cond:
      printf(clock, enable, "assume:custom label 1:foo 4", value)

    when cond:
      printf(clock, enable, "cover:custom label 2:foo 5", value)

    ; Optional `stop` with same clock and condition should be removed.
    when cond:
      printf(clock, enable, "assert:without_stop")
      stop(clock, enable, 1)

    when cond:
      printf(clock, enable, "assert:foo 6, %d", value, value)

    ; AssertNotX -- usually `cond` only checks for not-in-reset, and `enable` is
    ; just set to 1; the actual check `^value !== 'x` is implicit.
    when cond:
      printf(clock, enable, "assertNotX:%d:value must not be X!", value)

    ; Chisel built-in assertions
    when cond:
      printf(clock, enable, "Assertion failed with value %d", value)
      stop(clock, enable, 1)

    when cond:
      printf(clock, enable, "Assertion failed: some message with value %d", value)
      stop(clock, enable, 1)

    ; Verification Library Assertions

    ; Predicate modifier `noMod`
    when cond:
      printf(clock, enable, "Assertion failed: [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"},{\"type\":\"formalOnly\"}],\"labelExts\":[\"label\",\"magic\"],\"format\":{\"type\":\"sva\"},\"baseMsg\":\"Hello Assert\"}", value)
      stop(clock, enable, 1)

    ; Predicate modifier `trueOrIsX`
    when cond:
      printf(clock, enable, "Assertion failed: [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"trueOrIsX\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"},{\"type\":\"formalOnly\"}],\"labelExts\":[\"label\",\"magic\"],\"format\":{\"type\":\"sva\"},\"baseMsg\":\"Hello Assert\"}", value)
      stop(clock, enable, 1)

    ; Verification Library Assumptions

    ; Predicate modifier `noMod`
    when cond:
      printf(clock, enable, "Assumption failed: [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"},{\"type\":\"formalOnly\"}],\"labelExts\":[\"label\",\"voodoo\"],\"baseMsg\":\"Hello Assume\"}", value)
      stop(clock, enable, 1)

    ; Predicate modifier `trueOrIsX`
    when cond:
      printf(clock, enable, "Assumption failed: [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"trueOrIsX\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"},{\"type\":\"formalOnly\"}],\"labelExts\":[\"label\",\"voodoo\"],\"baseMsg\":\"Hello Assume\"}", value)
      stop(clock, enable, 1)

    ; New flavor of when-encoded verification that also includes an assert
    assert(clock, cond, enable, "hello")
    node not_cond = eq(cond, UInt<1>(0))
    when not_cond:
      printf(clock, enable, "Assertion failed: hello")

    when not_reset:
      assert(clock, cond, enable, "hello outside reset")
      node not_cond2 = eq(cond, UInt<1>(0))
      when not_cond2:
        printf(clock, enable, "Assertion failed: hello outside reset")

    ; Check that the above doesn't error if the assert is a double user of the
    ; condition.
    when not_reset:
      assert(clock, UInt<1>(1), UInt<1>(1), "double user assert")
      node not_cond3 = eq(UInt<1>(1), UInt<1>(0))
      when not_cond3:
        printf(clock, UInt<1>(1), "Assertion failed: double user assert")

    when cond :
        printf(clock, not(enable), "assert: bar")

    ; Verification Library Covers

    ; Predicate modifier `noMod`
    when not(cond):
      printf(clock, enable, "Assertion failed: [verif-library-cover]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"},{\"type\":\"formalOnly\"}],\"labelExts\":[\"cover\",\"label\"],\"format\":{\"type\":\"sva\"},\"baseMsg\":\"cover hello world\"}", value)
    assert(clock, cond, enable, "")
