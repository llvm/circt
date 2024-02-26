; RUN: circt-translate -import-firrtl -verify-diagnostics -split-input-file %s | circt-opt | FileCheck %s
circuit WhenEncodedVerification:
  ; CHECK-LABEL: @WhenEncodedVerification
  ; expected-warning @below {{module contains 23 printf-encoded verification operation(s), which are deprecated and will be removed in the future}}
  module WhenEncodedVerification:
    input clock: Clock
    input cond: UInt<1>
    input enable: UInt<1>
    input not_reset: UInt<1>
    input value: UInt<42>

    ; expected-note @+3 {{example printf here, will just be a printf in the future}}
    ; rocket-chip properties
    when cond:
      printf(clock, enable, "assert:foo 0", value)
    ; CHECK: [[TMP:%.+]] = firrtl.not %cond
    ; CHECK-NEXT: firrtl.assert %clock, [[TMP]], %enable, "foo 0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = true}

    when cond:
      printf(clock, enable, "assume:foo 1", value)
    ; CHECK: [[TMP:%.+]] = firrtl.not %cond
    ; CHECK-NEXT: firrtl.assume %clock, [[TMP]], %enable, "foo 1" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = true}

    when cond:
      printf(clock, enable, "cover:foo 2", value)
    ; CHECK-NEXT: firrtl.cover %clock, %cond, %enable, "foo 2" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = true}

    when cond:
      printf(clock, enable, "assert:foo_0:", value)
    ; CHECK: [[TMP:%.+]] = firrtl.not %cond
    ; CHECK-NEXT: firrtl.assert %clock, [[TMP]], %enable, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = true, name = "foo_0"}

    when cond:
      printf(clock, enable, "assume:foo_1:", value)
    ; CHECK: [[TMP:%.+]] = firrtl.not %cond
    ; CHECK-NEXT: firrtl.assume %clock, [[TMP]], %enable, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = true, name = "foo_1"}

    when cond:
      printf(clock, enable, "cover:foo_2:", value)
    ; CHECK-NEXT: firrtl.cover %clock, %cond, %enable, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = true, name = "foo_2"}

    when cond:
      printf(clock, enable, "assert:custom label 0:foo 3", value)
    ; CHECK: [[TMP:%.+]] = firrtl.not %cond
    ; CHECK-NEXT: firrtl.assert %clock, [[TMP]], %enable, "foo 3" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = true, name = "custom label 0"}

    when cond:
      printf(clock, enable, "assume:custom label 1:foo 4", value)
    ; CHECK: [[TMP:%.+]] = firrtl.not %cond
    ; CHECK-NEXT: firrtl.assume %clock, [[TMP]], %enable, "foo 4" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = true, name = "custom label 1"}

    when cond:
      printf(clock, enable, "cover:custom label 2:foo 5", value)
    ; CHECK-NEXT: firrtl.cover %clock, %cond, %enable, "foo 5" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = true, name = "custom label 2"}

    ; Optional `stop` with same clock and condition should be removed.
    when cond:
      printf(clock, enable, "assert:without_stop")
      stop(clock, enable, 1)
    ; CHECK: firrtl.assert %clock, {{%.+}}, %enable, "without_stop" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    ; CHECK-NOT: firrtl.stop

    when cond:
      ; expected-warning @+1 {{printf-encoded assertion has format string arguments which may cause lint warnings}}
      printf(clock, enable, "assert:foo 6, %d", value, value)
    ; CHECK: firrtl.assert {{.+}} "foo 6, %d"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42>

    ; AssertNotX -- usually `cond` only checks for not-in-reset, and `enable` is
    ; just set to 1; the actual check `^value !== 'x` is implicit.
    when cond:
      printf(clock, enable, "assertNotX:%d:value must not be X!", value)
    ; CHECK: [[TMP1:%.+]] = firrtl.not %cond
    ; CHECK: [[TMP2:%.+]] = firrtl.xorr %value
    ; CHECK: [[TMP3:%.+]] = firrtl.int.isX
    ; CHECK: [[TMP4:%.+]] = firrtl.not
    ; CHECK: [[TMP5:%.+]] = firrtl.or [[TMP1]], [[TMP4]]
    ; CHECK: firrtl.assert %clock, [[TMP5]], %enable, "value must not be X!" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    ; CHECK-SAME: name = "notX"

    ; Chisel built-in assertions
    when cond:
      printf(clock, enable, "Assertion failed with value %d", value)
      stop(clock, enable, 1)
    ; CHECK: [[TMP:%.+]] = firrtl.not %cond
    ; CHECK-NEXT: firrtl.assert %clock, [[TMP]], %enable, "Assertion failed with value %d"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42>
    ; CHECK-SAME: format = "ifElseFatal"
    ; CHECK-SAME: isConcurrent = true
    ; CHECK-SAME: name = "chisel3_builtin"

    when cond:
      printf(clock, enable, "Assertion failed: some message with value %d", value)
      stop(clock, enable, 1)
    ; CHECK: [[TMP:%.+]] = firrtl.not %cond
    ; CHECK-NEXT: firrtl.assert %clock, [[TMP]], %enable, "Assertion failed: some message with value %d"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42>
    ; CHECK-SAME: format = "ifElseFatal"
    ; CHECK-SAME: isConcurrent = true
    ; CHECK-SAME: name = "chisel3_builtin"

    ; Verification Library Assertions

    ; Predicate modifier `noMod`
    when cond:
      printf(clock, enable, "Assertion failed: [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"},{\"type\":\"formalOnly\"}],\"labelExts\":[\"label\",\"magic\"],\"format\":{\"type\":\"sva\"},\"baseMsg\":\"Hello Assert\"}", value)
      stop(clock, enable, 1)
    ; CHECK: [[TMP:%.+]] = firrtl.not %cond
    ; CHECK-NEXT: firrtl.assert %clock, [[TMP]], %enable, "Hello Assert"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42>
    ; CHECK-SAME: format = "sva"
    ; CHECK-SAME: guards = ["USE_UNR_ONLY_CONSTRAINTS", "USE_FORMAL_ONLY_CONSTRAINTS"]
    ; CHECK-SAME: name = "verif_library_label_magic"

    ; Predicate modifier `trueOrIsX`
    when cond:
      printf(clock, enable, "Assertion failed: [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"trueOrIsX\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"},{\"type\":\"formalOnly\"}],\"labelExts\":[\"label\",\"magic\"],\"format\":{\"type\":\"sva\"},\"baseMsg\":\"Hello Assert\"}", value)
      stop(clock, enable, 1)
    ; CHECK: [[CONDINV:%.+]] = firrtl.not %cond
    ; CHECK: [[TMP1:%.+]] = firrtl.xorr [[CONDINV]]
    ; CHECK-NEXT: [[TMP2:%.+]] = firrtl.int.isX [[TMP1]]
    ; CHECK-NEXT: [[TMP:%.+]] = firrtl.or [[CONDINV]], [[TMP2]]
    ; CHECK-NEXT: firrtl.assert %clock, [[TMP]], %enable, "Hello Assert"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42>
    ; CHECK-SAME: format = "sva"
    ; CHECK-SAME: guards = ["USE_UNR_ONLY_CONSTRAINTS", "USE_FORMAL_ONLY_CONSTRAINTS"]
    ; CHECK-SAME: name = "verif_library_label_magic"

    ; Verification Library Assumptions

    ; Predicate modifier `noMod`
    when cond:
      printf(clock, enable, "Assumption failed: [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"},{\"type\":\"formalOnly\"}],\"labelExts\":[\"label\",\"voodoo\"],\"baseMsg\":\"Hello Assume\"}", value)
      stop(clock, enable, 1)
    ; CHECK: [[TMP:%.+]] = firrtl.not %cond
    ; CHECK-NEXT: firrtl.assume %clock, [[TMP]], %enable, "Hello Assume"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42>
    ; CHECK-SAME: guards = ["USE_UNR_ONLY_CONSTRAINTS", "USE_FORMAL_ONLY_CONSTRAINTS"]
    ; CHECK-SAME: name = "verif_library_label_voodoo"

    ; Predicate modifier `trueOrIsX`
    when cond:
      printf(clock, enable, "Assumption failed: [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"trueOrIsX\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"},{\"type\":\"formalOnly\"}],\"labelExts\":[\"label\",\"voodoo\"],\"baseMsg\":\"Hello Assume\"}", value)
      stop(clock, enable, 1)
    ; CHECK: [[CONDINV:%.+]] = firrtl.not %cond
    ; CHECK: [[TMP1:%.+]] = firrtl.xorr [[CONDINV]]
    ; CHECK-NEXT: [[TMP2:%.+]] = firrtl.int.isX [[TMP1]]
    ; CHECK-NEXT: [[TMP:%.+]] = firrtl.or [[CONDINV]], [[TMP2]]
    ; CHECK-NEXT: firrtl.assume %clock, [[TMP]], %enable, "Hello Assume"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42>
    ; CHECK-SAME: guards = ["USE_UNR_ONLY_CONSTRAINTS", "USE_FORMAL_ONLY_CONSTRAINTS"]
    ; CHECK-SAME: name = "verif_library_label_voodoo"

    ; New flavor of when-encoded verification that also includes an assert
    assert(clock, cond, enable, "hello")
    node not_cond = eq(cond, UInt<1>(0))
    when not_cond:
      printf(clock, enable, "Assertion failed: hello")
    ; CHECK-NOT: firrtl.assert %clock, %cond, %enable, "hello" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    ; CHECK: [[TMP:%.+]] = firrtl.not %not_cond
    ; CHECK-NEXT: firrtl.assert %clock, [[TMP]], %enable, "Assertion failed: hello" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    ; CHECK-SAME: format = "ifElseFatal"
    ; CHECK-SAME: name = "chisel3_builtin"

    when not_reset:
      assert(clock, cond, enable, "hello outside reset")
      node not_cond2 = eq(cond, UInt<1>(0))
      when not_cond2:
        printf(clock, enable, "Assertion failed: hello outside reset")
    ; CHECK: firrtl.when %not_reset : !firrtl.uint<1> {
    ; CHECK-NOT: firrtl.assert %clock, %cond, %enable, "hello outside reset" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    ; CHECK: [[TMP:%.+]] = firrtl.not %not_cond2
    ; CHECK-NEXT: firrtl.assert %clock, [[TMP]], %enable, "Assertion failed: hello outside reset" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    ; CHECK-SAME: format = "ifElseFatal"
    ; CHECK-SAME: name = "chisel3_builtin"
    ; CHECK: }

    ; Check that the above doesn't error if the assert is a double user of the
    ; condition.
    when not_reset:
      assert(clock, UInt<1>(1), UInt<1>(1), "double user assert")
      node not_cond3 = eq(UInt<1>(1), UInt<1>(0))
      when not_cond3:
        printf(clock, UInt<1>(1), "Assertion failed: double user assert")
    ; CHECK-NOT: firrtl.assert %clock, {{.+}} "double user assert" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>

    when cond :
        printf(clock, not(enable), "assert: bar")
    ; CHECK: [[NOT_ENABLE:%.+]] = firrtl.not %enable
    ; CHECK-NEXT: [[NOT_COND:%.+]] = firrtl.not %cond
    ; CHECK-NEXT: firrtl.assert %clock, [[NOT_COND]], [[NOT_ENABLE]], " bar" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = true}

    ; Verification Library Covers

    ; Predicate modifier `noMod`
    when not(cond):
      printf(clock, enable, "Assertion failed: [verif-library-cover]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"},{\"type\":\"formalOnly\"}],\"labelExts\":[\"cover\",\"label\"],\"format\":{\"type\":\"sva\"},\"baseMsg\":\"cover hello world\"}", value)
    assert(clock, cond, enable, "")
    ; CHECK: [[TMP_INV:%.+]] = firrtl.not %cond
    ; CHECK: [[TMP:%.+]] = firrtl.not [[TMP_INV]]
    ; CHECK-NOT: firrtl.assert %clock, [[TMP]], %enable, "cover hello world"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42>
    ; CHECK: firrtl.cover %clock, [[TMP]], %enable, "cover hello world"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42>
    ; CHECK-SAME: guards = ["USE_UNR_ONLY_CONSTRAINTS", "USE_FORMAL_ONLY_CONSTRAINTS"]
    ; CHECK-SAME: name = "verif_library_cover_label"


