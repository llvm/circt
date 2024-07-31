; RUN: circt-translate -import-firrtl -verify-diagnostics %s

; Tests extracted from:
; - test/scala/firrtl/extractverif/ExtractAssertsSpec.scala

FIRRTL version 4.0.0
circuit Foo:
  ; expected-error @below {{module contains 10 printf-encoded verification operation(s), which are no longer supported.}}
  ; expected-note @below {{For more information, see https://github.com/llvm/circt/issues/6970}}
  public module Foo:
    input clock : Clock
    input reset : AsyncReset
    input predicate1 : UInt<1>
    input predicate2 : UInt<1>
    input predicate3 : UInt<1>
    input predicate4 : UInt<1>
    input predicate5 : UInt<1>
    input predicate6 : UInt<1>
    input predicate7 : UInt<1>
    input predicate8 : UInt<1>
    input predicate9 : UInt<1>
    input predicate10 : UInt<1>
    input enable : UInt<1>
    input other : UInt<1>
    input sum : UInt<42>


    ; assert with predicate only

    ; expected-note @+2 {{example printf here, this is now just a printf and nothing more}}
    when not(or(predicate1, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"format\":{\"type\":\"ifElseFatal\"},\"baseMsg\":\"Assertion failed (verification library): \"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; assert with message
    when not(or(predicate2, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"format\":{\"type\":\"ifElseFatal\"},\"baseMsg\":\"Assertion failed (verification library): sum =/= 1.U\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; assert with when
    when other :
      when not(or(predicate3, asUInt(reset))) :
        printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"format\":{\"type\":\"ifElseFatal\"},\"baseMsg\":\"Assertion failed (verification library): Assert with when\"}</extraction-summary> bar")
        stop(clock, enable, 1)

    ; assert with message with arguments
    when not(or(predicate4, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"format\":{\"type\":\"ifElseFatal\"},\"baseMsg\":\"Assertion failed (verification library): expected sum === 2.U but got %d\"}</extraction-summary> bar", sum)
      stop(clock, enable, 1)

    ; assert emitted as SVA
    when not(or(predicate5, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"format\":{\"type\":\"sva\"},\"baseMsg\":\"Assertion failed (verification library): SVA assert example, sum was %d\"}</extraction-summary> bar", sum)
      stop(clock, enable, 1)

    ; assert with custom label
    when not(or(predicate6, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"format\":{\"type\":\"sva\"},\"labelExts\":[\"hello\",\"world\"],\"baseMsg\":\"Assertion failed (verification library): Custom label example\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; assert with predicate option for X-passing
    when not(predicate7) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"trueOrIsX\"},\"format\":{\"type\":\"sva\"},\"baseMsg\":\"Assertion failed (verification library): X-passing assert example\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; assert with toggle option e.g. UNROnly
    when not(or(predicate8, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"}],\"format\":{\"type\":\"sva\"},\"baseMsg\":\"Assertion failed (verification library): Conditional compilation example for UNR-only assert\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; if-else-fatal style assert with conditional compilation toggles
    when not(or(predicate9, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"}],\"format\":{\"type\":\"ifElseFatal\"},\"baseMsg\":\"Assertion failed (verification library): Conditional compilation example with if-else-fatal style assert\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; assert with multiple toggle options
    when not(or(predicate10, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"formalOnly\"},{\"type\":\"unrOnly\"}],\"format\":{\"type\":\"sva\"},\"baseMsg\":\"Assertion failed (verification library): Conditional compilation example for UNR-only and formal-only assert\"}</extraction-summary> bar")
      stop(clock, enable, 1)
