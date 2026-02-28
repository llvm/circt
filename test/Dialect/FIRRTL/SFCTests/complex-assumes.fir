; RUN: circt-translate -import-firrtl -verify-diagnostics %s
; Tests extracted from:
; - test/scala/firrtl/extractverif/ExtractAssumesSpec.scala

FIRRTL version 4.0.0
circuit Foo:
  ; expected-error @below {{module contains 8 printf-encoded verification operation(s), which are no longer supported.}}
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
    input enable : UInt<1>
    input other : UInt<1>
    input sum : UInt<42>

    ; assume with predicate only
    ; expected-note @+2 {{example printf here, this is now just a printf and nothing more}}
    when not(or(predicate1, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"baseMsg\":\"Assumption does not hold (verification library): \"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; assume with message
    when not(or(predicate2, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"baseMsg\":\"Assumption does not hold (verification library): sum =/= 1.U\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; assume with when
    when other :
      when not(or(predicate3, asUInt(reset))) :
        printf(clock, enable, "foo [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"baseMsg\":\"Assumption does not hold (verification library): assume with when\"}</extraction-summary> bar")
        stop(clock, enable, 1)

    ; assume with message with arguments
    when not(or(predicate4, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"baseMsg\":\"Assumption does not hold (verification library): expected sum === 2.U but got %d\"}</extraction-summary> bar", sum)
      stop(clock, enable, 1)

    ; assume with custom label
    when not(or(predicate5, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"labelExts\":[\"hello\",\"world\"],\"baseMsg\":\"Assumption does not hold (verification library): Custom label example\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; assume with predicate option for X-passing
    when not(predicate6) :
      printf(clock, enable, "foo [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"trueOrIsX\"},\"baseMsg\":\"Assumption does not hold (verification library): X-passing assume example\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; assume with toggle option e.g. UNROnly
    when not(or(predicate7, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"}],\"baseMsg\":\"Assumption does not hold (verification library): Conditional compilation example for UNR-only assume\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; assume with multiple toggle options
    when not(or(predicate8, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"formalOnly\"},{\"type\":\"unrOnly\"}],\"baseMsg\":\"Assumption does not hold (verification library): Conditional compilation example for UNR-only and formal-only assume\"}</extraction-summary> bar")
      stop(clock, enable, 1)
