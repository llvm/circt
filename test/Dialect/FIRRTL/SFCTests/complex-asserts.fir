; RUN: firtool --verify-diagnostics --verilog %s --lowering-options=emittedLineLength=200 --add-companion-assume | FileCheck %s
; RUN: firtool --verify-diagnostics --verilog --verification-flavor=sva %s --lowering-options=emittedLineLength=200 --add-companion-assume | FileCheck %s --check-prefix=SVA
; Tests extracted from:
; - test/scala/firrtl/extractverif/ExtractAssertsSpec.scala

circuit Foo:
  module Foo:
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

    ; CHECK:      wire [[GEN:.+]] = predicate1 | reset;
    ; CHECK-NEXT: wire [[GEN_0:.+]] = predicate2 | reset;
    ; CHECK-NEXT: wire [[GEN_1:.+]] = predicate3 | reset;
    ; CHECK-NEXT: wire [[GEN_2:.+]] = other & enable;
    ; CHECK-NEXT: wire [[GEN_3:.+]] = predicate4 | reset;

    ; CHECK: `ifndef SYNTHESIS
    ; CHECK-NEXT: always @(posedge clock) begin
    ; assert with predicate only
    ; CHECK-NEXT: if (enable & ~[[GEN]]) begin
    ; CHECK-NEXT:   if (`ASSERT_VERBOSE_COND_)
    ; CHECK-NEXT:     $error("Assertion failed (verification library): ");
    ; CHECK-NEXT:   if (`STOP_COND_)
    ; CHECK-NEXT:     $fatal;
    ; CHECK-NEXT: end

    ; SVA: wire _GEN = ~enable | predicate1 | reset;
    ; SVA-NEXT: assert__verif_library: assert property (@(posedge clock) _GEN) else $error("Assertion failed (verification library): ");

    when not(or(predicate1, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"format\":{\"type\":\"ifElseFatal\"},\"baseMsg\":\"Assertion failed (verification library): \"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; assert with message
    ; CHECK-NEXT: if (enable & ~[[GEN_0]]) begin
    ; CHECK-NEXT:   if (`ASSERT_VERBOSE_COND_)
    ; CHECK-NEXT:     $error("Assertion failed (verification library): sum =/= 1.U");
    ; CHECK-NEXT:   if (`STOP_COND_)
    ; CHECK-NEXT:     $fatal;
    ; CHECK-NEXT: end
    when not(or(predicate2, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"format\":{\"type\":\"ifElseFatal\"},\"baseMsg\":\"Assertion failed (verification library): sum =/= 1.U\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; assert with when
    ; CHECK-NEXT: if ([[GEN_2]] & ~[[GEN_1]]) begin
    ; CHECK-NEXT:   if (`ASSERT_VERBOSE_COND_)
    ; CHECK-NEXT:     $error("Assertion failed (verification library): Assert with when");
    ; CHECK-NEXT:   if (`STOP_COND_)
    ; CHECK-NEXT:     $fatal;
    ; CHECK-NEXT: end
    when other :
      when not(or(predicate3, asUInt(reset))) :
        printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"format\":{\"type\":\"ifElseFatal\"},\"baseMsg\":\"Assertion failed (verification library): Assert with when\"}</extraction-summary> bar")
        stop(clock, enable, 1)

    ; assert with message with arguments
    ; CHECK-NEXT: if (enable & ~[[GEN_3]]) begin
    ; CHECK-NEXT:   if (`ASSERT_VERBOSE_COND_)
    ; CHECK-NEXT:     $error("Assertion failed (verification library): expected sum === 2.U but got %d", sum);
    ; CHECK-NEXT:   if (`STOP_COND_)
    ; CHECK-NEXT:     $fatal;
    ; CHECK-NEXT: end
    when not(or(predicate4, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"format\":{\"type\":\"ifElseFatal\"},\"baseMsg\":\"Assertion failed (verification library): expected sum === 2.U but got %d\"}</extraction-summary> bar", sum)
      stop(clock, enable, 1)

    ; CHECK-NEXT: end
    ; CHECK-NEXT: `endif


    ; assert emitted as SVA
    ; CHECK: wire [[GEN_4:.+]] = ~enable | predicate5 | reset;
    ; CHECK-NEXT: assert__verif_library: assert property (@(posedge clock) [[GEN_4]])
    ; CHECK-SAME:   else $error("Assertion failed (verification library): SVA assert example, sum was %d", $sampled(sum));
    when not(or(predicate5, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"format\":{\"type\":\"sva\"},\"baseMsg\":\"Assertion failed (verification library): SVA assert example, sum was %d\"}</extraction-summary> bar", sum)
      stop(clock, enable, 1)

    ; assert with custom label
    ; CHECK: wire [[GEN_5:.+]] = ~enable | predicate6 | reset;
    ; CHECK-NEXT: assert__verif_library_hello_world: assert property (@(posedge clock) [[GEN_5]])
    ; CHECK-SAME:   else $error("Assertion failed (verification library): Custom label example");
    when not(or(predicate6, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"format\":{\"type\":\"sva\"},\"labelExts\":[\"hello\",\"world\"],\"baseMsg\":\"Assertion failed (verification library): Custom label example\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; assert with predicate option for X-passing
    ; CHECK-NEXT: wire [[GEN_6:.+]] = ~enable | predicate7 | predicate7 === 1'bx;
    ; CHECK-NEXT: assert__verif_library_0: assert property (@(posedge clock) [[GEN_6]])
    ; CHECK-SAME:   else $error("Assertion failed (verification library): X-passing assert example");
    when not(predicate7) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"trueOrIsX\"},\"format\":{\"type\":\"sva\"},\"baseMsg\":\"Assertion failed (verification library): X-passing assert example\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; CHECK-NEXT: wire [[GEN_7:.+]] = ~enable | predicate8 | reset;
    ; CHECK-NEXT: wire [[GEN_8:.+]] = predicate9 | reset;
    ; assert with toggle option e.g. UNROnly
    ; CHECK-NEXT: `ifdef USE_UNR_ONLY_CONSTRAINTS
    ; CHECK-NEXT:   assert__verif_library_1: assert property (@(posedge clock) [[GEN_7]])
    ; CHECK-SAME:     else $error("Assertion failed (verification library): Conditional compilation example for UNR-only assert");
    when not(or(predicate8, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"}],\"format\":{\"type\":\"sva\"},\"baseMsg\":\"Assertion failed (verification library): Conditional compilation example for UNR-only assert\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; if-else-fatal style assert with conditional compilation toggles
    ; CHECK-NEXT:  `ifndef SYNTHESIS
    ; CHECK-NEXT:    always @(posedge clock) begin
    ; CHECK-NEXT:     if (enable & ~[[GEN_8]]) begin
    ; CHECK-NEXT:       if (`ASSERT_VERBOSE_COND_)
    ; CHECK-NEXT:         $error("Assertion failed (verification library): Conditional compilation example with if-else-fatal style assert");
    ; CHECK-NEXT:       if (`STOP_COND_)
    ; CHECK-NEXT:         $fatal;
    ; CHECK-NEXT:     end
    ; CHECK-NEXT:    end
    ; CHECK-NEXT:  `endif
    ; CHECK-NEXT: `endif
    when not(or(predicate9, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"}],\"format\":{\"type\":\"ifElseFatal\"},\"baseMsg\":\"Assertion failed (verification library): Conditional compilation example with if-else-fatal style assert\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; assert with multiple toggle options
    ; CHECK-NEXT:  wire [[GEN_9:.+]] = predicate10 | reset;
    ; CHECK-NEXT: `ifdef USE_FORMAL_ONLY_CONSTRAINTS
    ; CHECK-NEXT:   `ifdef USE_UNR_ONLY_CONSTRAINTS
    ; CHECK-NEXT:     assert__verif_library_2: 
    ; CHECK-NEXT:       assert property (@(posedge clock) ~enable | [[GEN_9]])
    ; CHECK-SAME:       else $error("Assertion failed (verification library): Conditional compilation example for UNR-only and formal-only assert");
    ; CHECK-NEXT:   `endif
    ; CHECK-NEXT: `endif
    when not(or(predicate10, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"formalOnly\"},{\"type\":\"unrOnly\"}],\"format\":{\"type\":\"sva\"},\"baseMsg\":\"Assertion failed (verification library): Conditional compilation example for UNR-only and formal-only assert\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; The companion assumes get bunched up in an ifdef block.
    ; CHECK-NEXT: `ifdef USE_PROPERTY_AS_CONSTRAINT
    ; CHECK-NEXT:   assume__verif_library: assume property (@(posedge clock) ~enable | [[GEN]]);
    ; CHECK-NEXT:   assume__verif_library_0: assume property (@(posedge clock) ~enable | [[GEN_0]]);
    ; CHECK-NEXT:   assume__verif_library_1: assume property (@(posedge clock) ~[[GEN_2]] | [[GEN_1]]);
    ; CHECK-NEXT:   assume__verif_library_2: assume property (@(posedge clock) ~enable | [[GEN_3]]);
    ; CHECK-NEXT:   assume__verif_library_3: assume property (@(posedge clock) [[GEN_4]]);
    ; CHECK-NEXT:   assume__verif_library_hello_world: assume property (@(posedge clock) [[GEN_5]]);
    ; CHECK-NEXT:   assume__verif_library_4: assume property (@(posedge clock) [[GEN_6]]);
    ; CHECK-NEXT:   `ifdef USE_UNR_ONLY_CONSTRAINTS
    ; CHECK-NEXT:   wire [[GEN_10:.+]] = ~enable | [[GEN_8]];
    ; CHECK-NEXT:   always @(edge [[GEN_7]])
    ; CHECK-NEXT:     assume__verif_library_5: assume([[GEN_7]]);
    ; CHECK-NEXT:   always @(edge [[GEN_10]])
    ; CHECK-NEXT:     assume__verif_library_6: assume([[GEN_10]]);
    ; CHECK-NEXT:   `endif // USE_UNR_ONLY_CONSTRAINTS
    ; CHECK-NEXT:   `ifdef USE_FORMAL_ONLY_CONSTRAINTS
    ; CHECK-NEXT:     `ifdef USE_UNR_ONLY_CONSTRAINTS
    ; CHECK-NEXT:       wire [[GEN_11:.+]] = ~enable | [[GEN_9]];
    ; CHECK-NEXT:       always @(edge [[GEN_11]])
    ; CHECK-NEXT:         assume__verif_library_7: assume([[GEN_11]]);
    ; CHECK-NEXT:     `endif // USE_UNR_ONLY_CONSTRAINTS
    ; CHECK-NEXT:   `endif // USE_FORMAL_ONLY_CONSTRAINTS
    ; CHECK-NEXT: `endif
