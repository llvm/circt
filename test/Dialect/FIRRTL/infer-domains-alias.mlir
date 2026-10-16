// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains{mode=infer-all}))' %s | FileCheck %s

// A two-level alias chain must be instantiated at each hierarchy boundary.
// The connection in AliasTop is legal only if the alias is propagated through
// both AliasMiddle and AliasLeaf.
// CHECK-LABEL: firrtl.circuit "AliasAcrossModules"
firrtl.circuit "AliasAcrossModules" {
  firrtl.domain @ClockDomain

  firrtl.module @AliasLeaf(
    in %A: !firrtl.domain<@ClockDomain()>,
    out %B: !firrtl.domain<@ClockDomain()>
  ) {
    firrtl.domain.define %B, %A : !firrtl.domain<@ClockDomain()>
  }

  firrtl.module @AliasMiddle(
    in %M: !firrtl.domain<@ClockDomain()>,
    out %N: !firrtl.domain<@ClockDomain()>
  ) {
    %leaf_A, %leaf_B = firrtl.instance leaf @AliasLeaf(
      in A: !firrtl.domain<@ClockDomain()>,
      out B: !firrtl.domain<@ClockDomain()>)
    firrtl.domain.define %leaf_A, %M : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %N, %leaf_B : !firrtl.domain<@ClockDomain()>
  }

  firrtl.module @AliasAcrossModules(
    in %D: !firrtl.domain<@ClockDomain()>
  ) {
    %mid_M, %mid_N = firrtl.instance mid @AliasMiddle(
      in M: !firrtl.domain<@ClockDomain()>,
      out N: !firrtl.domain<@ClockDomain()>)
    firrtl.domain.define %mid_M, %D : !firrtl.domain<@ClockDomain()>
    %X = firrtl.wire : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %X, %mid_N : !firrtl.domain<@ClockDomain()>
    %x = firrtl.wire domains[%X] : !firrtl.uint<1>
        domains[!firrtl.domain<@ClockDomain()>]
    %y = firrtl.wire domains[%D] : !firrtl.uint<1>
        domains[!firrtl.domain<@ClockDomain()>]
    firrtl.matchingconnect %x, %y : !firrtl.uint<1>
  }
}

// Alias summaries use the effective interface after domain-port inference.
// The inferred port is inserted before B, so the summary must use B's final
// index when it is instantiated in the parent.
// CHECK-LABEL: firrtl.circuit "AliasWithInferredPort"
firrtl.circuit "AliasWithInferredPort" {
  firrtl.domain @ClockDomain

  firrtl.module @InferredAliasChild(
    in %A: !firrtl.domain<@ClockDomain()>,
    in %i: !firrtl.uint<1>,
    out %B: !firrtl.domain<@ClockDomain()>
  ) {
    firrtl.domain.define %B, %A : !firrtl.domain<@ClockDomain()>
  }

  firrtl.module @AliasWithInferredPort(
    in %D: !firrtl.domain<@ClockDomain()>
  ) {
    %A, %i, %B = firrtl.instance child @InferredAliasChild(
      in A: !firrtl.domain<@ClockDomain()>,
      in i: !firrtl.uint<1>,
      out B: !firrtl.domain<@ClockDomain()>)
    firrtl.domain.define %A, %D : !firrtl.domain<@ClockDomain()>
    %X = firrtl.wire : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %X, %B : !firrtl.domain<@ClockDomain()>
    %x = firrtl.wire domains[%X] : !firrtl.uint<1>
        domains[!firrtl.domain<@ClockDomain()>]
    %y = firrtl.wire domains[%D] : !firrtl.uint<1>
        domains[!firrtl.domain<@ClockDomain()>]
    firrtl.matchingconnect %x, %y : !firrtl.uint<1>
  }
}

// An alias constraint from the parent into a child instance must flow back
// through the child's aliased output.
// CHECK-LABEL: firrtl.circuit "ParentToChildAndBack"
firrtl.circuit "ParentToChildAndBack" {
  firrtl.domain @ClockDomain

  firrtl.module @ParentChild(
    in %A: !firrtl.domain<@ClockDomain()>,
    out %B: !firrtl.domain<@ClockDomain()>
  ) {
    firrtl.domain.define %B, %A : !firrtl.domain<@ClockDomain()>
  }

  firrtl.module @ParentToChildAndBack(
    in %D: !firrtl.domain<@ClockDomain()>
  ) {
    %child_A, %child_B = firrtl.instance child @ParentChild(
      in A: !firrtl.domain<@ClockDomain()>,
      out B: !firrtl.domain<@ClockDomain()>)
    firrtl.domain.define %child_A, %D : !firrtl.domain<@ClockDomain()>
    %X = firrtl.wire : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %X, %child_B : !firrtl.domain<@ClockDomain()>
    %x = firrtl.wire domains[%X] : !firrtl.uint<1>
        domains[!firrtl.domain<@ClockDomain()>]
    %y = firrtl.wire domains[%D] : !firrtl.uint<1>
        domains[!firrtl.domain<@ClockDomain()>]
    firrtl.matchingconnect %x, %y : !firrtl.uint<1>
  }
}

// Aliases common to all InstanceChoiceOp candidates are sound.
// CHECK-LABEL: firrtl.circuit "CommonChoiceAlias"
firrtl.circuit "CommonChoiceAlias" {
  firrtl.domain @ClockDomain
  firrtl.option @Choice {
    firrtl.option_case @A
    firrtl.option_case @B
  }

  firrtl.module @ChoiceA(
    in %A: !firrtl.domain<@ClockDomain()>,
    in %B: !firrtl.domain<@ClockDomain()>,
    out %O: !firrtl.domain<@ClockDomain()>,
    out %P: !firrtl.domain<@ClockDomain()>
  ) {
    firrtl.domain.define %O, %A : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %P, %B : !firrtl.domain<@ClockDomain()>
  }
  firrtl.module @ChoiceB(
    in %A: !firrtl.domain<@ClockDomain()>,
    in %B: !firrtl.domain<@ClockDomain()>,
    out %O: !firrtl.domain<@ClockDomain()>,
    out %P: !firrtl.domain<@ClockDomain()>
  ) {
    // O aliases A in every candidate, while P aliases a different input.
    firrtl.domain.define %O, %A : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %P, %A : !firrtl.domain<@ClockDomain()>
  }

  firrtl.module @CommonChoiceAlias(
    in %D1: !firrtl.domain<@ClockDomain()>,
    in %D2: !firrtl.domain<@ClockDomain()>
  ) {
    %A, %B, %O, %P = firrtl.instance_choice choice @ChoiceA alternatives @Choice {
      @A -> @ChoiceB, @B -> @ChoiceB
    } (
      in A: !firrtl.domain<@ClockDomain()>,
      in B: !firrtl.domain<@ClockDomain()>,
      out O: !firrtl.domain<@ClockDomain()>,
      out P: !firrtl.domain<@ClockDomain()>)
    firrtl.domain.define %A, %D1 : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %B, %D2 : !firrtl.domain<@ClockDomain()>
    %X = firrtl.wire : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %X, %O : !firrtl.domain<@ClockDomain()>
    %x = firrtl.wire domains[%X] : !firrtl.uint<1>
        domains[!firrtl.domain<@ClockDomain()>]
    %y = firrtl.wire domains[%D1] : !firrtl.uint<1>
        domains[!firrtl.domain<@ClockDomain()>]
    firrtl.matchingconnect %x, %y : !firrtl.uint<1>
  }
}
