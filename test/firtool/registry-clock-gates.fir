; RUN: firtool %s -domain-mode=check -output-final-mlir %t.mlir -o /dev/null
; RUN: domaintool --module Foo \
; RUN:     --domain ClockDomain,A,,synchronous \
; RUN:     --domain ClockDomain,B,A,synchronous \
; RUN:     --assign 0 --assign 1 \
; RUN:     %t.mlir | FileCheck %s --check-prefix=FOO
; RUN: domaintool --module MidA \
; RUN:     --domain ClockDomain,A,,synchronous --assign 0 \
; RUN:     %t.mlir | FileCheck %s --check-prefix=MIDA
; RUN: domaintool --module MidB \
; RUN:     --domain ClockDomain,B,,synchronous --assign 0 \
; RUN:     %t.mlir | FileCheck %s --check-prefix=MIDB
; RUN: domaintool --module LeafA \
; RUN:     --domain ClockDomain,A,,synchronous --assign 0 \
; RUN:     %t.mlir | FileCheck %s --check-prefix=LEAFA
; RUN: domaintool --module LeafB \
; RUN:     --domain ClockDomain,B,,synchronous --assign 0 \
; RUN:     %t.mlir | FileCheck %s --check-prefix=LEAFB
; RUN: firtool %s -domain-mode=strip -o /dev/null

; Multi-domain, multi-depth Registry clockGates through firtool + domaintool.
;
; Hierarchy:
;   Foo
;     gateA0          -- local gate on domain A (depth 0)
;     midA : MidA
;       gateA1        -- domain A (depth 1)
;       leafA : LeafA
;         gateA2      -- domain A (depth 2)
;     midB : MidB
;       gateB1        -- domain B (depth 1)
;       leafB : LeafB
;         gateB2a     -- domain B (depth 2)
;         gateB2b     -- domain B (depth 2, second gate)

FIRRTL version 7.0.0
circuit Foo:
  ; Fields required by domaintool's Clock Spec JSON handler, plus a registry.
  domain ClockDomain:
    name: String
    source: String
    relationship: String
    clockGates: Registry<Path>

  extmodule ClockGate:

  ; Deepest leaf on domain A: one gate.
  public module LeafA:
    input A: Domain of ClockDomain
    input d: UInt<1> domains [A]

    inst gateA2 of ClockGate
    insert A[clockGates], path("OMInstanceTarget:~Foo|LeafA/gateA2:ClockGate")

  ; Mid level on domain A: local gate + deeper leaf.
  public module MidA:
    input A: Domain of ClockDomain
    input d: UInt<1> domains [A]

    inst gateA1 of ClockGate
    insert A[clockGates], path("OMInstanceTarget:~Foo|MidA/gateA1:ClockGate")

    inst leafA of LeafA
    domain_define leafA.A = A
    connect leafA.d, d

  ; Deepest leaf on domain B: two gates at the same level.
  public module LeafB:
    input B: Domain of ClockDomain
    input d: UInt<1> domains [B]

    inst gateB2a of ClockGate
    inst gateB2b of ClockGate
    insert B[clockGates], path("OMInstanceTarget:~Foo|LeafB/gateB2a:ClockGate")
    insert B[clockGates], path("OMInstanceTarget:~Foo|LeafB/gateB2b:ClockGate")

  ; Mid level on domain B: local gate + deeper leaf.
  public module MidB:
    input B: Domain of ClockDomain
    input d: UInt<1> domains [B]

    inst gateB1 of ClockGate
    insert B[clockGates], path("OMInstanceTarget:~Foo|MidB/gateB1:ClockGate")

    inst leafB of LeafB
    domain_define leafB.B = B
    connect leafB.d, d

  ; Top: two domains, mixed depths, plus a local A gate.
  public module Foo:
    input A: Domain of ClockDomain
    input B: Domain of ClockDomain
    input a: UInt<1> domains [A]
    input b: UInt<1> domains [B]

    ; Domain A local at the top (depth 0).
    inst gateA0 of ClockGate
    insert A[clockGates], path("OMInstanceTarget:~Foo|Foo/gateA0:ClockGate")

    ; Domain A subtree: depth 1 and 2.
    inst midA of MidA
    domain_define midA.A = A
    connect midA.d, a

    ; Domain B subtree: depth 1 and 2 (two gates at depth 2).
    inst midB of MidB
    domain_define midB.B = B
    connect midB.d, b

; FOO:      "name_pattern": "A"
; FOO:      "clock_gates": [
; FOO-NEXT:   "OMInstanceTarget:~Foo|Foo/gateA0:ClockGate",
; FOO-NEXT:   "OMInstanceTarget:~Foo|Foo/midA:MidA/gateA1:ClockGate",
; FOO-NEXT:   "OMInstanceTarget:~Foo|Foo/midA:MidA/leafA:LeafA/gateA2:ClockGate"
; FOO-NEXT: ]
; FOO:      "name_pattern": "B"
; FOO:      "clock_gates": [
; FOO-NEXT:   "OMInstanceTarget:~Foo|Foo/midB:MidB/gateB1:ClockGate",
; FOO-NEXT:   "OMInstanceTarget:~Foo|Foo/midB:MidB/leafB:LeafB/gateB2a:ClockGate",
; FOO-NEXT:   "OMInstanceTarget:~Foo|Foo/midB:MidB/leafB:LeafB/gateB2b:ClockGate"
; FOO-NEXT: ]

; Mid-level tops root paths at the selected module.
; MIDA:      "name_pattern": "A"
; MIDA:      "clock_gates": [
; MIDA-NEXT:   "OMInstanceTarget:~MidA|MidA/gateA1:ClockGate",
; MIDA-NEXT:   "OMInstanceTarget:~MidA|MidA/leafA:LeafA/gateA2:ClockGate"
; MIDA-NEXT: ]

; MIDB:      "name_pattern": "B"
; MIDB:      "clock_gates": [
; MIDB-NEXT:   "OMInstanceTarget:~MidB|MidB/gateB1:ClockGate",
; MIDB-NEXT:   "OMInstanceTarget:~MidB|MidB/leafB:LeafB/gateB2a:ClockGate",
; MIDB-NEXT:   "OMInstanceTarget:~MidB|MidB/leafB:LeafB/gateB2b:ClockGate"
; MIDB-NEXT: ]

; Leaf tops only report local contributions.
; LEAFA:      "name_pattern": "A"
; LEAFA:      "clock_gates": [
; LEAFA-NEXT:   "OMInstanceTarget:~LeafA|LeafA/gateA2:ClockGate"
; LEAFA-NEXT: ]

; LEAFB:      "name_pattern": "B"
; LEAFB:      "clock_gates": [
; LEAFB-NEXT:   "OMInstanceTarget:~LeafB|LeafB/gateB2a:ClockGate",
; LEAFB-NEXT:   "OMInstanceTarget:~LeafB|LeafB/gateB2b:ClockGate"
; LEAFB-NEXT: ]
