; RUN: firtool %s -output-final-mlir - -o /dev/null -domain-mode=check -strip-domain=false | FileCheck %s
; RUN: firtool %s -output-final-mlir - -o /dev/null -domain-mode=check -strip-domain=false | domaintool --module Top --domain ClockDomain,A --assign 0 | FileCheck %s --check-prefix=JSON

; A domain field can be a MutableCollection. Every module associated with the
; domain contributes to the same collection through the domain's handle, which
; replaces bottom-up registry plumbing: no extra ports are generated, and each
; instance contributes its own instance-specific path.

FIRRTL version 7.0.0
circuit Top:
  domain ClockDomain:
    name: String
    clockGates: MutableCollection<Path>

  ; CHECK-LABEL: om.class private @ClockGate_Class
  ; CHECK-SAME:    (%basepath: !om.frozenbasepath, %D: !om.class.type<@ClockDomain>)
  ; CHECK:         %[[GATES:.+]] = om.object.field %D["clockGates_out"]
  ; CHECK:         %[[PATH:.+]] = om.frozenpath_create reference %basepath "ClockGate>gated"
  ; CHECK:         om.collection_push %[[GATES]], %[[PATH]] : !om.collection<!om.frozenpath>, !om.frozenpath
  module ClockGate:
    input D: Domain of ClockDomain
    input clock: Clock domains [D]
    input en: UInt<1> domains [D]
    output gated: Clock domains [D]
    connect gated, clock
    push D.clockGates, path("OMReferenceTarget:~Top|ClockGate>gated")

  ; Elaboration inlines both gates. Their pushes target the domain parameter,
  ; so they remain until the evaluator imports a concrete domain object.
  ; CHECK-LABEL: om.class @Top_Class
  ; CHECK-SAME:    (%basepath: !om.frozenbasepath, %A: !om.class.type<@ClockDomain>)
  ; CHECK:         %[[G0:.+]] = om.frozenbasepath_create %basepath "Top/g0"
  ; CHECK:         %[[GATES0:.+]] = om.object.field %A["clockGates_out"]
  ; CHECK:         %[[PATH0:.+]] = om.frozenpath_create reference %[[G0]] "ClockGate>gated"
  ; CHECK:         om.collection_push %[[GATES0]], %[[PATH0]]
  ; CHECK:         %[[G1:.+]] = om.frozenbasepath_create %basepath "Top/g1"
  ; CHECK:         %[[GATES1:.+]] = om.object.field %A["clockGates_out"]
  ; CHECK:         %[[PATH1:.+]] = om.frozenpath_create reference %[[G1]] "ClockGate>gated"
  ; CHECK:         om.collection_push %[[GATES1]], %[[PATH1]]
  public module Top:
    input A: Domain of ClockDomain
    input clock: Clock domains [A]
    input en: UInt<1> domains [A]
    output gated0: Clock domains [A]
    output gated1: Clock domains [A]
    inst g0 of ClockGate
    domain_define g0.D = A
    connect g0.clock, clock
    connect g0.en, en
    connect gated0, g0.gated
    inst g1 of ClockGate
    domain_define g1.D = A
    connect g1.clock, clock
    connect g1.en, en
    connect gated1, g1.gated

; The domain class carries the collection like any other field.
; CHECK-LABEL: om.class @ClockDomain
; CHECK-SAME:    (%basepath: !om.frozenbasepath, %name_in: !om.string, %clockGates_in: !om.collection<!om.frozenpath>)
; CHECK-SAME:    -> (name_out: !om.string, clockGates_out: !om.collection<!om.frozenpath>)

; The domain tool supplies the initial empty collection and emits the populated
; registry from A_out.domainInfo_out, including each gate's instance path.
; JSON:      "name_pattern": "A",
; JSON-NEXT: "define_period": "A_PERIOD",
; JSON-NEXT: "clock_gates": [
; JSON-NEXT:   "OMReferenceTarget:~Top|Top/g0:ClockGate>gated",
; JSON-NEXT:   "OMReferenceTarget:~Top|Top/g1:ClockGate>gated"
; JSON-NEXT: ],
; JSON-NEXT: "clock_relationships": []
; JSON:      "port_patterns": [
; JSON-NEXT:   "clock",
; JSON-NEXT:   "en",
; JSON-NEXT:   "gated0",
; JSON-NEXT:   "gated1"
; JSON-NEXT: ]
