; RUN: firtool %s -output-final-mlir - -o /dev/null -domain-mode=check -strip-domain=false | FileCheck %s

; Modules sharing a domain can add their instance paths to its registry.

FIRRTL version 7.0.0
circuit Top:
  domain ClockDomain:
    name: String
    clockGates: Registry<Path>

  ; CHECK-LABEL: om.class private @ClockGate_Class
  ; CHECK-SAME:    (%basepath: !om.frozenbasepath, %D: !om.class.type<@ClockDomain>)
  ; CHECK:         %[[GATES:.+]] = om.object.field %D["clockGates_out"]
  ; CHECK:         %[[PATH:.+]] = om.frozenpath_create reference %basepath "ClockGate>gated"
  ; CHECK:         om.registry_insert %[[GATES]], %[[PATH]] : !om.registry<!om.frozenpath>, !om.frozenpath
  module ClockGate:
    input D: Domain of ClockDomain
    input clock: Clock domains [D]
    output gated: Clock domains [D]
    connect gated, clock
    registry_insert D.clockGates, path("OMReferenceTarget:~Top|ClockGate>gated")

  ; The insertions remain until the evaluator receives a domain object.
  ; CHECK-LABEL: om.class @Top_Class
  ; CHECK-SAME:    (%basepath: !om.frozenbasepath, %A: !om.class.type<@ClockDomain>)
  ; CHECK:         %[[G0:.+]] = om.frozenbasepath_create %basepath "Top/g0"
  ; CHECK:         %[[GATES0:.+]] = om.object.field %A["clockGates_out"]
  ; CHECK:         %[[PATH0:.+]] = om.frozenpath_create reference %[[G0]] "ClockGate>gated"
  ; CHECK:         om.registry_insert %[[GATES0]], %[[PATH0]]
  ; CHECK:         %[[G1:.+]] = om.frozenbasepath_create %basepath "Top/g1"
  ; CHECK:         %[[GATES1:.+]] = om.object.field %A["clockGates_out"]
  ; CHECK:         %[[PATH1:.+]] = om.frozenpath_create reference %[[G1]] "ClockGate>gated"
  ; CHECK:         om.registry_insert %[[GATES1]], %[[PATH1]]
  public module Top:
    input A: Domain of ClockDomain
    input clock: Clock domains [A]
    output gated0: Clock domains [A]
    output gated1: Clock domains [A]
    inst g0 of ClockGate
    domain_define g0.D = A
    connect g0.clock, clock
    connect gated0, g0.gated
    inst g1 of ClockGate
    domain_define g1.D = A
    connect g1.clock, clock
    connect gated1, g1.gated

; CHECK-LABEL: om.class @ClockDomain
; CHECK-SAME:    (%basepath: !om.frozenbasepath, %name_in: !om.string, %clockGates_in: !om.registry<!om.frozenpath>)
; CHECK-SAME:    -> (name_out: !om.string, clockGates_out: !om.registry<!om.frozenpath>)
