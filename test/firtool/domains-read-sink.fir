; RUN: firtool %s -domain-mode=infer-all -strip-domain=false -output-final-mlir - -o /dev/null | FileCheck %s

; https://github.com/llvm/circt/issues/11601

FIRRTL version 7.0.0
circuit Top :
  domain ClockDomain :
    name : String
    source : String
    relationship : String

  extmodule Divider :
    input in : Clock domains [A]
    output out : Clock domains [B]
    input A : Domain of ClockDomain
    input B : Domain of ClockDomain
    defname = Divider

  module Wrapper :
    input in : Clock
    output out : Clock
    inst divider of Divider
    wire name : String
    propassign name, string_concat(divider.A.name, String("_div2"))
    domain divided of ClockDomain(name, divider.A.name, String("synchronous"))
    domain_define divider.B = divided
    connect divider.in, in
    connect out, divider.out

  public module Top :
    input clock : Clock
    output out : Clock
    inst wrapper of Wrapper
    connect wrapper.in, clock
    connect out, wrapper.out

; CHECK-LABEL: om.class private @Wrapper_Class(
; CHECK-SAME:    %ClockDomain: !om.class.type<@ClockDomain>
; CHECK:         om.object @Divider_Class(%{{.+}}, %ClockDomain, %[[divided:.+]]) :
; CHECK:         %[[name:.+]] = om.object.field %ClockDomain["name_out"]
; CHECK-NEXT:    %[[concat:.+]] = om.string.concat %[[name]],
; CHECK-NEXT:    %[[divided]] = om.object @ClockDomain(%basepath, %[[concat]], %[[name]],
