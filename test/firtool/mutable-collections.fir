; RUN: firtool %s --output-final-mlir - -o /dev/null | FileCheck %s

FIRRTL version 7.0.0
circuit Collections:
  class Writer:
    input items: MutableCollection<Integer>
    input value: Integer
    push items, value

  ; CHECK-LABEL: om.class @Collections_Class
  ; CHECK-SAME: items: !om.collection<!om.integer>
  ; CHECK-DAG: om.constant #om.integer<1 : si32>
  ; CHECK-DAG: om.constant #om.integer<2 : si32>
  ; CHECK: om.collection %{{.+}}, %{{.+}} : !om.collection<!om.integer>
  ; CHECK-NOT: om.collection_push
  ; CHECK-NOT: om.object @Writer
  ; CHECK: om.class.fields
  public module Collections:
    output items: MutableCollection<Integer>
    wire local: MutableCollection<Integer>
    propassign local, MutableCollection<Integer>()
    propassign items, local
    object a of Writer
    object b of Writer
    propassign a.items, local
    propassign a.value, Integer(1)
    propassign b.items, local
    propassign b.value, Integer(2)
