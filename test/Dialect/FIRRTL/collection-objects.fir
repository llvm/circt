; RUN: circt-translate --import-firrtl %s | FileCheck %s --check-prefix=IMPORT
; RUN: circt-translate --import-firrtl %s | circt-opt --firrtl-lower-classes | FileCheck %s --check-prefix=LOWER

FIRRTL version 7.0.0
circuit Objects:
  class Item:
    output value: Integer
    propassign value, Integer(1)

  ; IMPORT-LABEL: firrtl.module @Objects
  ; IMPORT: firrtl.prop.collection : !firrtl.mutable_collection<anyref>
  ; IMPORT: firrtl.object.anyref_cast
  ; IMPORT: firrtl.prop.push %local,
  ; LOWER-LABEL: om.class @Objects_Class
  ; LOWER: %[[ITEMS:.*]] = om.collection : !om.collection<!om.any>
  ; LOWER: %[[OBJECT:.*]] = om.object @Item
  ; LOWER: %[[ANY:.*]] = om.any_cast %[[OBJECT]]
  ; LOWER: om.collection_push %[[ITEMS]], %[[ANY]] : !om.collection<!om.any>, !om.any
  ; LOWER: om.class.fields %[[ITEMS]] : !om.collection<!om.any>
  public module Objects:
    output items: MutableCollection<AnyRef>
    wire local: MutableCollection<AnyRef>
    propassign local, MutableCollection<AnyRef>()
    propassign items, local
    object item of Item
    push local, item
