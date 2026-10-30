; RUN: circt-translate --import-firrtl %s | FileCheck %s --check-prefix=IMPORT
; RUN: circt-translate --import-firrtl %s | circt-translate --export-firrtl | FileCheck %s --check-prefix=EXPORT
; RUN: circt-translate --import-firrtl %s | circt-translate --export-firrtl | circt-translate --import-firrtl | FileCheck %s --check-prefix=IMPORT

FIRRTL version 7.0.0
circuit Collections:
  ; IMPORT-LABEL: firrtl.module @Collections
  ; IMPORT-SAME: in %items: !firrtl.mutable_collection<integer>
  ; IMPORT-SAME: out %result: !firrtl.mutable_collection<integer>
  ; EXPORT-LABEL: public module Collections :
  ; EXPORT: input items : MutableCollection<Integer>
  ; EXPORT: output result : MutableCollection<Integer>
  public module Collections:
    input items: MutableCollection<Integer>
    output result: MutableCollection<Integer>
    ; IMPORT: %[[LOCAL:.*]] = firrtl.prop.collection : !firrtl.mutable_collection<integer>
    ; EXPORT: wire local : MutableCollection<Integer>
    ; EXPORT: propassign local, MutableCollection<Integer>()
    wire local: MutableCollection<Integer>
    propassign local, MutableCollection<Integer>()
    ; IMPORT: firrtl.prop.push %items,
    ; IMPORT: firrtl.prop.push %items,
    ; IMPORT: firrtl.prop.push %local,
    ; EXPORT: push items, Integer(1)
    ; EXPORT: push items, Integer(1)
    ; EXPORT: push local, Integer(1)
    push items, Integer(1)
    push items, Integer(1)
    push local, Integer(1)
    propassign result, local
