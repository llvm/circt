// RUN: circt-opt %s --om-freeze-paths | FileCheck %s

// CHECK-LABEL: om.class @Paths(%input: !om.collection<!om.frozenpath>)
// CHECK-SAME: -> (items: !om.collection<!om.frozenpath>, nested: !om.collection<!om.list<!om.collection<!om.frozenpath>>>)
om.class @Paths(%input: !om.collection<!om.path>) -> (items: !om.collection<!om.path>, nested: !om.collection<!om.list<!om.collection<!om.path>>>) {
  // CHECK: %[[ITEMS:.*]] = om.collection {test = "keep"} : !om.collection<!om.frozenpath>
  %items = om.collection {test = "keep"} : !om.collection<!om.path>
  // CHECK: %[[PATH:.*]] = om.frozenpath_empty
  %path = om.path_empty
  // CHECK: om.collection_push %input, %[[PATH]] : !om.collection<!om.frozenpath>, !om.frozenpath
  om.collection_push %input, %path : !om.collection<!om.path>, !om.path
  // CHECK: om.collection_push %[[ITEMS]], %[[PATH]] : !om.collection<!om.frozenpath>, !om.frozenpath
  om.collection_push %items, %path : !om.collection<!om.path>, !om.path
  // CHECK: om.collection %[[PATH]] : !om.collection<!om.frozenpath>
  %initial = om.collection %path : !om.collection<!om.path>
  // CHECK: %[[NESTED:.*]] = om.collection : !om.collection<!om.list<!om.collection<!om.frozenpath>>>
  %nested = om.collection : !om.collection<!om.list<!om.collection<!om.path>>>
  // CHECK: %[[LIST:.*]] = om.list_create %[[ITEMS]] : !om.collection<!om.frozenpath>
  %list = om.list_create %items : !om.collection<!om.path>
  // CHECK: om.collection_push %[[NESTED]], %[[LIST]] : !om.collection<!om.list<!om.collection<!om.frozenpath>>>, !om.list<!om.collection<!om.frozenpath>>
  om.collection_push %nested, %list : !om.collection<!om.list<!om.collection<!om.path>>>, !om.list<!om.collection<!om.path>>
  om.class.fields %items, %nested : !om.collection<!om.path>, !om.collection<!om.list<!om.collection<!om.path>>>
}
