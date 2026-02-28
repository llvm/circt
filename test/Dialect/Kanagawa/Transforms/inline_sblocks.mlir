// RUN: circt-opt --pass-pipeline='builtin.module(kanagawa.design(kanagawa.class(kanagawa.method(kanagawa-inline-sblocks))))' \
// RUN:   --allow-unregistered-dialect %s | FileCheck %s

kanagawa.design @foo {

// CHECK-LABEL:   kanagawa.class sym @Inline1 {
// CHECK:           kanagawa.method @foo(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32) -> () {
// CHECK:             kanagawa.sblock.inline.begin {maxThreads = 1 : i64}
// CHECK:             %[[VAL_3:.*]] = "foo.op1"(%[[VAL_1]], %[[VAL_2]]) : (i32, i32) -> i32
// CHECK:             kanagawa.sblock.inline.end
// CHECK:             kanagawa.return
// CHECK:           }
// CHECK:         }
kanagawa.class sym @Inline1 {

  kanagawa.method @foo(%a : i32, %b : i32) {
    %0 = kanagawa.sblock() -> (i32) attributes {maxThreads = 1} {
      %res = "foo.op1"(%a, %b) : (i32, i32) -> i32
      kanagawa.sblock.return %res : i32
    }
    kanagawa.return
  }
}

// CHECK-LABEL:   kanagawa.class sym @Inline2 {
// CHECK:           kanagawa.method @foo(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32) -> () {
// CHECK:             "foo.unused1"() : () -> ()
// CHECK:             kanagawa.sblock.inline.begin {maxThreads = 1 : i64}
// CHECK:             %[[VAL_3:.*]] = "foo.op1"(%[[VAL_1]], %[[VAL_2]]) : (i32, i32) -> i32
// CHECK:             kanagawa.sblock.inline.end
// CHECK:             "foo.unused2"() : () -> ()
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             "foo.unused3"() : () -> ()
// CHECK:             %[[VAL_4:.*]] = "foo.op2"(%[[VAL_3]], %[[VAL_1]]) : (i32, i32) -> i32
// CHECK:             "foo.unused4"() : () -> ()
// CHECK:             kanagawa.return
// CHECK:           }
// CHECK:         }
kanagawa.class sym @Inline2 {
  kanagawa.method @foo(%a : i32, %b : i32) {
    "foo.unused1"() : () -> ()
    %0 = kanagawa.sblock() -> (i32) attributes {maxThreads = 1} {
      %res = "foo.op1"(%a, %b) : (i32, i32) -> i32
      kanagawa.sblock.return %res : i32
    }
    "foo.unused2"() : () -> ()
    cf.br ^bb1
  ^bb1:
    "foo.unused3"() : () -> ()
    %1 = kanagawa.sblock() -> (i32) {
      %res = "foo.op2"(%0, %a) : (i32, i32) -> i32
      kanagawa.sblock.return %res : i32
    }
    "foo.unused4"() : () -> ()
    kanagawa.return
  }
}

}
