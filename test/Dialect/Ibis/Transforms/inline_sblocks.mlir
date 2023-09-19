// RUN: circt-opt %s --pass-pipeline='builtin.module(ibis.class(ibis.method(ibis-inline-sblocks)))' --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL:   ibis.class @Inline1 {
// CHECK:           %[[VAL_0:.*]] = ibis.this @Inline1
// CHECK:           ibis.method @foo(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32) {
// CHECK:             ibis.sblock.inline.begin {maxThreads = 1 : i64}
// CHECK:             %[[VAL_3:.*]] = "foo.op1"(%[[VAL_1]], %[[VAL_2]]) : (i32, i32) -> i32
// CHECK:             ibis.sblock.inline.end
// CHECK:             ibis.return
// CHECK:           }
// CHECK:         }
ibis.class @Inline1 {
  %this = ibis.this @Inline1

  ibis.method @foo(%a : i32, %b : i32) {
    %0 = ibis.sblock() -> (i32) attributes {maxThreads = 1} {
      %res = "foo.op1"(%a, %b) : (i32, i32) -> i32
      ibis.sblock.return %res : i32
    }
    ibis.return
  }
}

// CHECK-LABEL:   ibis.class @Inline2 {
// CHECK:           %[[VAL_0:.*]] = ibis.this @Inline2
// CHECK:           ibis.method @foo(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32) {
// CHECK:             ibis.sblock.inline.begin {maxThreads = 1 : i64}
// CHECK:             %[[VAL_3:.*]] = "foo.op1"(%[[VAL_1]], %[[VAL_2]]) : (i32, i32) -> i32
// CHECK:             ibis.sblock.inline.end
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             ibis.sblock.inline.begin
// CHECK:             %[[VAL_4:.*]] = "foo.op2"(%[[VAL_3]], %[[VAL_1]]) : (i32, i32) -> i32
// CHECK:             ibis.sblock.inline.end
// CHECK:             ibis.return
// CHECK:           }
// CHECK:         }
ibis.class @Inline2 {
  %this = ibis.this @Inline2

  // Given the "simple" blocks (i.e. the ibis.sblock is the entirety of the MLIR
  // block), we should see that no superflous MLIR blocks are created.
  ibis.method @foo(%a : i32, %b : i32) {
    %0 = ibis.sblock() -> (i32) attributes {maxThreads = 1} {
      %res = "foo.op1"(%a, %b) : (i32, i32) -> i32
      ibis.sblock.return %res : i32
    }
    cf.br ^bb1
  ^bb1:
    %1 = ibis.sblock() -> (i32) {
      %res = "foo.op2"(%0, %a) : (i32, i32) -> i32
      ibis.sblock.return %res : i32
    }
    ibis.return
  }
}


// CHECK-LABEL:   ibis.class @Inline3 {
// CHECK:           %[[VAL_0:.*]] = ibis.this @Inline3
// CHECK:           ibis.method @foo(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32) {
// CHECK:             "foo.unused1"() : () -> ()
// CHECK:             ibis.sblock.inline.begin {maxThreads = 1 : i64}
// CHECK:             %[[VAL_3:.*]] = "foo.op1"(%[[VAL_1]], %[[VAL_2]]) : (i32, i32) -> i32
// CHECK:             ibis.sblock.inline.end
// CHECK:             "foo.unused2"() : () -> ()
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             "foo.unused3"() : () -> ()
// CHECK:             ibis.sblock.inline.begin
// CHECK:             %[[VAL_4:.*]] = "foo.op2"(%[[VAL_3]], %[[VAL_1]]) : (i32, i32) -> i32
// CHECK:             ibis.sblock.inline.end
// CHECK:             "foo.unused4"() : () -> ()
// CHECK:             ibis.return
// CHECK:           }
// CHECK:         }
ibis.class @Inline3 {
  %this = ibis.this @Inline3

  // Opposite to the above, we have extra ops surrounding the ibis.sblock,
  // which requires the addition of new MLIR blocks and cf ops.
  ibis.method @foo(%a : i32, %b : i32) {
    "foo.unused1"() : () -> ()
    %0 = ibis.sblock() -> (i32) attributes {maxThreads = 1} {
      %res = "foo.op1"(%a, %b) : (i32, i32) -> i32
      ibis.sblock.return %res : i32
    }
    "foo.unused2"() : () -> ()
    cf.br ^bb1
  ^bb1:
    "foo.unused3"() : () -> ()
    %1 = ibis.sblock() -> (i32) {
      %res = "foo.op2"(%0, %a) : (i32, i32) -> i32
      ibis.sblock.return %res : i32
    }
    "foo.unused4"() : () -> ()
    ibis.return
  }
}
