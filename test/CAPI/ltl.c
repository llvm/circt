//===- ltl.c - LTL Dialect CAPI unit tests --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: circt-capi-ltl-test 2>&1 | FileCheck %s

#include <assert.h>
#include <stdio.h>

#include "circt-c/Dialect/LTL.h"
#include "mlir-c/IR.h"

static void testTypesAndAttrs(MlirContext ctx) {
  {
    MlirType sequenceTy = ltlSequenceTypeGet(ctx);

    // CHECK: is_sequence
    fprintf(stderr, ltlTypeIsASequence(sequenceTy) ? "is_sequence\n"
                                                   : "isnot_sequence\n");
    // CHECK: !ltl.sequence
    mlirTypeDump(sequenceTy);
  }

  {
    MlirType propertyTy = ltlPropertyTypeGet(ctx);

    // CHECK: is_property
    fprintf(stderr, ltlTypeIsAProperty(propertyTy) ? "is_property\n"
                                                   : "isnot_property\n");
    // CHECK: !ltl.property
    mlirTypeDump(propertyTy);
  }

  {
    MlirAttribute edgeAttr = ltlClockEdgeAttrGet(ctx, LTL_CLOCK_EDGE_BOTH);

    // CHECK: is_clock_edge
    fprintf(stderr, ltlAttrIsAClockEdgeAttr(edgeAttr) ? "is_clock_edge\n"
                                                      : "isnot_clock_edge\n");
    // CHECK: 2 : i32
    mlirAttributeDump(edgeAttr);

    // CHECK: edge
    switch (ltlClockEdgeAttrGetValue(edgeAttr)) {
    case LTL_CLOCK_EDGE_POS:
      fprintf(stderr, "posedge\n");
      break;
    case LTL_CLOCK_EDGE_NEG:
      fprintf(stderr, "negedge\n");
      break;
    case LTL_CLOCK_EDGE_BOTH:
      fprintf(stderr, "edge\n");
      break;
    }
  }
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  MlirDialectHandle ltlHandle = mlirGetDialectHandle__ltl__();
  MlirDialect ltl = mlirDialectHandleLoadDialect(ltlHandle, ctx);
  assert(!mlirDialectIsNull(ltl));

  testTypesAndAttrs(ctx);

  mlirContextDestroy(ctx);
  return 0;
}
