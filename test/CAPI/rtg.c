//===- rtg.c - RTG Dialect CAPI unit tests --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: circt-capi-rtg-test 2>&1 | FileCheck %s

#include <stdio.h>

#include "circt-c/Dialect/RTG.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"

static void testSequenceType(MlirContext ctx) {
  MlirType sequenceTy = rtgSequenceTypeGet(ctx);

  // CHECK: is_sequence
  fprintf(stderr, rtgTypeIsASequence(sequenceTy) ? "is_sequence\n"
                                                 : "isnot_sequence\n");
  // CHECK: !rtg.sequence
  mlirTypeDump(sequenceTy);
}

static void testLabelType(MlirContext ctx) {
  MlirType labelTy = rtgLabelTypeGet(ctx);

  // CHECK: is_label
  fprintf(stderr, rtgTypeIsALabel(labelTy) ? "is_label\n" : "isnot_label\n");
  // CHECK: !rtg.label
  mlirTypeDump(labelTy);
}

static void testSetType(MlirContext ctx) {
  MlirType elTy = mlirIntegerTypeGet(ctx, 32);
  MlirType setTy = rtgSetTypeGet(elTy);

  // CHECK: is_set
  fprintf(stderr, rtgTypeIsASet(setTy) ? "is_set\n" : "isnot_set\n");
  // CHECK: !rtg.set<i32>
  mlirTypeDump(setTy);
}

static void testBagType(MlirContext ctx) {
  MlirType elTy = mlirIntegerTypeGet(ctx, 32);
  MlirType bagTy = rtgBagTypeGet(elTy);

  // CHECK: is_bag
  fprintf(stderr, rtgTypeIsABag(bagTy) ? "is_bag\n" : "isnot_bag\n");
  // CHECK: !rtg.bag<i32>
  mlirTypeDump(bagTy);
}

static void testDictType(MlirContext ctx) {
  MlirType elTy = mlirIntegerTypeGet(ctx, 32);
  MlirAttribute name0 =
      mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("entry0"));
  MlirAttribute name1 =
      mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("entry1"));
  const MlirAttribute names[] = {name0, name1};
  const MlirType types[] = {elTy, elTy};
  MlirType dictTy = rtgDictTypeGet(ctx, 2, names, types);

  // CHECK: is_dict
  fprintf(stderr, rtgTypeIsADict(dictTy) ? "is_dict\n" : "isnot_dict\n");
  // CHECK: !rtg.dict<entry0: i32, entry1: i32>
  mlirTypeDump(dictTy);

  MlirType emptyDictTy = rtgDictTypeGet(ctx, 0, NULL, NULL);
  // CHECK: is_dict
  fprintf(stderr, rtgTypeIsADict(dictTy) ? "is_dict\n" : "isnot_dict\n");
  // CHECK: !rtg.dict<>
  mlirTypeDump(emptyDictTy);
}

int main(int argc, char **argv) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__rtg__(), ctx);

  testSequenceType(ctx);
  testLabelType(ctx);
  testSetType(ctx);
  testBagType(ctx);
  testDictType(ctx);

  mlirContextDestroy(ctx);

  return 0;
}
