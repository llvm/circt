//===- rtg.c - RTG Dialect CAPI unit tests --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: circt-capi-rtg-test 2>&1 | FileCheck %s

#include <inttypes.h>
#include <stdio.h>

#include "circt-c/Dialect/RTG.h"
#include "circt-c/Dialect/RTGTest.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"

static void testSequenceType(MlirContext ctx) {
  MlirType sequenceTy = rtgSequenceTypeGet(ctx, 0, NULL);

  // CHECK: is_sequence
  fprintf(stderr, rtgTypeIsASequence(sequenceTy) ? "is_sequence\n"
                                                 : "isnot_sequence\n");
  // CHECK: !rtg.sequence
  mlirTypeDump(sequenceTy);

  MlirType sequenceWithArgsTy = rtgSequenceTypeGet(ctx, 1, &sequenceTy);
  // CHECK: is_sequence
  fprintf(stderr, rtgTypeIsASequence(sequenceWithArgsTy) ? "is_sequence\n"
                                                         : "isnot_sequence\n");
  // CHECK: 1
  fprintf(stderr, "%d\n", rtgSequenceTypeGetNumElements(sequenceWithArgsTy));
  // CHECK: !rtg.sequence
  mlirTypeDump(rtgSequenceTypeGetElement(sequenceWithArgsTy, 0));
  // CHECK: !rtg.sequence<!rtg.sequence>
  mlirTypeDump(sequenceWithArgsTy);
}

static void testRandomizedSequenceType(MlirContext ctx) {
  MlirType sequenceTy = rtgRandomizedSequenceTypeGet(ctx);

  // CHECK: is_randomized_sequence
  fprintf(stderr, rtgTypeIsARandomizedSequence(sequenceTy)
                      ? "is_randomized_sequence\n"
                      : "isnot_randomized_sequence\n");
  // CHECK: !rtg.randomized_sequence
  mlirTypeDump(sequenceTy);
}

static void testLabelType(MlirContext ctx) {
  MlirType labelTy = rtgLabelTypeGet(ctx);

  // CHECK: is_label
  fprintf(stderr, rtgTypeIsALabel(labelTy) ? "is_label\n" : "isnot_label\n");
  // CHECK: !rtg.isa.label
  mlirTypeDump(labelTy);
}

static void testSetType(MlirContext ctx) {
  MlirType elTy = mlirIntegerTypeGet(ctx, 32);
  MlirType setTy = rtgSetTypeGet(elTy);

  // CHECK: is_set
  fprintf(stderr, rtgTypeIsASet(setTy) ? "is_set\n" : "isnot_set\n");
  // CHECK: !rtg.set<i32>
  mlirTypeDump(setTy);
  // CHECK: i32{{$}}
  mlirTypeDump(rtgSetTypeGetElementType(setTy));
}

static void testBagType(MlirContext ctx) {
  MlirType elTy = mlirIntegerTypeGet(ctx, 32);
  MlirType bagTy = rtgBagTypeGet(elTy);

  // CHECK: is_bag
  fprintf(stderr, rtgTypeIsABag(bagTy) ? "is_bag\n" : "isnot_bag\n");
  // CHECK: !rtg.bag<i32>
  mlirTypeDump(bagTy);
  // CHECK: i32{{$}}
  mlirTypeDump(rtgBagTypeGetElementType(bagTy));
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

static void testArrayType(MlirContext ctx) {
  MlirType i32Type = mlirIntegerTypeGet(ctx, 32);
  MlirType arrayTy = rtgArrayTypeGet(i32Type);

  // CHECK: is_array
  fprintf(stderr, rtgTypeIsAArray(arrayTy) ? "is_array\n" : "isnot_array\n");
  // CHECK: i32
  mlirTypeDump(rtgArrayTypeGetElementType(arrayTy));
  // CHECK: !rtg.array<i32>
  mlirTypeDump(arrayTy);
}

static void testTupleType(MlirContext ctx) {
  MlirType i32Type = mlirIntegerTypeGet(ctx, 32);
  MlirType tupleTy = rtgTupleTypeGet(ctx, 2, (MlirType[]){i32Type, i32Type});

  // CHECK: is_tuple
  fprintf(stderr, rtgTypeIsATuple(tupleTy) ? "is_tuple\n" : "isnot_tuple\n");
  // CHECK: 2
  fprintf(stderr, "%ld\n", rtgTypeGetNumFields(tupleTy));
  // CHECK: i32
  mlirTypeDump(rtgTupleTypeGetFieldType(tupleTy, 1));
  // CHECK: !rtg.tuple<i32, i32>
  mlirTypeDump(tupleTy);

  MlirType emptyTupleTy = rtgTupleTypeGet(ctx, 0, NULL);
  // CHECK: is_tuple
  fprintf(stderr,
          rtgTypeIsATuple(emptyTupleTy) ? "is_tuple\n" : "isnot_tuple\n");
  // CHECK: !rtg.tuple
  mlirTypeDump(emptyTupleTy);
}

static void testLabelVisibilityAttr(MlirContext ctx) {
  MlirAttribute labelVisibility =
      rtgLabelVisibilityAttrGet(ctx, RTG_LABEL_VISIBILITY_GLOBAL);

  // CHECK: is_label_visibility
  fprintf(stderr, rtgAttrIsALabelVisibilityAttr(labelVisibility)
                      ? "is_label_visibility\n"
                      : "isnot_label_visibility\n");

  // CHECK: global_label
  switch (rtgLabelVisibilityAttrGetValue(labelVisibility)) {
  case RTG_LABEL_VISIBILITY_LOCAL:
    fprintf(stderr, "local_label\n");
    break;
  case RTG_LABEL_VISIBILITY_GLOBAL:
    fprintf(stderr, "global_label\n");
    break;
  case RTG_LABEL_VISIBILITY_EXTERNAL:
    fprintf(stderr, "external_label\n");
    break;
  }
}

static void testContextAttrs(MlirContext ctx) {
  MlirType cpuTy = rtgtestCPUTypeGet(ctx);

  // DefaultContext
  MlirAttribute defaultCtxtAttr = rtgDefaultContextAttrGet(ctx, cpuTy);
  // CHECK: is_default_context
  fprintf(stderr, rtgAttrIsADefaultContextAttr(defaultCtxtAttr)
                      ? "is_default_context\n"
                      : "isnot_default_context\n");
  // CHECK: !rtgtest.cpu
  mlirTypeDump(mlirAttributeGetType(defaultCtxtAttr));

  // AnyContext
  MlirAttribute anyCtxtAttr = rtgAnyContextAttrGet(ctx, cpuTy);
  // CHECK: is_any_context
  fprintf(stderr, rtgAttrIsAAnyContextAttr(anyCtxtAttr)
                      ? "is_any_context\n"
                      : "isnot_any_context\n");
  // CHECK: !rtgtest.cpu
  mlirTypeDump(mlirAttributeGetType(anyCtxtAttr));
}

static void testImmediate(MlirContext ctx) {
  MlirType immediateTy = rtgImmediateTypeGet(ctx, 32);
  // CHECK: is_immediate
  fprintf(stderr, rtgTypeIsAImmediate(immediateTy) ? "is_immediate\n"
                                                   : "isnot_immediate\n");
  // CHECK: !rtg.isa.immediate<32>
  mlirTypeDump(immediateTy);
  // CHECK: width=32
  fprintf(stderr, "width=%u\n", rtgImmediateTypeGetWidth(immediateTy));

  MlirAttribute immediateAttr = rtgImmediateAttrGet(ctx, 32, 42);
  // CHECK: is_immediate_attr
  fprintf(stderr, rtgAttrIsAImmediate(immediateAttr)
                      ? "is_immediate_attr\n"
                      : "isnot_immediate_attr\n");
  // CHECK: #rtg.isa.immediate<32, 42>
  mlirAttributeDump(immediateAttr);
  // CHECK: width=32
  fprintf(stderr, "width=%u\n", rtgImmediateAttrGetWidth(immediateAttr));
  // CHECK: value=42
  fprintf(stderr, "value=%" PRIu64 "\n",
          rtgImmediateAttrGetValue(immediateAttr));
}

static void testMemories(MlirContext ctx) {
  MlirType memoryBlockTy = rtgMemoryBlockTypeGet(ctx, 32);

  // CHECK: is_memory_block
  fprintf(stderr, rtgTypeIsAMemoryBlock(memoryBlockTy)
                      ? "is_memory_block\n"
                      : "isnot_memory_block\n");
  // CHECK: !rtg.isa.memory_block<32>
  mlirTypeDump(memoryBlockTy);
  // CHECK: address_width=32
  fprintf(stderr, "address_width=%u\n",
          rtgMemoryBlockTypeGetAddressWidth(memoryBlockTy));

  MlirType memoryTy = rtgMemoryTypeGet(ctx, 32);
  // CHECK: is_memory
  fprintf(stderr,
          rtgTypeIsAMemory(memoryTy) ? "is_memory\n" : "isnot_memory\n");
  // CHECK: addressWidth=32
  fprintf(stderr, "addressWidth=%u\n", rtgMemoryTypeGetAddressWidth(memoryTy));
  // CHECK: !rtg.isa.memory<32>
  mlirTypeDump(memoryTy);
}

int main(int argc, char **argv) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__rtg__(), ctx);
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__rtgtest__(), ctx);

  testSequenceType(ctx);
  testRandomizedSequenceType(ctx);
  testLabelType(ctx);
  testSetType(ctx);
  testBagType(ctx);
  testDictType(ctx);
  testArrayType(ctx);
  testTupleType(ctx);

  testLabelVisibilityAttr(ctx);
  testContextAttrs(ctx);
  testImmediate(ctx);
  testMemories(ctx);

  mlirContextDestroy(ctx);

  return 0;
}
