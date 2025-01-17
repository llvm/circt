//===- rtgtest.c - RTGTest Dialect CAPI unit tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: circt-capi-rtgtest-test 2>&1 | FileCheck %s

#include <stdio.h>

#include "circt-c/Dialect/RTGTest.h"

static void testCPUType(MlirContext ctx) {
  MlirType cpuTy = rtgtestCPUTypeGet(ctx);

  // CHECK: is_cpu
  fprintf(stderr, rtgtestTypeIsACPU(cpuTy) ? "is_cpu\n" : "isnot_cpu\n");
  // CHECK: !rtgtest.cpu
  mlirTypeDump(cpuTy);
}

static void testCPUAttr(MlirContext ctx) {
  MlirAttribute cpuAttr = rtgtestCPUAttrGet(ctx, 3);

  // CHECK: is_cpu
  fprintf(stderr, rtgtestAttrIsACPU(cpuAttr) ? "is_cpu\n" : "isnot_cpu\n");
  // CHECK: 3
  fprintf(stderr, "%u\n", rtgtestCPUAttrGetId(cpuAttr));
  // CHECK: #rtgtest.cpu<3>
  mlirAttributeDump(cpuAttr);
}

int main(int argc, char **argv) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__rtgtest__(), ctx);

  testCPUType(ctx);
  testCPUAttr(ctx);

  mlirContextDestroy(ctx);

  return 0;
}
