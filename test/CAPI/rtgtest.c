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

int main(int argc, char **argv) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__rtgtest__(), ctx);

  testCPUType(ctx);

  mlirContextDestroy(ctx);

  return 0;
}
