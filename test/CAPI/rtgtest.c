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

static void testIntegerRegisterType(MlirContext ctx) {
  MlirType iregTy = rtgtestIntegerRegisterTypeGet(ctx);

  // CHECK: is_ireg
  fprintf(stderr,
          rtgtestTypeIsAIntegerRegister(iregTy) ? "is_ireg\n" : "isnot_ireg\n");
  // CHECK: !rtgtest.ireg
  mlirTypeDump(iregTy);
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

static void testRegisters(MlirContext ctx) {
  {
    MlirAttribute regAttr = rtgtestRegZeroAttrGet(ctx);
    // CHECK: is_zero
    fprintf(stderr,
            rtgtestAttrIsARegZero(regAttr) ? "is_zero\n" : "isnot_zero\n");
    // CHECK: #rtgtest.zero
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegRaAttrGet(ctx);
    // CHECK: is_ra
    fprintf(stderr, rtgtestAttrIsARegRa(regAttr) ? "is_ra\n" : "isnot_ra\n");
    // CHECK: #rtgtest.ra
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegSpAttrGet(ctx);
    // CHECK: is_sp
    fprintf(stderr, rtgtestAttrIsARegSp(regAttr) ? "is_sp\n" : "isnot_sp\n");
    // CHECK: #rtgtest.sp
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegGpAttrGet(ctx);
    // CHECK: is_gp
    fprintf(stderr, rtgtestAttrIsARegGp(regAttr) ? "is_gp\n" : "isnot_gp\n");
    // CHECK: #rtgtest.gp
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegTpAttrGet(ctx);
    // CHECK: is_tp
    fprintf(stderr, rtgtestAttrIsARegTp(regAttr) ? "is_tp\n" : "isnot_tp\n");
    // CHECK: #rtgtest.tp
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegT0AttrGet(ctx);
    // CHECK: is_t0
    fprintf(stderr, rtgtestAttrIsARegT0(regAttr) ? "is_t0\n" : "isnot_t0\n");
    // CHECK: #rtgtest.t0
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegT1AttrGet(ctx);
    // CHECK: is_t1
    fprintf(stderr, rtgtestAttrIsARegT1(regAttr) ? "is_t1\n" : "isnot_t1\n");
    // CHECK: #rtgtest.t1
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegT2AttrGet(ctx);
    // CHECK: is_t2
    fprintf(stderr, rtgtestAttrIsARegT2(regAttr) ? "is_t2\n" : "isnot_t2\n");
    // CHECK: #rtgtest.t2
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegS0AttrGet(ctx);
    // CHECK: is_s0
    fprintf(stderr, rtgtestAttrIsARegS0(regAttr) ? "is_s0\n" : "isnot_s0\n");
    // CHECK: #rtgtest.s0
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegS1AttrGet(ctx);
    // CHECK: is_s1
    fprintf(stderr, rtgtestAttrIsARegS1(regAttr) ? "is_s1\n" : "isnot_s1\n");
    // CHECK: #rtgtest.s1
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegA0AttrGet(ctx);
    // CHECK: is_a0
    fprintf(stderr, rtgtestAttrIsARegA0(regAttr) ? "is_a0\n" : "isnot_a0\n");
    // CHECK: #rtgtest.a0
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegA1AttrGet(ctx);
    // CHECK: is_a1
    fprintf(stderr, rtgtestAttrIsARegA1(regAttr) ? "is_a1\n" : "isnot_a1\n");
    // CHECK: #rtgtest.a1
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegA2AttrGet(ctx);
    // CHECK: is_a2
    fprintf(stderr, rtgtestAttrIsARegA2(regAttr) ? "is_a2\n" : "isnot_a2\n");
    // CHECK: #rtgtest.a2
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegA3AttrGet(ctx);
    // CHECK: is_a3
    fprintf(stderr, rtgtestAttrIsARegA3(regAttr) ? "is_a3\n" : "isnot_a3\n");
    // CHECK: #rtgtest.a3
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegA4AttrGet(ctx);
    // CHECK: is_a4
    fprintf(stderr, rtgtestAttrIsARegA4(regAttr) ? "is_a4\n" : "isnot_a4\n");
    // CHECK: #rtgtest.a4
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegA5AttrGet(ctx);
    // CHECK: is_a5
    fprintf(stderr, rtgtestAttrIsARegA5(regAttr) ? "is_a5\n" : "isnot_a5\n");
    // CHECK: #rtgtest.a5
    mlirAttributeDump(regAttr);
  }
}

int main(int argc, char **argv) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__rtgtest__(), ctx);

  testCPUType(ctx);
  testIntegerRegisterType(ctx);
  testCPUAttr(ctx);
  testRegisters(ctx);

  mlirContextDestroy(ctx);

  return 0;
}
