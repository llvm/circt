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
  {
    MlirAttribute regAttr = rtgtestRegA6AttrGet(ctx);
    // CHECK: is_a6
    fprintf(stderr, rtgtestAttrIsARegA6(regAttr) ? "is_a6\n" : "isnot_a6\n");
    // CHECK: #rtgtest.a6
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegA7AttrGet(ctx);
    // CHECK: is_a7
    fprintf(stderr, rtgtestAttrIsARegA7(regAttr) ? "is_a7\n" : "isnot_a7\n");
    // CHECK: #rtgtest.a7
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegS2AttrGet(ctx);
    // CHECK: is_s2
    fprintf(stderr, rtgtestAttrIsARegS2(regAttr) ? "is_s2\n" : "isnot_s2\n");
    // CHECK: #rtgtest.s2
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegS3AttrGet(ctx);
    // CHECK: is_s3
    fprintf(stderr, rtgtestAttrIsARegS3(regAttr) ? "is_s3\n" : "isnot_s3\n");
    // CHECK: #rtgtest.s3
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegS4AttrGet(ctx);
    // CHECK: is_s4
    fprintf(stderr, rtgtestAttrIsARegS4(regAttr) ? "is_s4\n" : "isnot_s4\n");
    // CHECK: #rtgtest.s4
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegS5AttrGet(ctx);
    // CHECK: is_s5
    fprintf(stderr, rtgtestAttrIsARegS5(regAttr) ? "is_s5\n" : "isnot_s5\n");
    // CHECK: #rtgtest.s5
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegS6AttrGet(ctx);
    // CHECK: is_s6
    fprintf(stderr, rtgtestAttrIsARegS6(regAttr) ? "is_s6\n" : "isnot_s6\n");
    // CHECK: #rtgtest.s6
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegS7AttrGet(ctx);
    // CHECK: is_s7
    fprintf(stderr, rtgtestAttrIsARegS7(regAttr) ? "is_s7\n" : "isnot_s7\n");
    // CHECK: #rtgtest.s7
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegS8AttrGet(ctx);
    // CHECK: is_s8
    fprintf(stderr, rtgtestAttrIsARegS8(regAttr) ? "is_s8\n" : "isnot_s8\n");
    // CHECK: #rtgtest.s8
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegS9AttrGet(ctx);
    // CHECK: is_s9
    fprintf(stderr, rtgtestAttrIsARegS9(regAttr) ? "is_s9\n" : "isnot_s9\n");
    // CHECK: #rtgtest.s9
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegS10AttrGet(ctx);
    // CHECK: is_s10
    fprintf(stderr, rtgtestAttrIsARegS10(regAttr) ? "is_s10\n" : "isnot_s10\n");
    // CHECK: #rtgtest.s10
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegS11AttrGet(ctx);
    // CHECK: is_s11
    fprintf(stderr, rtgtestAttrIsARegS11(regAttr) ? "is_s11\n" : "isnot_s11\n");
    // CHECK: #rtgtest.s11
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegT3AttrGet(ctx);
    // CHECK: is_t3
    fprintf(stderr, rtgtestAttrIsARegT3(regAttr) ? "is_t3\n" : "isnot_t3\n");
    // CHECK: #rtgtest.t3
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegT4AttrGet(ctx);
    // CHECK: is_t4
    fprintf(stderr, rtgtestAttrIsARegT4(regAttr) ? "is_t4\n" : "isnot_t4\n");
    // CHECK: #rtgtest.t4
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegT5AttrGet(ctx);
    // CHECK: is_t5
    fprintf(stderr, rtgtestAttrIsARegT5(regAttr) ? "is_t5\n" : "isnot_t5\n");
    // CHECK: #rtgtest.t5
    mlirAttributeDump(regAttr);
  }
  {
    MlirAttribute regAttr = rtgtestRegT6AttrGet(ctx);
    // CHECK: is_t6
    fprintf(stderr, rtgtestAttrIsARegT6(regAttr) ? "is_t6\n" : "isnot_t6\n");
    // CHECK: #rtgtest.t6
    mlirAttributeDump(regAttr);
  }
}

static void testImmediates(MlirContext ctx) {
  MlirType imm12Type = rtgtestImm12TypeGet(ctx);
  // CHECK: is_imm12
  fprintf(stderr,
          rtgtestTypeIsAImm12(imm12Type) ? "is_imm12\n" : "isnot_imm12\n");
  // CHECK: !rtgtest.imm12
  mlirTypeDump(imm12Type);

  MlirType imm21Type = rtgtestImm21TypeGet(ctx);
  // CHECK: is_imm21
  fprintf(stderr,
          rtgtestTypeIsAImm21(imm21Type) ? "is_imm21\n" : "isnot_imm21\n");
  // CHECK: !rtgtest.imm21
  mlirTypeDump(imm21Type);

  MlirType imm32Type = rtgtestImm32TypeGet(ctx);
  // CHECK: is_imm32
  fprintf(stderr,
          rtgtestTypeIsAImm32(imm32Type) ? "is_imm32\n" : "isnot_imm32\n");
  // CHECK: !rtgtest.imm32
  mlirTypeDump(imm32Type);

  MlirAttribute imm12Attr = rtgtestImm12AttrGet(ctx, 3);
  // CHECK: is_imm12
  fprintf(stderr,
          rtgtestAttrIsAImm12(imm12Attr) ? "is_imm12\n" : "isnot_imm12\n");
  // CHECK: 3
  fprintf(stderr, "%u\n", rtgtestImm12AttrGetValue(imm12Attr));
  // CHECK: #rtgtest.imm12<3>
  mlirAttributeDump(imm12Attr);

  MlirAttribute imm21Attr = rtgtestImm21AttrGet(ctx, 3);
  // CHECK: is_imm21
  fprintf(stderr,
          rtgtestAttrIsAImm21(imm21Attr) ? "is_imm21\n" : "isnot_imm21\n");
  // CHECK: 3
  fprintf(stderr, "%u\n", rtgtestImm21AttrGetValue(imm21Attr));
  // CHECK: #rtgtest.imm21<3>
  mlirAttributeDump(imm21Attr);

  MlirAttribute imm32Attr = rtgtestImm32AttrGet(ctx, 3);
  // CHECK: is_imm32
  fprintf(stderr,
          rtgtestAttrIsAImm32(imm32Attr) ? "is_imm32\n" : "isnot_imm32\n");
  // CHECK: 3
  fprintf(stderr, "%u\n", rtgtestImm32AttrGetValue(imm32Attr));
  // CHECK: #rtgtest.imm32<3>
  mlirAttributeDump(imm32Attr);
}

int main(int argc, char **argv) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__rtgtest__(), ctx);

  testCPUType(ctx);
  testCPUAttr(ctx);
  testRegisters(ctx);
  testImmediates(ctx);

  mlirContextDestroy(ctx);

  return 0;
}
