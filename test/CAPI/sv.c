//===- sv.c - SystemVerilog Dialect CAPI unit tests -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: circt-capi-sv-test 2>&1 | FileCheck %s

#include "circt-c/Dialect/SV.h"
#include "circt-c/ImportVerilog.h"
#include "llvm-c/Core.h"

#include <stdio.h>
#include <string.h>

int testImport(MlirContext ctx) {
  const char testSv[] = "             \
      module ExportTestSimpleModule(  \
          input [31:0] in_1,          \
          in_2,                       \
          output [31:0] out           \
      );                              \
        assign out = in_1 & in_2;     \
      endmodule";
  LLVMMemoryBufferRef buf = LLVMCreateMemoryBufferWithMemoryRange(
      testSv, strlen(testSv), "/tmp/testModule.sv", false);
  MlirModule module = mlirModuleCreateEmpty(mlirLocationUnknownGet(ctx));

  MlirLogicalResult result = mlirImportVerilog(ctx, module, &buf, 1, NULL);
  if (!mlirLogicalResultIsSuccess(result)) {
    return 1;
  }

  // clang-format off
  // CHECK:      module {
  // CHECK-NEXT:   moore.module @ExportTestSimpleModule(in %in_1 : !moore.l32, in %in_2 : !moore.l32, out out : !moore.l32) {
  // CHECK-NEXT:     %in_1_0 = moore.net name "in_1" wire : <l32>
  // CHECK-NEXT:     %in_2_1 = moore.net name "in_2" wire : <l32>
  // CHECK-NEXT:     %out = moore.net wire : <l32>
  // CHECK-NEXT:     %0 = moore.read %in_1_0 : <l32>
  // CHECK-NEXT:     %1 = moore.read %in_2_1 : <l32>
  // CHECK-NEXT:     %2 = moore.and %0, %1 : l32
  // CHECK-NEXT:     moore.assign %out, %2 : l32
  // CHECK-NEXT:     moore.assign %in_1_0, %in_1 : l32
  // CHECK-NEXT:     moore.assign %in_2_1, %in_2 : l32
  // CHECK-NEXT:     %3 = moore.read %out : <l32>
  // CHECK-NEXT:     moore.output %3 : !moore.l32
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // clang-format on
  mlirOperationDump(mlirModuleGetOperation(module));

  return 0;
}

int main(int argc, char **argv) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__sv__(), ctx);
  return testImport(ctx);
}
