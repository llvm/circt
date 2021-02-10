/*===- ir.c - Simple test of C APIs ---------------------------------------===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* RUN: circt-capi-ir-test 2>&1 | FileCheck %s
 */

#include "mlir-c/IR.h"
#include "circt-c/Dialect/RTL.h"
#include "mlir-c/AffineExpr.h"
#include "mlir-c/AffineMap.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/Registration.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int registerOnlyRTL() {
  MlirContext ctx = mlirContextCreate();
  // The built-in dialect is always loaded.
  if (mlirContextGetNumLoadedDialects(ctx) != 1)
    return 1;

  MlirDialect std =
      mlirContextGetOrLoadDialect(ctx, mlirRTLDialectGetNamespace());
  if (!mlirDialectIsNull(std))
    return 2;

  mlirContextRegisterRTLDialect(ctx);
  if (mlirContextGetNumRegisteredDialects(ctx) != 1)
    return 3;
  if (mlirContextGetNumLoadedDialects(ctx) != 1)
    return 4;

  std = mlirContextGetOrLoadDialect(ctx, mlirRTLDialectGetNamespace());
  if (mlirDialectIsNull(std))
    return 5;
  if (mlirContextGetNumLoadedDialects(ctx) != 2)
    return 6;

  MlirDialect alsoStd = mlirContextLoadRTLDialect(ctx);
  if (!mlirDialectEqual(std, alsoStd))
    return 7;

  MlirStringRef stdNs = mlirDialectGetNamespace(std);
  MlirStringRef alsoStdNs = mlirRTLDialectGetNamespace();
  if (stdNs.length != alsoStdNs.length ||
      strncmp(stdNs.data, alsoStdNs.data, stdNs.length))
    return 8;

  return 0;
}

int main() {
  fprintf(stderr, "@registration\n");
  int errcode = registerOnlyRTL();
  fprintf(stderr, "%d\n", errcode);
  // clang-format off
  // CHECK-LABEL: @registration
  // CHECK: 0
  // clang-format on

  return 0;
}
