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

#if defined(_WIN32) || defined(__CYGWIN__)
/// The C API is currently not supported on Windows
/// (https://github.com/llvm/circt/issues/578)
int main() { return 0; }
#else
int registerOnlyRTL() {
  MlirContext ctx = mlirContextCreate();
  // The built-in dialect is always loaded.
  if (mlirContextGetNumLoadedDialects(ctx) != 1)
    return 1;

  MlirDialectHandle rtlHandle = mlirGetDialectHandle__rtl__();

  MlirDialect rtl = mlirContextGetOrLoadDialect(
      ctx, mlirDialectHandleGetNamespace(rtlHandle));
  if (!mlirDialectIsNull(rtl))
    return 2;

  mlirDialectHandleRegisterDialect(rtlHandle, ctx);
  if (mlirContextGetNumRegisteredDialects(ctx) != 1)
    return 3;
  if (mlirContextGetNumLoadedDialects(ctx) != 1)
    return 4;

  rtl = mlirContextGetOrLoadDialect(ctx,
                                    mlirDialectHandleGetNamespace(rtlHandle));
  if (mlirDialectIsNull(rtl))
    return 5;
  if (mlirContextGetNumLoadedDialects(ctx) != 2)
    return 6;

  MlirDialect alsoRtl = mlirDialectHandleLoadDialect(rtlHandle, ctx);
  if (!mlirDialectEqual(rtl, alsoRtl))
    return 7;

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
#endif
