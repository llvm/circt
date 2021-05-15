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
#include "circt-c/Dialect/HW.h"
#include "circt-c/Dialect/Seq.h"
#include "mlir-c/AffineExpr.h"
#include "mlir-c/AffineMap.h"
#include "mlir-c/BuiltinTypes.h"
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

  // RTL dialect tests.
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

  // Seq dialect tests.
  MlirDialectHandle seqHandle = mlirGetDialectHandle__seq__();
  mlirDialectHandleRegisterDialect(seqHandle, ctx);
  mlirDialectHandleLoadDialect(seqHandle, ctx);

  MlirDialect seq = mlirContextGetOrLoadDialect(
      ctx, mlirDialectHandleGetNamespace(seqHandle));
  if (mlirDialectIsNull(seq))
    return 8;

  MlirDialect alsoSeq = mlirDialectHandleLoadDialect(seqHandle, ctx);
  if (!mlirDialectEqual(seq, alsoSeq))
    return 9;

  registerSeqPasses();

  mlirContextDestroy(ctx);

  return 0;
}

int testRTLTypes() {
  MlirContext ctx = mlirContextCreate();
  MlirDialectHandle rtlHandle = mlirGetDialectHandle__rtl__();
  mlirDialectHandleRegisterDialect(rtlHandle, ctx);
  mlirDialectHandleLoadDialect(rtlHandle, ctx);

  MlirType i8type = mlirIntegerTypeGet(ctx, 8);
  MlirType io8type = rtlInOutTypeGet(i8type);
  if (mlirTypeIsNull(io8type))
    return 1;

  MlirType elementType = rtlInOutTypeGetElementType(io8type);
  if (mlirTypeIsNull(elementType))
    return 2;

  if (rtlTypeIsAInOut(i8type))
    return 3;

  if (!rtlTypeIsAInOut(io8type))
    return 4;

  mlirContextDestroy(ctx);

  return 0;
}

int main() {
  fprintf(stderr, "@registration\n");
  int errcode = registerOnlyRTL();
  fprintf(stderr, "%d\n", errcode);

  fprintf(stderr, "@rtltypes\n");
  errcode = testRTLTypes();
  fprintf(stderr, "%d\n", errcode);

  // clang-format off
  // CHECK-LABEL: @registration
  // CHECK: 0
  // CHECK-LABEL: @rtltypes
  // CHECK: 0
  // clang-format on

  return 0;
}
#endif
