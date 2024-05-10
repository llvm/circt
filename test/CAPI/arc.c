/*===- arc.c - Simple test of Arc Dialect C APIs  -------------------------===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* RUN: circt-capi-ir-test 2>&1 | FileCheck %s
 */

#include "circt-c/Dialect/Arc.h"

#include <stdio.h>
#include <string.h>

int registerOnlyArc(void) {
  MlirContext ctx = mlirContextCreate();
  // The built-in dialect is always loaded.
  if (mlirContextGetNumLoadedDialects(ctx) != 1)
    return 1;

  // Arc dialect tests.
  MlirDialectHandle arcHandle = mlirGetDialectHandle__arc__();

  MlirDialect arc = mlirContextGetOrLoadDialect(
      ctx, mlirDialectHandleGetNamespace(arcHandle));
  if (!mlirDialectIsNull(arc))
    return 2;

  mlirDialectHandleRegisterDialect(arcHandle, ctx);
  if (mlirContextGetNumRegisteredDialects(ctx) != 2)
    return 3;
  if (mlirContextGetNumLoadedDialects(ctx) != 1)
    return 4;

  arc = mlirContextGetOrLoadDialect(ctx,
                                    mlirDialectHandleGetNamespace(arcHandle));
  if (mlirDialectIsNull(arc))
    return 5;
  if (mlirContextGetNumLoadedDialects(ctx) != 2)
    return 6;

  MlirDialect alsoArc = mlirDialectHandleLoadDialect(arcHandle, ctx);
  if (!mlirDialectEqual(arc, alsoArc))
    return 7;

  registerArcPasses();

  mlirContextDestroy(ctx);

  return 0;
}

int main(void) {
  fprintf(stderr, "@registration\n");
  int errcode = registerOnlyArc();
  fprintf(stderr, "%d\n", errcode);

  // clang-format off
  // CHECK-LABEL: @registration
  // CHECK: 0
  // clang-format on

  return 0;
}
