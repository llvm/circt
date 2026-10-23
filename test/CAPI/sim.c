//===- sim.c - Sim Dialect C API tests ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: circt-capi-sim-test 2>&1 | FileCheck %s

#include "circt-c/Dialect/Sim.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Dialect/LLVM.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>

static void testEmptySignature(MlirContext ctx) {
  // CHECK-LABEL: Empty signature
  fprintf(stderr, "Empty signature\n");

  MlirType type = simDPIFunctionTypeGet(ctx, 0, NULL);
  // CHECK-NEXT: is DPI function: 1
  fprintf(stderr, "is DPI function: %d\n", simTypeIsADPIFunction(type));
  // CHECK-NEXT: arguments: 0
  fprintf(stderr, "arguments: %" PRIdPTR "\n",
          simDPIFunctionTypeGetNumArguments(type));
  // CHECK-NEXT: context matches: 1
  fprintf(stderr, "context matches: %d\n",
          mlirContextEqual(mlirTypeGetContext(type), ctx));
  // CHECK-NEXT: uniqued: 1
  fprintf(stderr, "uniqued: %d\n",
          mlirTypeEqual(type, simDPIFunctionTypeGet(ctx, 0, NULL)));
  // CHECK-NEXT: !sim.dpi_functy<>
  mlirTypeDump(type);

  MlirType callType = simDPIFunctionTypeGetFunctionType(type);
  // CHECK-NEXT: is function: 1
  fprintf(stderr, "is function: %d\n", mlirTypeIsAFunction(callType));
  // CHECK-NEXT: inputs: 0
  fprintf(stderr, "inputs: %" PRIdPTR "\n",
          mlirFunctionTypeGetNumInputs(callType));
  // CHECK-NEXT: results: 0
  fprintf(stderr, "results: %" PRIdPTR "\n",
          mlirFunctionTypeGetNumResults(callType));
  // CHECK-NEXT: call context matches: 1
  fprintf(stderr, "call context matches: %d\n",
          mlirContextEqual(mlirTypeGetContext(callType), ctx));
  // CHECK-NEXT: () -> ()
  mlirTypeDump(callType);
  // CHECK-NEXT: function is DPI function: 0
  fprintf(stderr, "function is DPI function: %d\n",
          simTypeIsADPIFunction(callType));
  // CHECK-NEXT: integer is DPI function: 0
  fprintf(stderr, "integer is DPI function: %d\n",
          simTypeIsADPIFunction(mlirIntegerTypeGet(ctx, 32)));
}

static void testArgumentsAndCallSignature(MlirContext ctx) {
  // CHECK-LABEL: Arguments and call signature
  fprintf(stderr, "Arguments and call signature\n");

  MlirType i7 = mlirIntegerTypeGet(ctx, 7);
  MlirType i32 = mlirIntegerTypeGet(ctx, 32);
  MlirType i64 = mlirIntegerTypeGet(ctx, 64);
  MlirType i1024 = mlirIntegerTypeGet(ctx, 1024);
  MlirType ptr = mlirLLVMPointerTypeGet(ctx, 0);
  SimDPIArgument arguments[] = {
      {mlirStringRefCreateFromCString("value"), i7, SIM_DPI_DIRECTION_OUTPUT},
      {mlirStringRefCreateFromCString("cycle"), i32, SIM_DPI_DIRECTION_INPUT},
      {mlirStringRefCreateFromCString("state"), i1024, SIM_DPI_DIRECTION_INOUT},
      {mlirStringRefCreateFromCString("data"), ptr, SIM_DPI_DIRECTION_REF},
      {mlirStringRefCreateFromCString("status"), i64, SIM_DPI_DIRECTION_RETURN},
  };
  intptr_t count = sizeof(arguments) / sizeof(arguments[0]);
  MlirType type = simDPIFunctionTypeGet(ctx, count, arguments);
  // CHECK-NEXT: is DPI function: 1
  fprintf(stderr, "is DPI function: %d\n", simTypeIsADPIFunction(type));
  // CHECK-NEXT: arguments: 5
  fprintf(stderr, "arguments: %" PRIdPTR "\n",
          simDPIFunctionTypeGetNumArguments(type));

  // CHECK-NEXT: argument 0: name matches 1, type matches 1, direction matches 1
  // CHECK-NEXT: argument 1: name matches 1, type matches 1, direction matches 1
  // CHECK-NEXT: argument 2: name matches 1, type matches 1, direction matches 1
  // CHECK-NEXT: argument 3: name matches 1, type matches 1, direction matches 1
  // CHECK-NEXT: argument 4: name matches 1, type matches 1, direction matches 1
  for (intptr_t i = 0; i < count; ++i) {
    SimDPIArgument argument = simDPIFunctionTypeGetArgument(type, i);
    fprintf(stderr,
            "argument %" PRIdPTR
            ": name matches %d, type matches %d, direction matches %d\n",
            i, mlirStringRefEqual(argument.name, arguments[i].name),
            mlirTypeEqual(argument.type, arguments[i].type),
            argument.direction == arguments[i].direction);
  }

  // CHECK-NEXT: !sim.dpi_functy<out "value" : i7, in "cycle" : i32,
  // CHECK-SAME: inout "state" : i1024, ref "data" : !llvm.ptr,
  // CHECK-SAME: return "status" : i64>
  mlirTypeDump(type);

  MlirType callType = simDPIFunctionTypeGetFunctionType(type);
  // CHECK-NEXT: is function: 1
  fprintf(stderr, "is function: %d\n", mlirTypeIsAFunction(callType));
  // CHECK-NEXT: inputs: 3
  fprintf(stderr, "inputs: %" PRIdPTR "\n",
          mlirFunctionTypeGetNumInputs(callType));
  // CHECK-NEXT: results: 3
  fprintf(stderr, "results: %" PRIdPTR "\n",
          mlirFunctionTypeGetNumResults(callType));

  // CHECK-NEXT: input 0: i32
  // CHECK-NEXT: result 0: i7
  // CHECK-NEXT: input 1: i1024
  // CHECK-NEXT: result 1: i1024
  // CHECK-NEXT: input 2: !llvm.ptr
  // CHECK-NEXT: result 2: i64
  for (intptr_t i = 0; i < 3; ++i) {
    fprintf(stderr, "input %" PRIdPTR ": ", i);
    mlirTypeDump(mlirFunctionTypeGetInput(callType, i));
    fprintf(stderr, "result %" PRIdPTR ": ", i);
    mlirTypeDump(mlirFunctionTypeGetResult(callType, i));
  }

  // CHECK-NEXT: (i32, i1024, !llvm.ptr) -> (i7, i1024, i64)
  mlirTypeDump(callType);
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  MlirDialect sim =
      mlirDialectHandleLoadDialect(mlirGetDialectHandle__sim__(), ctx);
  MlirDialect llvm =
      mlirDialectHandleLoadDialect(mlirGetDialectHandle__llvm__(), ctx);
  assert(!mlirDialectIsNull(sim));
  assert(!mlirDialectIsNull(llvm));

  testEmptySignature(ctx);
  testArgumentsAndCallSignature(ctx);

  mlirContextDestroy(ctx);
  return 0;
}
