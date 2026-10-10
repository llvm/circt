//===- Sim.cpp - C interface for the Sim dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/Sim.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Sim/SimOps.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace circt;
using namespace circt::sim;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Sim, sim, SimDialect)

//===----------------------------------------------------------------------===//
// DPI function types.
//===----------------------------------------------------------------------===//

static DPIDirection unwrapDirection(SimDPIDirection direction) {
  switch (direction) {
  case SIM_DPI_DIRECTION_INPUT:
    return DPIDirection::Input;
  case SIM_DPI_DIRECTION_OUTPUT:
    return DPIDirection::Output;
  case SIM_DPI_DIRECTION_INOUT:
    return DPIDirection::InOut;
  case SIM_DPI_DIRECTION_RETURN:
    return DPIDirection::Return;
  case SIM_DPI_DIRECTION_REF:
    return DPIDirection::Ref;
  }
  llvm_unreachable("invalid DPI direction");
}

static SimDPIDirection wrapDirection(DPIDirection direction) {
  switch (direction) {
  case DPIDirection::Input:
    return SIM_DPI_DIRECTION_INPUT;
  case DPIDirection::Output:
    return SIM_DPI_DIRECTION_OUTPUT;
  case DPIDirection::InOut:
    return SIM_DPI_DIRECTION_INOUT;
  case DPIDirection::Return:
    return SIM_DPI_DIRECTION_RETURN;
  case DPIDirection::Ref:
    return SIM_DPI_DIRECTION_REF;
  }
  llvm_unreachable("invalid DPI direction");
}

bool simTypeIsADPIFunction(MlirType type) {
  return isa<DPIFunctionType>(unwrap(type));
}

MlirType simDPIFunctionTypeGet(MlirContext ctx, intptr_t numArguments,
                               const SimDPIArgument *arguments) {
  SmallVector<DPIArgument> args;
  for (intptr_t i = 0; i < numArguments; ++i)
    args.push_back(
        {mlir::StringAttr::get(unwrap(ctx), unwrap(arguments[i].name)),
         unwrap(arguments[i].type), unwrapDirection(arguments[i].direction)});
  return wrap(DPIFunctionType::get(unwrap(ctx), args));
}

intptr_t simDPIFunctionTypeGetNumArguments(MlirType type) {
  return cast<DPIFunctionType>(unwrap(type)).getNumArguments();
}

SimDPIArgument simDPIFunctionTypeGetArgument(MlirType type, intptr_t index) {
  auto args = cast<DPIFunctionType>(unwrap(type)).getArguments();
  assert(index >= 0 && static_cast<size_t>(index) < args.size());
  const auto &arg = args[index];
  return {wrap(arg.name.getValue()), wrap(arg.type), wrapDirection(arg.dir)};
}

MlirType simDPIFunctionTypeGetFunctionType(MlirType type) {
  return wrap(cast<DPIFunctionType>(unwrap(type)).getFunctionType());
}
