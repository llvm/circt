//===- ProbeDialect.cpp - Probe dialect implementation --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Probe/ProbeDialect.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Probe/ProbeOps.h"
#include "circt/Dialect/Probe/ProbeTypes.h"

using namespace circt;
using namespace mlir;
using namespace probe;

namespace {
struct ProbeHWModulePortTypeInterface : public hw::HWModulePortTypeInterface {
  using HWModulePortTypeInterface::HWModulePortTypeInterface;

  LogicalResult
  verifyHWModulePortType(function_ref<InFlightDiagnostic()> emitError,
                         hw::ModulePort::Direction direction,
                         Type type) const override {
    if (!isa<probe::RefType>(type) || direction == hw::ModulePort::Output)
      return success();
    return emitError() << "input probe refs are not supported";
  }
};
} // namespace

void ProbeDialect::initialize() {
  registerOps();
  registerTypes();
  addInterfaces<ProbeHWModulePortTypeInterface>();
}

// Dialect implementation generated from `ProbeDialect.td`
#include "circt/Dialect/Probe/ProbeDialect.cpp.inc"
