//===- ESIDialect.cpp - ESI dialect code defs -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dialect definitions. Should be relatively standard boilerplate.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/RTL/RTLOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/Support/FormatVariadic.h"

using namespace circt;
using namespace circt::esi;

ESIDialect::ESIDialect(MLIRContext *context)
    : Dialect("esi", context, TypeID::get<ESIDialect>()) {

  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/ESI/ESI.cpp.inc"
      >();
}

/// Find all the port triples on a module which fit the
/// <name>/<name>_valid/<name>_ready pattern. Ready must be the opposite
/// direction of the other two.
void findValidReadySignals(rtl::RTLModuleOp *mod,
                           SmallVectorImpl<rtl::ModulePortInfo> &names) {
  SmallVector<rtl::ModulePortInfo, 64> ports;
  mod->getPortInfo(ports);

  llvm::StringMap<rtl::ModulePortInfo> nameMap(ports.size());
  for (auto port : ports)
    nameMap[port.getName()] = port;

  for (auto port : ports) {
    if (port.direction == rtl::PortDirection::INOUT)
      continue;

    // Try to find a corresponding 'valid' port.
    SmallString<64> portName = port.getName();
    portName.append("_valid");
    auto valid = nameMap.find(portName);
    if (valid == nameMap.end() || valid->second.direction != port.direction ||
        !valid->second.type.isSignlessInteger(1))
      continue;

    // Try to find a corresponding 'ready' port.
    portName = port.getName();
    portName.append("_ready");
    rtl::PortDirection readyDir = port.direction == rtl::PortDirection::INPUT
                                      ? rtl::PortDirection::OUTPUT
                                      : rtl::PortDirection::INPUT;
    auto ready = nameMap.find(portName);
    if (ready == nameMap.end() || ready->second.direction != readyDir ||
        !valid->second.type.isSignlessInteger(1))
      continue;

    // Found one.
    names.push_back(port);
  }
}

/// Convert a port triple (same pattern as above) to an ESI channel. Does not
/// modify any instances which might exist.
Value convertPortToChannel(rtl::RTLModuleOp *mod, StringRef name);

#include "circt/Dialect/ESI/ESIAttrs.cpp.inc"
