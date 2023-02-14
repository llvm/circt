//===- ESIDialect.h - ESI dialect Dialect class -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The Elastic Silicon Interconnect (ESI) dialect
//
// ESI is a system interconnect generator. It is type safe and
// latency-insensitive. It can be used for on-chip, inter-chip, and host-chip
// communication. It is also intended to help with incremental adoption and
// integration with existing RTL as it provides a standardized, typed interface
// to the outside world.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_ESIDIALECT_H
#define CIRCT_DIALECT_ESI_ESIDIALECT_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

namespace circt {
namespace esi {

void registerESIPasses();
void registerESITranslations();
LogicalResult exportCosimSchema(ModuleOp module, llvm::raw_ostream &os);

/// A triple of signals which represent a latency insensitive interface with
/// valid/ready semantics.
struct ESIPortValidReadyMapping {
  hw::PortInfo data, valid, ready;
};

/// Name of dialect attribute which governs whether or not to bundle (i.e. use
/// SystemVerilog interfaces) channel signal wires on external modules.
constexpr StringRef extModBundleSignalsAttrName = "esi.bundle";

/// Find all the port triples on a module which fit the
/// <name>/<name>_valid/<name>_ready pattern. Ready must be the opposite
/// direction of the other two.
void findValidReadySignals(Operation *modOp,
                           SmallVectorImpl<ESIPortValidReadyMapping> &names);

/// Given a list of logical port names, find the data/valid/ready port triples.
void resolvePortNames(Operation *modOp, ArrayRef<StringRef> portNames,
                      SmallVectorImpl<ESIPortValidReadyMapping> &portTriples);

/// Build an ESI module wrapper, converting the wires with latency-insensitive
/// semantics to ESI channels and passing through the rest.
Operation *buildESIWrapper(OpBuilder &b, Operation *mod,
                           ArrayRef<ESIPortValidReadyMapping> esiPortNames);

} // namespace esi
} // namespace circt

#include "circt/Dialect/ESI/ESIDialect.h.inc"

// Pull in all enum type definitions and utility function declarations.
#include "circt/Dialect/ESI/ESIEnums.h.inc"

#endif
