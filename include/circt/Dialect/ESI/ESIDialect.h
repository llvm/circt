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

/// Name of dialect attribute which governs whether or not to bundle (i.e. use
/// SystemVerilog interfaces) channel signal wires on external modules.
constexpr StringRef extModBundleSignalsAttrName = "esi.bundle";

/// Name of dialect attribute which governs whether or not to flatten struct
/// ports into a bunch of individual 'data' wires.
constexpr StringRef extModPortFlattenStructsAttrName = "esi.portFlattenStructs";

/// Suffix _all_ lowered input ports with this suffix. Defaults to nothing.
constexpr StringRef extModPortInSuffix = "esi.portInSuffix";
/// Suffix _all_ lowered output ports with this suffix. Defaults to nothing.
constexpr StringRef extModPortOutSuffix = "esi.portOutSuffix";
/// Suffix lowered valid ports with this suffix. Defaults to "_valid". Applies
/// only to ValidReady channels.
constexpr StringRef extModPortValidSuffix = "esi.portValidSuffix";
/// Suffix lowered ready ports with this suffix. Defaults to "_ready". Applies
/// only to ValidReady channels.
constexpr StringRef extModPortReadySuffix = "esi.portReadySuffix";
/// Suffix lowered read enable ports with this suffix. Defaults to "_rden".
/// Applies only to FIFO channels.
constexpr StringRef extModPortRdenSuffix = "esi.portRdenSuffix";
/// Suffix lowered empty ports with this suffix. Defaults to "_empty". Applies
/// only to FIFO channels.
constexpr StringRef extModPortEmptySuffix = "esi.portEmptySuffix";

} // namespace esi
} // namespace circt

#include "circt/Dialect/ESI/ESIDialect.h.inc"

// Pull in all enum type definitions and utility function declarations.
#include "circt/Dialect/ESI/ESIEnums.h.inc"

#endif
