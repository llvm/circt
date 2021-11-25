//===- InitAllDialects.h - CIRCT Dialects Registration ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all dialects and
// passes to the system.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_INITALLDIALECTS_H_
#define CIRCT_INITALLDIALECTS_H_

#include "circt/Dialect/APInt/APIntDialect.h"
#include "circt/Dialect/Calyx/CalyxDialect.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/StaticLogic/StaticLogic.h"
#include "mlir/IR/Dialect.h"

namespace circt {

// Add all the MLIR dialects to the provided registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<
    apint::APIntDialect,
    calyx::CalyxDialect,
    comb::CombDialect,
    esi::ESIDialect,
    firrtl::FIRRTLDialect,
    fsm::FSMDialect,
    handshake::HandshakeDialect,
    llhd::LLHDDialect,
    msft::MSFTDialect,
    hw::HWDialect,
    seq::SeqDialect,
    staticlogic::StaticLogicDialect,
    sv::SVDialect
  >();
  // clang-format on
}

} // namespace circt

#endif // CIRCT_INITALLDIALECTS_H_
