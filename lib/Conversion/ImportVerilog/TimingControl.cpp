//===- TimingControl.cpp - Slang timing control conversion ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "slang/ast/TimingControl.h"
#include "ImportVerilogInternals.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "mlir/IR/Diagnostics.h"
#include "slang/ast/ASTVisitor.h"

using namespace circt;
using namespace ImportVerilog;

LogicalResult Context::visitSignalEvent(
    const slang::ast::SignalEventControl *signalEventControl) {
  auto loc = convertLocation(signalEventControl->sourceRange.start());
  auto name = signalEventControl->expr.getSymbolReference()->name;
  builder.create<moore::EventControlOp>(
      loc, static_cast<moore::Edge>(signalEventControl->edge), name);
  return success();
}

LogicalResult Context::visitImplicitEvent(
    const slang::ast::ImplicitEventControl *implEventControl) {
  // Output a hint?
  return success();
}

LogicalResult
Context::visitTimingControl(const slang::ast::TimingControl *timingControl) {
  auto loc = convertLocation(timingControl->sourceRange.start());
  switch (timingControl->kind) {
  case slang::ast::TimingControlKind::Delay:
    return mlir::emitError(loc, "unsupported timing comtrol: delay control");
  case slang::ast::TimingControlKind::Delay3:
    return mlir::emitError(loc, "unsupported timing comtrol: delay3 control");
  case slang::ast::TimingControlKind::SignalEvent:
    return visitSignalEvent(
        &timingControl->as<slang::ast::SignalEventControl>());
  case slang::ast::TimingControlKind::EventList:
    for (auto *event : timingControl->as<slang::ast::EventListControl>().events)
      if (failed(visitTimingControl(event)))
        return failure();
    break;
  case slang::ast::TimingControlKind::ImplicitEvent:
    return visitImplicitEvent(
        &timingControl->as<slang::ast::ImplicitEventControl>());
  case slang::ast::TimingControlKind::RepeatedEvent:
    return mlir::emitError(
        loc, "unsupported timing comtrol: repeated event control");
  case slang::ast::TimingControlKind::OneStepDelay:
    return mlir::emitError(
        loc, "unsupported timing comtrol: one step delay control");
  case slang::ast::TimingControlKind::CycleDelay:
    return mlir::emitError(loc,
                           "unsupported timing comtrol: cycle delay control");
  case slang::ast::TimingControlKind::BlockEventList:
    return mlir::emitError(
        loc, "unsupported timing comtrol: block event list control");

  default:
    mlir::emitError(loc, "unsupported timing control");
    return failure();
  }
  return success();
}
