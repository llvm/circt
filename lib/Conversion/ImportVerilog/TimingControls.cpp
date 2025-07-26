//===- TimingControl.cpp - Slang timing control conversion ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/TimingControl.h"
#include "llvm/ADT/ScopeExit.h"

using namespace circt;
using namespace ImportVerilog;

static ltl::ClockEdge convertEdgeKindLTL(const slang::ast::EdgeKind edge) {
  using slang::ast::EdgeKind;
  switch (edge) {
  case EdgeKind::NegEdge:
    return ltl::ClockEdge::Neg;
  case EdgeKind::PosEdge:
    return ltl::ClockEdge::Pos;
  case EdgeKind::None:
    // TODO: SV 16.16, what to do when no edge is specified?
    // For now, assume all changes (two-valued should be the same as both
    // edges)
  case EdgeKind::BothEdges:
    return ltl::ClockEdge::Both;
  }
  llvm_unreachable("all edge kinds handled");
}

static moore::Edge convertEdgeKind(const slang::ast::EdgeKind edge) {
  using slang::ast::EdgeKind;
  switch (edge) {
  case EdgeKind::None:
    return moore::Edge::AnyChange;
  case EdgeKind::PosEdge:
    return moore::Edge::PosEdge;
  case EdgeKind::NegEdge:
    return moore::Edge::NegEdge;
  case EdgeKind::BothEdges:
    return moore::Edge::BothEdges;
  }
  llvm_unreachable("all edge kinds handled");
}

// NOLINTBEGIN(misc-no-recursion)
namespace {

// Handle any of the event control constructs.
struct EventControlVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  // Handle single signal events like `posedge x`, `negedge y iff z`, or `w`.
  LogicalResult visit(const slang::ast::SignalEventControl &ctrl) {
    auto edge = convertEdgeKind(ctrl.edge);
    auto expr = context.convertRvalueExpression(ctrl.expr);
    if (!expr)
      return failure();
    Value condition;
    if (ctrl.iffCondition) {
      condition = context.convertRvalueExpression(*ctrl.iffCondition);
      condition = context.convertToBool(condition, Domain::TwoValued);
      if (!condition)
        return failure();
    }
    moore::DetectEventOp::create(builder, loc, edge, expr, condition);
    return success();
  }

  // Handle a list of signal events.
  LogicalResult visit(const slang::ast::EventListControl &ctrl) {
    for (const auto *event : ctrl.events) {
      auto visitor = *this;
      visitor.loc = context.convertLocation(event->sourceRange);
      if (failed(event->visit(visitor)))
        return failure();
    }
    return success();
  }

  // Emit an error for all other timing controls.
  template <typename T>
  LogicalResult visit(T &&ctrl) {
    return mlir::emitError(loc)
           << "unsupported event control: " << slang::ast::toString(ctrl.kind);
  }
};

// Handle any of the delay control constructs.
struct DelayControlVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  // Emit an error for all other timing controls.
  template <typename T>
  LogicalResult visit(T &&ctrl) {
    return mlir::emitError(loc)
           << "unsupported delay control: " << slang::ast::toString(ctrl.kind);
  }
};

struct LTLClockControlVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;
  Value seqOrPro;

  Value visit(const slang::ast::SignalEventControl &ctrl) {
    auto edge = convertEdgeKindLTL(ctrl.edge);
    auto expr = context.convertRvalueExpression(ctrl.expr);
    if (!expr)
      return Value{};
    Value condition;
    if (ctrl.iffCondition) {
      condition = context.convertRvalueExpression(*ctrl.iffCondition);
      condition = context.convertToBool(condition, Domain::TwoValued);
      if (!condition)
        return Value{};
    }
    expr = context.convertToI1(expr);
    if (!expr)
      return Value{};
    return ltl::ClockOp::create(builder, loc, seqOrPro, edge, expr);
  }

  template <typename T>
  Value visit(T &&ctrl) {
    mlir::emitError(loc, "unsupported LTL clock control: ")
        << slang::ast::toString(ctrl.kind);
    return Value{};
  }
};

} // namespace

// Entry point to timing control handling. This deals with the layer of repeats
// that a timing control may be wrapped in, and also handles the implicit event
// control which may appear at that point. For any event control a `WaitEventOp`
// will be created and populated by `handleEventControl`. Any delay control will
// be handled by `handleDelayControl`.
static LogicalResult handleRoot(Context &context,
                                const slang::ast::TimingControl &ctrl,
                                moore::WaitEventOp &implicitWaitOp) {
  auto &builder = context.builder;
  auto loc = context.convertLocation(ctrl.sourceRange);

  using slang::ast::TimingControlKind;
  switch (ctrl.kind) {
    // TODO: Actually implement a lowering for repeated event control. The main
    // way to trigger this is through an intra-assignment timing control, which
    // is not yet supported:
    //
    //   a = repeat(3) @(posedge b) c;
    //
    // This will want to recursively call this function at the right insertion
    // point to handle the timing control being repeated.
  case TimingControlKind::RepeatedEvent:
    return mlir::emitError(loc) << "unsupported repeated event control";

    // Handle implicit events, i.e. `@*` and `@(*)`. This implicitly includes
    // all variables read within the statement that follows after the event
    // control. Since we haven't converted that statement yet, simply create and
    // empty wait op and let `Context::convertTimingControl` populate it once
    // the statement has been lowered.
  case TimingControlKind::ImplicitEvent:
    implicitWaitOp = moore::WaitEventOp::create(builder, loc);
    return success();

    // Handle event control.
  case TimingControlKind::SignalEvent:
  case TimingControlKind::EventList: {
    auto waitOp = moore::WaitEventOp::create(builder, loc);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&waitOp.getBody().emplaceBlock());
    EventControlVisitor visitor{context, loc, builder};
    return ctrl.visit(visitor);
  }

    // Handle delay control.
  case TimingControlKind::Delay:
  case TimingControlKind::Delay3:
  case TimingControlKind::OneStepDelay:
  case TimingControlKind::CycleDelay: {
    DelayControlVisitor visitor{context, loc, builder};
    return ctrl.visit(visitor);
  }

  default:
    return mlir::emitError(loc, "unsupported timing control: ")
           << slang::ast::toString(ctrl.kind);
  }
}

LogicalResult
Context::convertTimingControl(const slang::ast::TimingControl &ctrl,
                              const slang::ast::Statement &stmt) {
  // Convert the timing control. Implicit event control will create a new empty
  // `WaitEventOp` and assign it to `implicitWaitOp`. This op will be populated
  // further down.
  moore::WaitEventOp implicitWaitOp;
  {
    auto previousCallback = rvalueReadCallback;
    auto done =
        llvm::make_scope_exit([&] { rvalueReadCallback = previousCallback; });
    // Reads happening as part of the event control should not be added to a
    // surrounding implicit event control's list of implicitly observed
    // variables.
    rvalueReadCallback = nullptr;
    if (failed(handleRoot(*this, ctrl, implicitWaitOp)))
      return failure();
  }

  // Convert the statement. In case `implicitWaitOp` is set, we register a
  // callback to collect all the variables read by the statement into
  // `readValues`, such that we can populate the op with implicitly observed
  // variables afterwards.
  llvm::SmallSetVector<Value, 8> readValues;
  {
    auto previousCallback = rvalueReadCallback;
    auto done =
        llvm::make_scope_exit([&] { rvalueReadCallback = previousCallback; });
    if (implicitWaitOp) {
      rvalueReadCallback = [&](moore::ReadOp readOp) {
        readValues.insert(readOp.getInput());
        if (previousCallback)
          previousCallback(readOp);
      };
    }
    if (failed(convertStatement(stmt)))
      return failure();
  }

  // Populate the implicit wait op with reads from the variables read by the
  // statement.
  if (implicitWaitOp) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&implicitWaitOp.getBody().emplaceBlock());
    for (auto readValue : readValues) {
      auto value =
          moore::ReadOp::create(builder, implicitWaitOp.getLoc(), readValue);
      moore::DetectEventOp::create(builder, implicitWaitOp.getLoc(),
                                   moore::Edge::AnyChange, value, Value{});
    }
  }

  return success();
}

Value Context::convertLTLTimingControl(const slang::ast::TimingControl &ctrl,
                                       const Value &seqOrPro) {
  auto &builder = this->builder;
  auto loc = this->convertLocation(ctrl.sourceRange);
  LTLClockControlVisitor visitor{*this, loc, builder, seqOrPro};
  return ctrl.visit(visitor);
}
// NOLINTEND(misc-no-recursion)
