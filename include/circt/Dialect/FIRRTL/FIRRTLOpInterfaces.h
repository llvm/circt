//===- FIRRTLOpInterfaces.h - Declare FIRRTL op interfaces ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation interfaces for the FIRRTL IR and supporting
// types.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/CommandLine.h"

namespace circt {
namespace firrtl {

class FIRRTLType;

//===----------------------------------------------------------------------===//
// FlowKind
//===----------------------------------------------------------------------===//

enum class Flow { Source, Sink, Duplex };

/// Get a flow's reverse.
Flow flipFlow(Flow flow);

/// This is used to represent the result of querying the flow of an operation.
/// If the operation has sink or source flow, this can be implicitly created
/// from a Flow instance.  If an operation has the same flow as a different
/// value, presumably one of its arguments, this can be implcitly created from
/// a Value.  If the operation has the flipped flow of a value, it can signal so
/// by using `FlowResult::flipOf(value)`.
struct FlowResult {
  /*implicit*/ FlowResult(Flow flow) : flow(flow) {}
  /*implicit*/ FlowResult(Value value) : value(value) {}
  static FlowResult flipOf(Value value) {
    return FlowResult(Flow::Sink, value);
  }

  Flow getFlow() { return flow; }
  Value getValue() { return value; }

private:
  FlowResult(Flow flow, Value value) : flow(flow), value(value) {}
  Flow flow = Flow::Source;
  Value value;
};

//===----------------------------------------------------------------------===//
// FModuleLike
//===----------------------------------------------------------------------===//

/// This holds the name and type that describes the module's ports.
struct PortInfo {
  StringAttr name;
  FIRRTLType type;
  Direction direction;
  Location loc;
  AnnotationSet annotations = AnnotationSet(type.getContext());

  StringRef getName() const { return name ? name.getValue() : ""; }

  /// Return true if this is a simple output-only port.  If you want the
  /// direction of the port, use the \p direction parameter.
  bool isOutput() {
    auto flags = type.getRecursiveTypeProperties();
    return flags.isPassive && !flags.containsAnalog &&
           direction == Direction::Out;
  }

  /// Return true if this is a simple input-only port.  If you want the
  /// direction of the port, use the \p direction parameter.
  bool isInput() {
    auto flags = type.getRecursiveTypeProperties();
    return flags.isPassive && !flags.containsAnalog &&
           direction == Direction::In;
  }

  /// Return true if this is an inout port.  This will be true if the port
  /// contains either bi-directional signals or analog types.
  bool isInOut() { return !isOutput() && !isInput(); }
};

/// Verification hook for verifying module like operations.
LogicalResult verifyModuleLikeOpInterface(FModuleLike module);

} // namespace firrtl
} // namespace circt

#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h.inc"
