//===- RTLDialect.cpp - Implement the RTL dialect -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RTL dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTL/RTLDialect.h"
#include "circt/Dialect/RTL/RTLOps.h"
#include "circt/Dialect/RTL/RTLTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace circt;
using namespace rtl;

//===----------------------------------------------------------------------===//
// Inliner Dialect Interface
//===----------------------------------------------------------------------===//

namespace {
struct RTLInlinerInterface : public mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;

  /// Returns true is the `callable` operation can be inlined into the location
  /// of the `call` operation.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const override final {
    // Only inline an  operation which is marked inline
    return call->hasAttr("inline");
  }

  /// Returns true if the operation from RTL `op` can be inlined in to the
  /// destination region `dest.
  bool
  isLegalToInline(Operation *op, Region *dest, bool,
                  BlockAndValueMapping &valueMapping) const override final {
    // No operations in RTL prevent inlining.
    return true;
  }

  /// Returns true if the given region `src` can be inlined into the region
  /// `dest`.
  bool isLegalToInline(Region *dest, Region *src, bool,
                       BlockAndValueMapping &) const override final {
    // No operations in RTL prevent inlining.
    return true;
  }

  /// When considering if `op` can be inlined, should this operation's regions
  /// be recursively querried for legality and cost analysis.
  bool shouldAnalyzeRecursively(Operation *op) const override final {
    // We don't need recursive analysis.  The only requirement is that the
    // instance is marked for inlining.
    return false;
  }

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToReplace) const override final {
    // Only terminator is the OutputOp
    auto outputOp = cast<rtl::OutputOp>(op);
    for (auto retValue : llvm::zip(valuesToReplace, outputOp.getOperands()))
      std::get<0>(retValue).replaceAllUsesWith(std::get<1>(retValue));
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// OpAsm Dialect Interface
//===----------------------------------------------------------------------===//

namespace {
// We implement the OpAsmDialectInterface so that RTL dialect operations
// automatically interpret the name attribute on operations as their SSA name.
struct RTLOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  /// Get a special name to use when printing the given operation. See
  /// OpAsmInterface.td#getAsmResultNames for usage details and documentation.
  void getAsmResultNames(Operation *op,
                         OpAsmSetValueNameFn setNameFn) const override {}

  /// Get a special name to use when printing the entry block arguments of the
  /// region contained by an operation in this dialect.
  void getAsmBlockArgumentNames(Block *block,
                                OpAsmSetValueNameFn setNameFn) const override {
    // Check to see if the operation containing the arguments has 'rtl.name'
    // attributes for them.  If so, use that as the name.
    auto *parentOp = block->getParentOp();

    for (size_t i = 0, e = block->getNumArguments(); i != e; ++i) {
      // Scan for a 'rtl.name' attribute.
      if (auto str = getRTLNameAttr(mlir::impl::getArgAttrs(parentOp, i)))
        setNameFn(block->getArgument(i), str.getValue());
    }
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

RTLDialect::RTLDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context,
              ::mlir::TypeID::get<RTLDialect>()) {

  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/RTL/RTLTypes.cpp.inc"
      >();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/RTL/RTL.cpp.inc"
      >();

  // Register interface implementations.
  addInterfaces<RTLInlinerInterface, RTLOpAsmDialectInterface>();
}

RTLDialect::~RTLDialect() {}

// Provide implementations for the enums we use.
#include "circt/Dialect/RTL/RTLEnums.cpp.inc"
