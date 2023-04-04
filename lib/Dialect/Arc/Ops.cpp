//===- Ops.cpp ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace circt;
using namespace arc;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static LogicalResult verifyArcSymbolUse(Operation *op, ValueRange inputs,
                                        ValueRange results,
                                        SymbolTableCollection &symbolTable) {
  // Check that the arc attribute was specified.
  auto arcName = op->getAttrOfType<FlatSymbolRefAttr>("arc");
  // The arc attribute is verified by the tablegen generated verifier as it is
  // an ODS defined attribute.
  assert(arcName && "FlatSymbolRefAttr called 'arc' missing");
  DefineOp arc = symbolTable.lookupNearestSymbolFrom<DefineOp>(op, arcName);
  if (!arc)
    return op->emitOpError() << "`" << arcName.getValue()
                             << "` does not reference a valid `arc.define`";

  // Verify that the operand and result types match the arc.
  auto type = arc.getFunctionType();
  if (type.getNumInputs() != inputs.size())
    return op->emitOpError("incorrect number of operands for arc");

  for (unsigned i = 0, e = type.getNumInputs(); i != e; ++i) {
    if (inputs[i].getType() != type.getInput(i)) {
      auto diag = op->emitOpError("operand type mismatch: operand ") << i;
      diag.attachNote() << "expected type: " << type.getInput(i);
      diag.attachNote() << "  actual type: " << inputs[i].getType();
      return diag;
    }
  }

  if (type.getNumResults() != results.size())
    return op->emitOpError("incorrect number of results for arc");

  for (unsigned i = 0, e = type.getNumResults(); i != e; ++i) {
    if (results[i].getType() != type.getResult(i)) {
      auto diag = op->emitOpError("result type mismatch: result ") << i;
      diag.attachNote() << "expected type: " << type.getResult(i);
      diag.attachNote() << "  actual type: " << results[i].getType();
      return diag;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DefineOp
//===----------------------------------------------------------------------===//

ParseResult DefineOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void DefineOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, "function_type", getArgAttrsAttrName(),
      getResAttrsAttrName());
}

LogicalResult DefineOp::verifyRegions() {
  // Check that the body does not contain any side-effecting operations. We can
  // simply iterate over the ops directly within the body; operations with
  // regions, like scf::IfOp, implement the `HasRecursiveMemoryEffects` trait
  // which causes the `isMemoryEffectFree` check to already recur into their
  // regions.
  for (auto &op : getBodyBlock()) {
    if (isMemoryEffectFree(&op))
      continue;

    // We don't use a op-error here because that leads to the whole arc being
    // printed. This can be switched of when creating the context, but one
    // might not want to switch that off for other error messages. Here it's
    // definitely not desirable as arcs can be very big and would fill up the
    // error log, making it hard to read. Currently, only the signature (first
    // line) of the arc is printed.
    auto diag = mlir::emitError(getLoc(), "body contains non-pure operation");
    diag.attachNote(op.getLoc()).append("first non-pure operation here: ");
    return diag;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

LogicalResult OutputOp::verify() {
  return success();

  auto parent = cast<DefineOp>((*this)->getParentOp());
  ArrayRef<Type> types = parent.getResultTypes();
  OperandRange values = getOperands();
  if (types.size() != values.size()) {
    emitOpError("must have same number of operands as parent arc has results");
    return failure();
  }

  for (size_t i = 0, e = types.size(); i < e; ++i) {
    if (types[i] != values[i].getType()) {
      emitOpError("output operand ")
          << i << " type mismatch: arc requires " << types[i] << ", operand is "
          << values[i].getType();
      return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// StateOp
//===----------------------------------------------------------------------===//

LogicalResult StateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyArcSymbolUse(*this, getInputs(), getResults(), symbolTable);
}

LogicalResult StateOp::canonicalize(StateOp op, PatternRewriter &rewriter) {
  // When there are no names attached, the state is not externaly observable.
  // When there are also no internal users, we can remove it.
  if (op->use_empty() && !op->hasAttr("name") && !op->hasAttr("names")) {
    rewriter.eraseOp(op);
    return success();
  }

  return failure();
}

LogicalResult StateOp::verify() {
  if (getLatency() > 0 && !getClock())
    return emitOpError("with non-zero latency requires a clock");

  if (getLatency() == 0) {
    if (getClock())
      return emitOpError("with zero latency cannot have a clock");
    if (getEnable())
      return emitOpError("with zero latency cannot have an enable");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyArcSymbolUse(*this, getInputs(), getResults(), symbolTable);
}

//===----------------------------------------------------------------------===//
// MemoryWriteOp
//===----------------------------------------------------------------------===//

LogicalResult MemoryWriteOp::verify() {
  if (getMask() && getMask().getType() != getData().getType())
    return emitOpError("mask and data operand types do not match");

  return success();
}

//===----------------------------------------------------------------------===//
// RootInputOp
//===----------------------------------------------------------------------===//

void RootInputOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  SmallString<32> buf("in_");
  buf += getName();
  setNameFn(getState(), buf);
}

//===----------------------------------------------------------------------===//
// RootOutputOp
//===----------------------------------------------------------------------===//

void RootOutputOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  SmallString<32> buf("out_");
  buf += getName();
  setNameFn(getState(), buf);
}

//===----------------------------------------------------------------------===//
// ModelOp
//===----------------------------------------------------------------------===//

LogicalResult ModelOp::verify() {
  if (getBodyBlock().getArguments().size() != 1)
    return emitOpError("must have exactly one argument");
  if (auto type = getBodyBlock().getArgument(0).getType();
      !isa<StorageType>(type))
    return emitOpError("argument must be of storage type");
  return success();
}

//===----------------------------------------------------------------------===//
// LutOp
//===----------------------------------------------------------------------===//

LogicalResult LutOp::verify() {
  Location firstSideEffectOpLoc = UnknownLoc::get(getContext());
  const WalkResult result = getBody().walk([&](Operation *op) {
    if (auto memOp = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
      memOp.getEffects(effects);

      if (!effects.empty()) {
        firstSideEffectOpLoc = memOp->getLoc();
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return emitOpError("no operations with side-effects allowed inside a LUT")
               .attachNote(firstSideEffectOpLoc)
           << "first operation with side-effects here";

  return success();
}

#define GET_OP_CLASSES
#include "circt/Dialect/Arc/Arc.cpp.inc"
