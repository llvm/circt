//===- SimOps.cpp - Implement the Sim operations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements `sim` dialect ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/HW/ModuleImplementation.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace circt;
using namespace sim;

ParseResult DPIFuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto builder = parser.getBuilder();
  // Parse visibility.
  (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  SmallVector<hw::module_like_impl::PortParse> ports;
  TypeAttr modType;
  if (failed(
          hw::module_like_impl::parseModuleSignature(parser, ports, modType)))
    return failure();

  result.addAttribute(DPIFuncOp::getModuleTypeAttrName(result.name), modType);

  // Convert the specified array of dictionary attrs (which may have null
  // entries) to an ArrayAttr of dictionaries.
  auto unknownLoc = builder.getUnknownLoc();
  SmallVector<Attribute> attrs, locs;
  auto nonEmptyLocsFn = [unknownLoc](Attribute attr) {
    return attr && cast<Location>(attr) != unknownLoc;
  };

  for (auto &port : ports) {
    attrs.push_back(port.attrs ? port.attrs : builder.getDictionaryAttr({}));
    locs.push_back(port.sourceLoc ? Location(*port.sourceLoc) : unknownLoc);
  }

  result.addAttribute(DPIFuncOp::getPerArgumentAttrsAttrName(result.name),
                      builder.getArrayAttr(attrs));
  result.addRegion();

  if (llvm::any_of(locs, nonEmptyLocsFn))
    result.addAttribute(DPIFuncOp::getArgumentLocsAttrName(result.name),
                        builder.getArrayAttr(locs));

  // Parse the attribute dict.
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  return success();
}

LogicalResult
sim::DPICallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto referencedOp =
      symbolTable.lookupNearestSymbolFrom(*this, getCalleeAttr());
  if (!referencedOp)
    return emitError("cannot find function declaration '")
           << getCallee() << "'";
  if (isa<func::FuncOp, sim::DPIFuncOp>(referencedOp))
    return success();
  return emitError("callee must be 'sim.dpi.func' or 'func.func' but got '")
         << referencedOp->getName() << "'";
}

void DPIFuncOp::print(OpAsmPrinter &p) {
  DPIFuncOp op = *this;
  // Print the operation and the function name.
  auto funcName =
      op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  p << ' ';

  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibility = op->getAttrOfType<StringAttr>(visibilityAttrName))
    p << visibility.getValue() << ' ';
  p.printSymbolName(funcName);
  hw::module_like_impl::printModuleSignatureNew(
      p, op->getRegion(0), op.getModuleType(),
      getPerArgumentAttrsAttr()
          ? ArrayRef<Attribute>(getPerArgumentAttrsAttr().getValue())
          : ArrayRef<Attribute>{},
      getArgumentLocs() ? SmallVector<Location>(
                              getArgumentLocs().value().getAsRange<Location>())
                        : ArrayRef<Location>{});

  mlir::function_interface_impl::printFunctionAttributes(
      p, op,
      {visibilityAttrName, getModuleTypeAttrName(),
       getPerArgumentAttrsAttrName(), getArgumentLocsAttrName()});
}

OpFoldResult FormatLitOp::fold(FoldAdaptor adaptor) { return getLiteralAttr(); }

OpFoldResult FormatDecOp::fold(FoldAdaptor adaptor) {
  if (getValue().getType() == IntegerType::get(getContext(), 0U))
    return StringAttr::get(getContext(), "0");

  if (auto intAttr = llvm::dyn_cast_or_null<IntegerAttr>(adaptor.getValue())) {
    SmallVector<char, 16> strBuf;
    intAttr.getValue().toString(strBuf, 10U, getIsSigned());

    unsigned width = intAttr.getType().getIntOrFloatBitWidth();
    unsigned padWidth = FormatDecOp::getDecimalWidth(width, getIsSigned());
    padWidth = padWidth > strBuf.size() ? padWidth - strBuf.size() : 0;

    SmallVector<char, 8> padding(padWidth, ' ');
    return StringAttr::get(getContext(), Twine(padding) + Twine(strBuf));
  }
  return {};
}

OpFoldResult FormatHexOp::fold(FoldAdaptor adaptor) {
  if (getValue().getType() == IntegerType::get(getContext(), 0U))
    return StringAttr::get(getContext(), "");

  if (auto intAttr = llvm::dyn_cast_or_null<IntegerAttr>(adaptor.getValue())) {
    SmallVector<char, 8> strBuf;
    intAttr.getValue().toString(strBuf, 16U, /*Signed*/ false,
                                /*formatAsCLiteral*/ false,
                                /*UpperCase*/ false);

    unsigned width = intAttr.getType().getIntOrFloatBitWidth();
    unsigned padWidth = width / 4;
    if (width % 4 != 0)
      padWidth++;
    padWidth = padWidth > strBuf.size() ? padWidth - strBuf.size() : 0;

    SmallVector<char, 8> padding(padWidth, '0');
    return StringAttr::get(getContext(), Twine(padding) + Twine(strBuf));
  }
  return {};
}

OpFoldResult FormatBinOp::fold(FoldAdaptor adaptor) {
  if (getValue().getType() == IntegerType::get(getContext(), 0U))
    return StringAttr::get(getContext(), "");

  if (auto intAttr = llvm::dyn_cast_or_null<IntegerAttr>(adaptor.getValue())) {
    SmallVector<char, 32> strBuf;
    intAttr.getValue().toString(strBuf, 2U, false);

    unsigned width = intAttr.getType().getIntOrFloatBitWidth();
    unsigned padWidth = width > strBuf.size() ? width - strBuf.size() : 0;

    SmallVector<char, 32> padding(padWidth, '0');
    return StringAttr::get(getContext(), Twine(padding) + Twine(strBuf));
  }
  return {};
}

OpFoldResult FormatCharOp::fold(FoldAdaptor adaptor) {
  auto width = getValue().getType().getIntOrFloatBitWidth();
  if (width > 8)
    return {};
  if (width == 0)
    return StringAttr::get(getContext(), Twine(static_cast<char>(0)));

  if (auto intAttr = llvm::dyn_cast_or_null<IntegerAttr>(adaptor.getValue())) {
    auto intValue = intAttr.getValue().getZExtValue();
    return StringAttr::get(getContext(), Twine(static_cast<char>(intValue)));
  }

  return {};
}

static StringAttr concatLiterals(MLIRContext *ctxt, ArrayRef<StringRef> lits) {
  assert(!lits.empty() && "No literals to concatenate");
  if (lits.size() == 1)
    return StringAttr::get(ctxt, lits.front());
  SmallString<64> newLit;
  for (auto lit : lits)
    newLit += lit;
  return StringAttr::get(ctxt, newLit);
}

OpFoldResult FormatStringConcatOp::fold(FoldAdaptor adaptor) {
  if (getNumOperands() == 0)
    return StringAttr::get(getContext(), "");
  if (getNumOperands() == 1) {
    // Don't fold to our own result to avoid an infinte loop.
    if (getResult() == getOperand(0))
      return {};
    return getOperand(0);
  }

  // Fold if all operands are literals.
  SmallVector<StringRef> lits;
  for (auto attr : adaptor.getInputs()) {
    auto lit = dyn_cast_or_null<StringAttr>(attr);
    if (!lit)
      return {};
    lits.push_back(lit);
  }
  return concatLiterals(getContext(), lits);
}

LogicalResult FormatStringConcatOp::getFlattenedInputs(
    llvm::SmallVectorImpl<Value> &flatOperands) {
  llvm::SmallMapVector<FormatStringConcatOp, unsigned, 4> concatStack;
  bool isCyclic = false;

  // Perform a DFS on this operation's concatenated operands,
  // collect the leaf format string fragments.
  concatStack.insert({*this, 0});
  while (!concatStack.empty()) {
    auto &top = concatStack.back();
    auto currentConcat = top.first;
    unsigned operandIndex = top.second;

    // Iterate over concatenated operands
    while (operandIndex < currentConcat.getNumOperands()) {
      auto currentOperand = currentConcat.getOperand(operandIndex);

      if (auto nextConcat =
              currentOperand.getDefiningOp<FormatStringConcatOp>()) {
        // Concat of a concat
        if (!concatStack.contains(nextConcat)) {
          // Save the next operand index to visit on the
          // stack and put the new concat on top.
          top.second = operandIndex + 1;
          concatStack.insert({nextConcat, 0});
          break;
        }
        // Cyclic concatenation encountered. Don't recurse.
        isCyclic = true;
      }

      flatOperands.push_back(currentOperand);
      operandIndex++;
    }

    // Pop the concat off of the stack if we have visited all operands.
    if (operandIndex >= currentConcat.getNumOperands())
      concatStack.pop_back();
  }

  return success(!isCyclic);
}

LogicalResult FormatStringConcatOp::verify() {
  if (llvm::any_of(getOperands(),
                   [&](Value operand) { return operand == getResult(); }))
    return emitOpError("is infinitely recursive.");
  return success();
}

LogicalResult FormatStringConcatOp::canonicalize(FormatStringConcatOp op,
                                                 PatternRewriter &rewriter) {

  auto fmtStrType = FormatStringType::get(op.getContext());

  // Check if we can flatten concats of concats
  bool hasBeenFlattened = false;
  SmallVector<Value, 0> flatOperands;
  if (!op.isFlat()) {
    // Get a new, flattened list of operands
    flatOperands.reserve(op.getNumOperands() + 4);
    auto isAcyclic = op.getFlattenedInputs(flatOperands);

    if (failed(isAcyclic)) {
      // Infinite recursion, but we cannot fail compilation right here (can we?)
      // so just emit a warning and bail out.
      op.emitWarning("Cyclic concatenation detected.");
      return failure();
    }

    hasBeenFlattened = true;
  }

  if (!hasBeenFlattened && op.getNumOperands() < 2)
    return failure(); // Should be handled by the folder

  // Check if there are adjacent literals we can merge or empty literals to
  // remove
  SmallVector<StringRef> litSequence;
  SmallVector<Value> newOperands;
  newOperands.reserve(op.getNumOperands());
  FormatLitOp prevLitOp;

  auto oldOperands = hasBeenFlattened ? flatOperands : op.getOperands();
  for (auto operand : oldOperands) {
    if (auto litOp = operand.getDefiningOp<FormatLitOp>()) {
      if (!litOp.getLiteral().empty()) {
        prevLitOp = litOp;
        litSequence.push_back(litOp.getLiteral());
      }
    } else {
      if (!litSequence.empty()) {
        if (litSequence.size() > 1) {
          // Create a fused literal.
          auto newLit = rewriter.createOrFold<FormatLitOp>(
              op.getLoc(), fmtStrType,
              concatLiterals(op.getContext(), litSequence));
          newOperands.push_back(newLit);
        } else {
          // Reuse the existing literal.
          newOperands.push_back(prevLitOp.getResult());
        }
        litSequence.clear();
      }
      newOperands.push_back(operand);
    }
  }

  // Push trailing literals into the new operand list
  if (!litSequence.empty()) {
    if (litSequence.size() > 1) {
      // Create a fused literal.
      auto newLit = rewriter.createOrFold<FormatLitOp>(
          op.getLoc(), fmtStrType,
          concatLiterals(op.getContext(), litSequence));
      newOperands.push_back(newLit);
    } else {
      // Reuse the existing literal.
      newOperands.push_back(prevLitOp.getResult());
    }
  }

  if (!hasBeenFlattened && newOperands.size() == op.getNumOperands())
    return failure(); // Nothing changed

  if (newOperands.empty())
    rewriter.replaceOpWithNewOp<FormatLitOp>(op, fmtStrType,
                                             rewriter.getStringAttr(""));
  else if (newOperands.size() == 1)
    rewriter.replaceOp(op, newOperands);
  else
    rewriter.modifyOpInPlace(op, [&]() { op->setOperands(newOperands); });

  return success();
}

LogicalResult PrintFormattedOp::canonicalize(PrintFormattedOp op,
                                             PatternRewriter &rewriter) {
  // Remove ops with constant false condition.
  if (auto cstCond = op.getCondition().getDefiningOp<hw::ConstantOp>()) {
    if (cstCond.getValue().isZero()) {
      rewriter.eraseOp(op);
      return success();
    }
  }
  return failure();
}

LogicalResult PrintFormattedProcOp::verify() {
  // Check if we know for sure that the parent is not procedural.
  auto *parentOp = getOperation()->getParentOp();

  if (!parentOp)
    return emitOpError("must be within a procedural region.");

  if (isa<hw::HWDialect>(parentOp->getDialect())) {
    if (!isa<hw::TriggeredOp>(parentOp))
      return emitOpError("must be within a procedural region.");
    return success();
  }

  if (isa<sv::SVDialect>(parentOp->getDialect())) {
    if (!parentOp->hasTrait<sv::ProceduralRegion>())
      return emitOpError("must be within a procedural region.");
    return success();
  }

  // Don't fail for dialects that are not explicitly handled.
  return success();
}

LogicalResult PrintFormattedProcOp::canonicalize(PrintFormattedProcOp op,
                                                 PatternRewriter &rewriter) {
  // Remove empty prints.
  if (auto litInput = op.getInput().getDefiningOp<FormatLitOp>()) {
    if (litInput.getLiteral().empty()) {
      rewriter.eraseOp(op);
      return success();
    }
  }
  return failure();
}

// --- OnEdgeOp ---

LogicalResult OnEdgeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto eventAttr = properties.as<OnEdgeOp::Properties *>()->getEvent();
  inferredReturnTypes.emplace_back(
      EdgeTriggerType::get(context, eventAttr.getValue()));
  return success();
}

// --- TriggeredOp ---

LogicalResult TriggeredOp::verify() {
  if (getNumResults() > 0 && !getTieoffs())
    return emitError("Tie-off constants must be provided for all results.");
  auto numTieoffs = !getTieoffs() ? 0 : getTieoffsAttr().size();
  if (numTieoffs != getNumResults())
    return emitError(
        "Number of tie-off constants does not match number of results.");
  if (numTieoffs == 0)
    return success();
  unsigned idx = 0;
  bool failed = false;
  for (const auto &[res, tieoff] :
       llvm::zip(getResultTypes(), getTieoffsAttr())) {
    if (res != cast<TypedAttr>(tieoff).getType()) {
      emitError("Tie-off type does not match for result at index " +
                Twine(idx));
      failed = true;
    }
    ++idx;
  }
  return success(!failed);
}

LogicalResult TriggeredOp::fold(FoldAdaptor adaptor,
                                SmallVectorImpl<OpFoldResult> &results) {
  if (auto constCond = dyn_cast_or_null<IntegerAttr>(adaptor.getCondition())) {
    if (constCond.getValue().isAllOnes()) {
      // Strip constant true condition.
      getConditionMutable().clear();
      return success();
    }
    // Never enabled, fold to tie-offs.
    if (getNumResults() > 0) {
      results.append(adaptor.getTieoffsAttr().begin(),
                     adaptor.getTieoffsAttr().end());
      return success();
    }
  }
  return failure();
}

LogicalResult TriggeredOp::canonicalize(TriggeredOp op,
                                        PatternRewriter &rewriter) {
  if (op.getNumResults() > 0)
    return failure();

  bool isDeadOrEmpty = false;

  auto *bodyBlock = &op.getBodyRegion().front();
  isDeadOrEmpty = bodyBlock->without_terminator().empty();

  if (!isDeadOrEmpty && !!op.getCondition())
    if (auto cstCond = op.getCondition().getDefiningOp<hw::ConstantOp>())
      isDeadOrEmpty = cstCond.getValue().isZero();

  if (!isDeadOrEmpty)
    return failure();

  rewriter.eraseOp(op);
  return success();
}

// --- TriggerSequenceOp ---

LogicalResult TriggerSequenceOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Create N results matching the type of the parent trigger, where N is the
  // specified length of the sequence.
  auto lengthAttr =
      properties.as<TriggerSequenceOp::Properties *>()->getLength();
  uint32_t len = lengthAttr.getValue().getZExtValue();
  Type trigType = operands.front().getType();
  inferredReturnTypes.resize_for_overwrite(len);
  for (size_t i = 0; i < len; ++i)
    inferredReturnTypes[i] = trigType;
  return success();
}

LogicalResult TriggerSequenceOp::verify() {
  if (getLength() != getNumResults())
    return emitOpError("specified length does not match number of results.");
  return success();
}

LogicalResult TriggerSequenceOp::fold(FoldAdaptor adaptor,
                                      SmallVectorImpl<OpFoldResult> &results) {
  // Fold trivial sequences to the parent trigger.
  if (getLength() == 1 && getResult(0) != getParent()) {
    results.push_back(getParent());
    return success();
  }
  return failure();
}

LogicalResult TriggerSequenceOp::canonicalize(TriggerSequenceOp op,
                                              PatternRewriter &rewriter) {
  if (op.getNumResults() == 0) {
    rewriter.eraseOp(op);
    return success();
  }

  // Check if there are unused results (which can be removed) or
  // non-concurrent sub-sequences (which can be inlined).
  auto getSingleSequenceUser = [](Value trigger) -> TriggerSequenceOp {
    if (!trigger.hasOneUse())
      return {};
    return dyn_cast<TriggerSequenceOp>(trigger.use_begin()->getOwner());
  };

  bool canBeChanged = false;
  for (auto res : op.getResults()) {
    auto singleSeqUser = getSingleSequenceUser(res);
    if (singleSeqUser == op) {
      op.emitWarning("Recursive trigger sequence.");
      return failure();
    }
    if (res.use_empty() || !!singleSeqUser) {
      canBeChanged = true;
      break;
    }
  }

  if (!canBeChanged)
    return failure();

  // Build a list of new result values.
  SmallVector<Value> resultValues;
  SmallVector<Location> locs;
  SmallVector<TriggerSequenceOp> childSeqs;
  locs.emplace_back(op.getLoc());
  resultValues.reserve(op.getNumResults());
  for (auto res : op.getResults()) {
    if (res.use_empty())
      continue;

    if (auto seqUser = getSingleSequenceUser(res)) {
      resultValues.append(seqUser.getResults().begin(),
                          seqUser.getResults().end());
      locs.emplace_back(seqUser.getLoc());
      childSeqs.emplace_back(seqUser);
    } else {
      resultValues.emplace_back(res);
    }
  }

  // Remove empty sequences.
  if (resultValues.empty()) {
    rewriter.eraseOp(op);
    return success();
  }

  // Replace the current operation with a new sequence.
  rewriter.setInsertionPoint(op);
  auto fusedLoc = FusedLoc::get(rewriter.getContext(), locs);
  auto newOp = rewriter.create<TriggerSequenceOp>(fusedLoc, op.getParent(),
                                                  resultValues.size());
  for (auto [rval, newRes] : llvm::zip(resultValues, newOp.getResults()))
    rewriter.replaceAllUsesWith(rval, newRes);
  // Remove sequences that have been inlined
  for (auto child : childSeqs)
    rewriter.eraseOp(child);

  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Sim/Sim.cpp.inc"
