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
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/ModuleImplementation.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Sim/SimAttributes.h"
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

OpFoldResult OnEdgeOp::fold(FoldAdaptor adaptor) {
  if (!!adaptor.getClock())
    return NeverTriggerAttr::get(getContext(), getType());
  return {};
}

// --- TriggeredOp ---

void TriggeredOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        Value trigger, ValueRange arguments,
                        std::function<void()> bodyCtor) {
  OpBuilder::InsertionGuard guard(odsBuilder);
  odsState.addOperands({trigger});
  odsState.addOperands({arguments});
  auto body = odsState.addRegion();
  auto block = odsBuilder.createBlock(body);
  block->addArguments(
      arguments.getTypes(),
      SmallVector<Location>(arguments.size(), odsState.location));
  if (bodyCtor)
    bodyCtor();
  odsBuilder.create<sim::YieldSeqOp>(odsState.location);
}

void TriggeredOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        Value trigger, std::function<void()> bodyCtor) {
  return build(odsBuilder, odsState, trigger, {}, bodyCtor);
}

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
  if (isa_and_nonnull<sim::NeverTriggerAttr>(adaptor.getTrigger())) {
    // Never enabled, fold to tie-offs.
    if (getNumResults() > 0) {
      results.append(adaptor.getTieoffsAttr().begin(),
                     adaptor.getTieoffsAttr().end());
      return success();
    }
  }
  return failure();
}

static LogicalResult sinkConstantArguments(TriggeredOp op,
                                           PatternRewriter &rewriter) {
  auto isConstant = [](Value arg) -> bool {
    if (!arg || !arg.getDefiningOp())
      return false;
    return arg.getDefiningOp()->hasTrait<mlir::OpTrait::ConstantLike>();
  };

  size_t numConstantInputs = 0;
  for (auto input : op.getInputs())
    if (isConstant(input))
      ++numConstantInputs;

  if (numConstantInputs == 0)
    return failure();

  SmallVector<Value> newInputs;
  SmallVector<Type> newInputTypes;
  SmallVector<Location> newInputLocs;
  newInputs.reserve(op.getInputs().size() - numConstantInputs);
  newInputTypes.reserve(op.getInputs().size() - numConstantInputs);
  newInputLocs.reserve(op.getInputs().size() - numConstantInputs);

  for (auto [i, input] : llvm::enumerate(op.getInputs())) {
    if (!isConstant(input)) {
      newInputs.push_back(input);
      newInputTypes.push_back(input.getType());
      newInputLocs.push_back(op.getBody().getArgument(i).getLoc());
    }
  }

  auto newBody = std::make_unique<Block>();
  newBody->addArguments(newInputTypes, newInputLocs);

  rewriter.setInsertionPoint(op);
  auto newTriggerdOp = rewriter.create<sim::TriggeredOp>(
      op.getLoc(), op.getResultTypes(), op.getTrigger(), newInputs,
      op.getTieoffsAttr());

  rewriter.setInsertionPointToStart(&op.getBody().front());

  SmallVector<Value> argRepl;
  argRepl.reserve(op.getInputs().size());

  size_t newArgIdx = 0;
  for (auto input : op.getInputs()) {
    if (!isConstant(input)) {
      argRepl.push_back(newBody->getArgument(newArgIdx));
      ++newArgIdx;
    } else {
      auto cloned = rewriter.clone(*input.getDefiningOp());
      argRepl.push_back(cloned->getResult(0));
    }
  }

  newTriggerdOp.getBodyRegion().push_back(newBody.release());
  rewriter.mergeBlocks(&op.getBody().front(), &newTriggerdOp.getBody().front(),
                       argRepl);
  rewriter.replaceOp(op, newTriggerdOp);
  return success();
}

LogicalResult TriggeredOp::canonicalize(TriggeredOp op,
                                        PatternRewriter &rewriter) {

  if (succeeded(sinkConstantArguments(op, rewriter)))
    return success();

  if (op.getNumResults() > 0)
    return failure();

  bool isDeadOrEmpty = false;

  auto *bodyBlock = &op.getBodyRegion().front();
  isDeadOrEmpty = bodyBlock->without_terminator().empty();
  if (isa_and_nonnull<sim::NeverOp>(op.getTrigger().getDefiningOp()))
    isDeadOrEmpty = true;

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

  // If the current op can be inlined into the parent,
  // leave it to the parent's canonicalization.
  if (auto parentSeq = op.getParent().getDefiningOp<TriggerSequenceOp>()) {
    if (parentSeq == op) {
      op.emitWarning("Recursive trigger sequence.");
      auto neverOp =
          rewriter.create<NeverOp>(op.getLoc(), op.getParent().getType());
      for (auto res : op.getResults())
        rewriter.replaceAllUsesWith(res, neverOp.getResult());
      rewriter.eraseOp(op);
      return success();
    }
    if (op.getParent().hasOneUse())
      return failure();
  }

  auto getSingleSequenceUser = [](Value trigger) -> TriggerSequenceOp {
    if (!trigger.hasOneUse())
      return {};
    return dyn_cast<TriggerSequenceOp>(trigger.use_begin()->getOwner());
  };

  // Check if there are unused results (which can be removed) or
  // non-concurrent sub-sequences (which can be inlined).

  bool canBeChanged = false;
  for (auto res : op.getResults()) {
    auto singleSeqUser = getSingleSequenceUser(res);
    if (res.use_empty() || !!singleSeqUser) {
      canBeChanged = true;
      break;
    }
  }

  if (!canBeChanged)
    return failure();

  // DFS for inlinable values.
  SmallVector<Value> newResultValues;
  SmallVector<TriggerSequenceOp> inlinedSequences;
  llvm::SmallVector<std::pair<TriggerSequenceOp, unsigned>> sequenceOpStack;

  sequenceOpStack.push_back({op, 0});
  while (!sequenceOpStack.empty()) {
    auto &top = sequenceOpStack.back();
    auto currentSequence = top.first;
    unsigned resultIndex = top.second;

    while (resultIndex < currentSequence.getNumResults()) {
      auto currentResult = currentSequence.getResult(resultIndex);
      // Check we do not walk in a cycle.
      if (currentResult == op.getParent()) {
        op.emitWarning("Recursive trigger sequence.");
        auto neverOp =
            rewriter.create<NeverOp>(op.getLoc(), op.getParent().getType());
        for (auto res : op.getResults())
          rewriter.replaceAllUsesWith(res, neverOp.getResult());
        rewriter.eraseOp(op);
        return success();
      }

      if (auto inlinableChildSequence = getSingleSequenceUser(currentResult)) {
        // Save the next result index to visit on the
        // stack and put the new sequence on top.
        top.second = resultIndex + 1;
        sequenceOpStack.push_back({inlinableChildSequence, 0});
        inlinedSequences.push_back(inlinableChildSequence);
        inlinableChildSequence->dropAllReferences();
        break;
      }

      if (!currentResult.use_empty())
        newResultValues.push_back(currentResult);
      resultIndex++;
    }
    // Pop the sequence off of the stack if we have visited all results.
    if (resultIndex >= currentSequence.getNumResults())
      sequenceOpStack.pop_back();
  }

  // Remove dead sequences.
  if (newResultValues.empty()) {
    for (auto deadSubSequence : inlinedSequences)
      rewriter.eraseOp(deadSubSequence);
    rewriter.eraseOp(op);
    return success();
  }

  // Replace the current operation with a new sequence.
  rewriter.setInsertionPoint(op);

  SmallVector<Location> inlinedLocs;
  inlinedLocs.reserve(inlinedSequences.size() + 1);
  inlinedLocs.push_back(op.getLoc());
  for (auto subSequence : inlinedSequences)
    inlinedLocs.push_back(subSequence.getLoc());
  auto fusedLoc = FusedLoc::get(op.getContext(), inlinedLocs);
  inlinedLocs.clear();

  auto newOp = rewriter.create<TriggerSequenceOp>(fusedLoc, op.getParent(),
                                                  newResultValues.size());
  for (auto [rval, newRes] : llvm::zip(newResultValues, newOp.getResults()))
    rewriter.replaceAllUsesWith(rval, newRes);

  for (auto deadSubSequence : inlinedSequences)
    rewriter.eraseOp(deadSubSequence);
  rewriter.eraseOp(op);
  return success();
}

// ------- NeverOp ---------

OpFoldResult NeverOp::fold(FoldAdaptor) {
  return NeverTriggerAttr::get(getContext(), getType());
}

// ------- TriggerGateOp -----
OpFoldResult TriggerGateOp::fold(FoldAdaptor adaptor) {
  if (llvm::isa_and_nonnull<NeverTriggerAttr>(adaptor.getInput()))
    return adaptor.getInput();
  if (auto cstEnable =
          llvm::dyn_cast_or_null<IntegerAttr>(adaptor.getEnable())) {
    if (cstEnable.getValue().isZero())
      return NeverTriggerAttr::get(getContext(), getType());
    return getInput();
  }
  return {};
}

static TriggerGateOp getSingleGateChild(Value trigger) {
  if (!trigger.hasOneUse())
    return {};
  return dyn_cast<TriggerGateOp>(trigger.getUses().begin()->getOwner());
}

static bool hasSingleGateChild(Value trigger) {
  return !!getSingleGateChild(trigger);
}

template <typename S>
static inline void insertFactors(TriggerGateOp gate, S &set) {
  if (auto andOp = gate.getEnable().getDefiningOp<comb::AndOp>())
    for (auto factor : andOp.getOperands())
      set.insert(factor);
  else
    set.insert(gate.getEnable());
}

static LogicalResult hoistCommonGateFactors(TriggerSequenceOp sequence,
                                            PatternRewriter &rewriter) {

  Value nonCompositeFactor = {};
  Value firstFactor = {};
  bool allSame = true;
  for (auto res : sequence.getResults()) {
    auto gate = getSingleGateChild(res);
    if (!gate)
      return failure();

    // Check if all conditions are identical
    if (!firstFactor)
      firstFactor = gate.getEnable();
    else if (gate.getEnable() != firstFactor)
      allSame = false;

    bool isCompositeFactor =
        isa_and_nonnull<comb::AndOp>(gate.getEnable().getDefiningOp());
    if (!isCompositeFactor) {
      // If there are two different non-composite conditions, there's nothing to
      // to.
      if (!nonCompositeFactor)
        nonCompositeFactor = gate.getEnable();
      else if (nonCompositeFactor != gate.getEnable())
        return failure();
    }
  }

  SmallVector<Location> locs;

  if (allSame) {
    SmallVector<Location> locs;
    // The easy path: All sequence elements are gated the same way.
    // Remove the old gates:
    for (auto res : sequence.getResults()) {
      auto gate = getSingleGateChild(res);
      locs.push_back(gate.getLoc());
      rewriter.replaceAllUsesWith(gate.getResult(), gate.getInput());
      rewriter.eraseOp(gate);
    }
    // Insert the gate above the sequence op:
    auto fusedLoc = FusedLoc::get(sequence.getContext(), locs);
    rewriter.setInsertionPoint(sequence);
    auto newGate = rewriter.create<TriggerGateOp>(
        fusedLoc, sequence.getParent(), firstFactor);
    rewriter.modifyOpInPlace(sequence, [&]() {
      sequence.getParentMutable().assign(newGate.getResult());
    });
    return success();
  }

  // The hard path: All sequence elements have at least one common condition,
  // but they are not identical.

  // See if we can find a set of common factors.
  auto firstGate = getSingleGateChild(sequence.getResults().front());
  llvm::SmallSetVector<Value, 4> commonFactors;
  insertFactors(firstGate, commonFactors);

  for (auto res : sequence.getResults().drop_front()) {
    auto gate = getSingleGateChild(res);
    // Intersect the common factors with the local ones
    SmallPtrSet<Value, 4> localFactors;
    insertFactors(gate, localFactors);
    commonFactors.remove_if([&](auto commonFactor) {
      return !localFactors.contains(commonFactor);
    });
    if (commonFactors.empty())
      return failure();
  }

  // Yay! We found some.
  // Delete the common factors below.
  bool allTwoState = true;
  for (auto res : sequence.getResults()) {
    SmallVector<Value> newFactors;
    auto gate = getSingleGateChild(res);
    locs.push_back(gate.getLoc());
    auto andOp = gate.getEnable().getDefiningOp<comb::AndOp>();
    if (!!andOp) {
      if (!andOp.getTwoState())
        allTwoState = false;
      for (auto factor : andOp.getOperands())
        if (!commonFactors.contains(factor))
          newFactors.push_back(factor);
    } else if (!commonFactors.contains(gate.getEnable())) {
      newFactors.push_back(gate.getEnable());
    }
    if (newFactors.empty()) {
      rewriter.replaceAllUsesWith(gate.getResult(), gate.getInput());
      rewriter.eraseOp(gate);
    } else if (newFactors.size() == 1) {
      rewriter.modifyOpInPlace(
          gate, [&]() { gate.getEnableMutable().assign(newFactors.front()); });
    } else {
      assert(!!andOp);
      rewriter.setInsertionPoint(gate);
      auto newAnd = rewriter.createOrFold<comb::AndOp>(
          andOp.getLoc(), newFactors, andOp.getTwoState());
      rewriter.modifyOpInPlace(
          gate, [&]() { gate.getEnableMutable().assign(newAnd); });
    }
  }

  // Create the new gate above.
  auto fusedLoc = FusedLoc::get(sequence.getContext(), locs);
  auto commonFactorsVec = commonFactors.takeVector();
  rewriter.setInsertionPoint(sequence);
  auto commonCond = rewriter.createOrFold<comb::AndOp>(
      fusedLoc, rewriter.getI1Type(), commonFactorsVec, allTwoState);
  auto newGate = rewriter.create<TriggerGateOp>(fusedLoc, sequence.getParent(),
                                                commonCond);
  rewriter.modifyOpInPlace(sequence, [&]() {
    sequence.getParentMutable().assign(newGate.getResult());
  });
  return success();
}

LogicalResult TriggerGateOp::canonicalize(TriggerGateOp op,
                                          PatternRewriter &rewriter) {

  // Squash chained gates into a single one
  if (!op.getInput().getDefiningOp<TriggerGateOp>()) {
    if (hasSingleGateChild(op.getResult())) {
      SmallVector<TriggerGateOp> squashedGates;
      TriggerGateOp prevGate = {};
      TriggerGateOp gate = op;
      llvm::SmallSetVector<Value, 4> factors;
      // Descend the chain and collect the enable factors
      bool allTwoState = true;
      while (!!gate) {
        if (auto andOp = gate.getEnable().getDefiningOp<comb::AndOp>()) {
          for (auto factor : andOp.getInputs())
            factors.insert(factor);
          allTwoState &= andOp.getTwoState();
        } else {
          factors.insert(gate.getEnable());
        }
        squashedGates.push_back(gate);
        gate = getSingleGateChild(gate.getResult());
      }
      // Combine them all
      SmallVector<Location> locs;
      locs.reserve(squashedGates.size());
      for (auto sgate : squashedGates)
        locs.push_back(sgate.getLoc());
      auto fusesdLoc = FusedLoc::get(op.getContext(), locs);
      auto factorsVec = factors.takeVector();
      auto newCond = rewriter.createOrFold<comb::AndOp>(
          fusesdLoc, rewriter.getI1Type(), factorsVec, allTwoState);
      auto lastGate = squashedGates.pop_back_val();
      rewriter.replaceOpWithNewOp<TriggerGateOp>(lastGate, op.getInput(),
                                                 newCond);
      for (auto remGate : squashedGates)
        rewriter.eraseOp(remGate);
      return success();
    }
  }

  if (!op.getInput().hasOneUse())
    return failure();

  if (auto seqParent = op.getInput().getDefiningOp<TriggerSequenceOp>()) {
    auto sequenceIndex = cast<OpResult>(op.getInput()).getResultNumber();

    // Hoist common factors of a sequence
    if (sequenceIndex == 0)
      if (succeeded(hoistCommonGateFactors(seqParent, rewriter)))
        return success();

    // Break out adjacent subsequences with identical conditions.
    SmallVector<Location> locs;
    size_t breakOutStart = sequenceIndex;
    while (breakOutStart > 0) {
      auto prevGate =
          getSingleGateChild(seqParent.getResult(breakOutStart - 1));
      if (!!prevGate && prevGate.getEnable() == op.getEnable()) {
        locs.push_back(prevGate.getLoc());
        breakOutStart--;
      } else {
        break;
      }
    }
    size_t breakOutEnd = sequenceIndex;
    while (breakOutEnd < seqParent.getLength() - 1) {
      auto nextGate = getSingleGateChild(seqParent.getResult(breakOutEnd + 1));
      if (!!nextGate && nextGate.getEnable() == op.getEnable()) {
        locs.push_back(nextGate.getLoc());
        breakOutEnd++;
      } else {
        break;
      }
    }

    if (breakOutStart != breakOutEnd) {
      locs.push_back(op.getLoc());
      auto fusedLoc = FusedLoc::get(op.getContext(), locs);
      op->setLoc(fusedLoc);
      SmallVector<Value> subSeqVals;
      subSeqVals.reserve(breakOutEnd - breakOutStart + 1);
      for (size_t i = breakOutStart; i <= breakOutEnd; ++i)
        subSeqVals.push_back(
            getSingleGateChild(seqParent.getResult(i)).getResult());
      rewriter.setInsertionPointAfter(op);
      auto subSeq = rewriter.create<TriggerSequenceOp>(
          op.getLoc(), op.getResult(), subSeqVals.size());
      for (auto [i, useVal] : llvm::enumerate(subSeqVals)) {
        auto parentGate = useVal.getDefiningOp();
        assert(useVal.hasOneUse() || parentGate == op);
        assert(cast<TriggerGateOp>(parentGate).getEnable() == op.getEnable());
        rewriter.replaceAllUsesExcept(useVal, subSeq.getResult(i), subSeq);
        if (parentGate != op)
          rewriter.eraseOp(parentGate);
      }
      return success();
    }
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Sim/Sim.cpp.inc"
