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
#include "circt/Support/CustomDirectiveImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace circt;
using namespace sim;

static StringAttr formatIntegersByRadix(MLIRContext *ctx, unsigned radix,
                                        const Attribute &value,
                                        bool isUpperCase, bool isLeftAligned,
                                        char paddingChar,
                                        std::optional<unsigned> specifierWidth,
                                        bool isSigned = false) {
  auto intAttr = llvm::dyn_cast_or_null<IntegerAttr>(value);
  if (!intAttr)
    return {};
  if (intAttr.getType().getIntOrFloatBitWidth() == 0)
    return StringAttr::get(ctx, "");

  SmallVector<char, 32> strBuf;
  intAttr.getValue().toString(strBuf, radix, isSigned, false, isUpperCase);
  unsigned width = intAttr.getType().getIntOrFloatBitWidth();

  unsigned padWidth;
  switch (radix) {
  case 2:
    padWidth = width;
    break;
  case 8:
    padWidth = (width + 2) / 3;
    break;
  case 16:
    padWidth = (width + 3) / 4;
    break;
  default:
    padWidth = width;
    break;
  }

  unsigned numSpaces = 0;
  if (specifierWidth.has_value() &&
      (specifierWidth.value() >
       std::max(padWidth, static_cast<unsigned>(strBuf.size())))) {
    numSpaces = std::max(
        0U, specifierWidth.value() -
                std::max(padWidth, static_cast<unsigned>(strBuf.size())));
  }

  SmallVector<char, 1> spacePadding(numSpaces, ' ');

  padWidth = padWidth > strBuf.size() ? padWidth - strBuf.size() : 0;

  SmallVector<char, 32> padding(padWidth, paddingChar);
  if (isLeftAligned) {
    return StringAttr::get(ctx, Twine(padding) + Twine(strBuf) +
                                    Twine(spacePadding));
  }
  return StringAttr::get(ctx,
                         Twine(spacePadding) + Twine(padding) + Twine(strBuf));
}

static StringAttr formatFloatsBySpecifier(MLIRContext *ctx, Attribute value,
                                          bool isLeftAligned,
                                          std::optional<unsigned> fieldWidth,
                                          std::optional<unsigned> fracDigits,
                                          std::string formatSpecifier) {
  if (auto floatAttr = llvm::dyn_cast_or_null<FloatAttr>(value)) {
    std::string widthString = isLeftAligned ? "-" : "";
    if (fieldWidth.has_value()) {
      widthString += std::to_string(fieldWidth.value());
    }
    std::string fmtSpecifier = "%" + widthString + "." +
                               std::to_string(fracDigits.value()) +
                               formatSpecifier;

    // Calculates number of bytes needed to store the format string
    // excluding the null terminator
    int bufferSize = std::snprintf(nullptr, 0, fmtSpecifier.c_str(),
                                   floatAttr.getValue().convertToDouble());
    std::string floatFmtBuffer(bufferSize, '\0');
    snprintf(floatFmtBuffer.data(), bufferSize + 1, fmtSpecifier.c_str(),
             floatAttr.getValue().convertToDouble());
    return StringAttr::get(ctx, floatFmtBuffer);
  }
  return {};
}

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

OpFoldResult FormatLiteralOp::fold(FoldAdaptor adaptor) {
  return getLiteralAttr();
}

// --- FormatDecOp ---

StringAttr FormatDecOp::formatConstant(Attribute constVal) {
  auto intAttr = llvm::dyn_cast<IntegerAttr>(constVal);
  if (!intAttr)
    return {};
  SmallVector<char, 16> strBuf;
  intAttr.getValue().toString(strBuf, 10, getIsSigned());
  unsigned padWidth;
  if (getSpecifierWidth().has_value()) {
    padWidth = getSpecifierWidth().value();
  } else {
    unsigned width = intAttr.getType().getIntOrFloatBitWidth();
    padWidth = FormatDecOp::getDecimalWidth(width, getIsSigned());
  }

  padWidth = padWidth > strBuf.size() ? padWidth - strBuf.size() : 0;

  SmallVector<char, 10> padding(padWidth, getPaddingChar());
  if (getIsLeftAligned())
    return StringAttr::get(getContext(), Twine(strBuf) + Twine(padding));
  return StringAttr::get(getContext(), Twine(padding) + Twine(strBuf));
}

OpFoldResult FormatDecOp::fold(FoldAdaptor adaptor) {
  if (getValue().getType().getIntOrFloatBitWidth() == 0)
    return StringAttr::get(getContext(), "0");
  return {};
}

// --- FormatHexOp ---

StringAttr FormatHexOp::formatConstant(Attribute constVal) {
  return formatIntegersByRadix(constVal.getContext(), 16, constVal,
                               getIsHexUppercase(), getIsLeftAligned(),
                               getPaddingChar(), getSpecifierWidth());
}

OpFoldResult FormatHexOp::fold(FoldAdaptor adaptor) {
  if (getValue().getType().getIntOrFloatBitWidth() == 0)
    return formatIntegersByRadix(
        getContext(), 16, IntegerAttr::get(getValue().getType(), 0), false,
        getIsLeftAligned(), getPaddingChar(), getSpecifierWidth());
  return {};
}

// --- FormatOctOp ---

StringAttr FormatOctOp::formatConstant(Attribute constVal) {
  return formatIntegersByRadix(constVal.getContext(), 8, constVal, false,
                               getIsLeftAligned(), getPaddingChar(),
                               getSpecifierWidth());
}

OpFoldResult FormatOctOp::fold(FoldAdaptor adaptor) {
  if (getValue().getType().getIntOrFloatBitWidth() == 0)
    return formatIntegersByRadix(
        getContext(), 8, IntegerAttr::get(getValue().getType(), 0), false,
        getIsLeftAligned(), getPaddingChar(), getSpecifierWidth());
  return {};
}

// --- FormatBinOp ---

StringAttr FormatBinOp::formatConstant(Attribute constVal) {
  return formatIntegersByRadix(constVal.getContext(), 2, constVal, false,
                               getIsLeftAligned(), getPaddingChar(),
                               getSpecifierWidth());
}

OpFoldResult FormatBinOp::fold(FoldAdaptor adaptor) {
  if (getValue().getType().getIntOrFloatBitWidth() == 0)
    return formatIntegersByRadix(
        getContext(), 2, IntegerAttr::get(getValue().getType(), 0), false,
        getIsLeftAligned(), getPaddingChar(), getSpecifierWidth());
  return {};
}

// --- FormatScientificOp ---

StringAttr FormatScientificOp::formatConstant(Attribute constVal) {
  return formatFloatsBySpecifier(getContext(), constVal, getIsLeftAligned(),
                                 getFieldWidth(), getFracDigits(), "e");
}

// --- FormatFloatOp ---

StringAttr FormatFloatOp::formatConstant(Attribute constVal) {
  return formatFloatsBySpecifier(getContext(), constVal, getIsLeftAligned(),
                                 getFieldWidth(), getFracDigits(), "f");
}

// --- FormatGeneralOp ---

StringAttr FormatGeneralOp::formatConstant(Attribute constVal) {
  return formatFloatsBySpecifier(getContext(), constVal, getIsLeftAligned(),
                                 getFieldWidth(), getFracDigits(), "g");
}

// --- FormatCharOp ---

StringAttr FormatCharOp::formatConstant(Attribute constVal) {
  auto intCst = dyn_cast<IntegerAttr>(constVal);
  if (!intCst)
    return {};
  if (intCst.getType().getIntOrFloatBitWidth() == 0)
    return StringAttr::get(getContext(), Twine(static_cast<char>(0)));
  if (intCst.getType().getIntOrFloatBitWidth() > 8)
    return {};
  auto intValue = intCst.getValue().getZExtValue();
  return StringAttr::get(getContext(), Twine(static_cast<char>(intValue)));
}

OpFoldResult FormatCharOp::fold(FoldAdaptor adaptor) {
  if (getValue().getType().getIntOrFloatBitWidth() == 0)
    return StringAttr::get(getContext(), Twine(static_cast<char>(0)));
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
  FormatLiteralOp prevLitOp;

  auto oldOperands = hasBeenFlattened ? flatOperands : op.getOperands();
  for (auto operand : oldOperands) {
    if (auto litOp = operand.getDefiningOp<FormatLiteralOp>()) {
      if (!litOp.getLiteral().empty()) {
        prevLitOp = litOp;
        litSequence.push_back(litOp.getLiteral());
      }
    } else {
      if (!litSequence.empty()) {
        if (litSequence.size() > 1) {
          // Create a fused literal.
          auto newLit = rewriter.createOrFold<FormatLiteralOp>(
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
      auto newLit = rewriter.createOrFold<FormatLiteralOp>(
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
    rewriter.replaceOpWithNewOp<FormatLiteralOp>(op, fmtStrType,
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

  if (isa_and_nonnull<hw::HWDialect>(parentOp->getDialect())) {
    if (!isa<hw::TriggeredOp>(parentOp))
      return emitOpError("must be within a procedural region.");
    return success();
  }

  if (isa_and_nonnull<sv::SVDialect>(parentOp->getDialect())) {
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
  if (auto litInput = op.getInput().getDefiningOp<FormatLiteralOp>()) {
    if (litInput.getLiteral().empty()) {
      rewriter.eraseOp(op);
      return success();
    }
  }
  return failure();
}

OpFoldResult StringConstantOp::fold(FoldAdaptor adaptor) {
  return adaptor.getLiteralAttr();
}

OpFoldResult StringConcatOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getInputs();
  if (operands.empty())
    return StringAttr::get(getContext(), "");

  SmallString<128> result;
  for (auto &operand : operands) {
    if (auto strAttr = cast<StringAttr>(operand))
      result += strAttr.getValue();
    else
      return {};
  }

  return StringAttr::get(getContext(), result);
}

OpFoldResult StringLengthOp::fold(FoldAdaptor adaptor) {
  auto inputAttr = adaptor.getInput();
  if (!inputAttr)
    return {};

  if (auto strAttr = cast<StringAttr>(inputAttr))
    return IntegerAttr::get(getType(), strAttr.getValue().size());

  return {};
}

OpFoldResult IntToStringOp::fold(FoldAdaptor adaptor) {
  auto inputAttr = adaptor.getInput();
  if (!inputAttr)
    return {};
  if (auto intAttr = cast<IntegerAttr>(inputAttr)) {
    std::string result{};
    auto width = intAttr.getType().getIntOrFloatBitWidth();
    result.reserve((width + 7) / 8);
    for (unsigned int i = 0; i < width; i += 8) {
      auto byte =
          intAttr.getValue().extractBitsAsZExtValue(std::min(width - i, 8U), i);
      if (byte)
        result.push_back(static_cast<char>(byte));
    }
    std::reverse(result.begin(), result.end());
    return StringAttr::get(getContext(), result);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/SimOpInterfaces.cpp.inc"

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Sim/Sim.cpp.inc"
