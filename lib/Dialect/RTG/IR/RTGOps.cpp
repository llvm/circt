//===- RTGOps.cpp - Implement the RTG operations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RTG ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Support/ParsingUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir;
using namespace circt;
using namespace rtg;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

LogicalResult
ConstantOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                             ValueRange operands, DictionaryAttr attributes,
                             OpaqueProperties properties, RegionRange regions,
                             SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(
      properties.as<Properties *>()->getValue().getType());
  return success();
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValueAttr(); }

//===----------------------------------------------------------------------===//
// SequenceOp
//===----------------------------------------------------------------------===//

LogicalResult SequenceOp::verifyRegions() {
  if (TypeRange(getSequenceType().getElementTypes()) !=
      getBody()->getArgumentTypes())
    return emitOpError("sequence type does not match block argument types");

  return success();
}

ParseResult SequenceOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the name as a symbol.
  if (parser.parseSymbolName(
          result.getOrAddProperties<SequenceOp::Properties>().sym_name))
    return failure();

  // Parse the function signature.
  SmallVector<OpAsmParser::Argument> arguments;
  if (parser.parseArgumentList(arguments, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/true))
    return failure();

  SmallVector<Type> argTypes;
  SmallVector<Location> argLocs;
  argTypes.reserve(arguments.size());
  argLocs.reserve(arguments.size());
  for (auto &arg : arguments) {
    argTypes.push_back(arg.type);
    argLocs.push_back(arg.sourceLoc ? *arg.sourceLoc : result.location);
  }
  Type type = SequenceType::get(result.getContext(), argTypes);
  result.getOrAddProperties<SequenceOp::Properties>().sequenceType =
      TypeAttr::get(type);

  auto loc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc)
               << "'" << result.name.getStringRef() << "' op ";
      })))
    return failure();

  std::unique_ptr<Region> bodyRegionRegion = std::make_unique<Region>();
  if (parser.parseRegion(*bodyRegionRegion, arguments))
    return failure();

  if (bodyRegionRegion->empty()) {
    bodyRegionRegion->emplaceBlock();
    bodyRegionRegion->addArguments(argTypes, argLocs);
  }
  result.addRegion(std::move(bodyRegionRegion));

  return success();
}

void SequenceOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymNameAttr().getValue());
  p << "(";
  llvm::interleaveComma(getBody()->getArguments(), p,
                        [&](auto arg) { p.printRegionArgument(arg); });
  p << ")";
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(), {getSymNameAttrName(), getSequenceTypeAttrName()});
  p << ' ';
  p.printRegion(getBodyRegion(), /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// GetSequenceOp
//===----------------------------------------------------------------------===//

LogicalResult
GetSequenceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  SequenceOp seq =
      symbolTable.lookupNearestSymbolFrom<SequenceOp>(*this, getSequenceAttr());
  if (!seq)
    return emitOpError()
           << "'" << getSequence()
           << "' does not reference a valid 'rtg.sequence' operation";

  if (seq.getSequenceType() != getType())
    return emitOpError("referenced 'rtg.sequence' op's type does not match");

  return success();
}

//===----------------------------------------------------------------------===//
// SubstituteSequenceOp
//===----------------------------------------------------------------------===//

LogicalResult SubstituteSequenceOp::verify() {
  if (getReplacements().empty())
    return emitOpError("must at least have one replacement value");

  if (getReplacements().size() >
      getSequence().getType().getElementTypes().size())
    return emitOpError(
        "must not have more replacement values than sequence arguments");

  if (getReplacements().getTypes() !=
      getSequence().getType().getElementTypes().take_front(
          getReplacements().size()))
    return emitOpError("replacement types must match the same number of "
                       "sequence argument types from the front");

  return success();
}

LogicalResult SubstituteSequenceOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  ArrayRef<Type> argTypes =
      cast<SequenceType>(operands[0].getType()).getElementTypes();
  auto seqType =
      SequenceType::get(context, argTypes.drop_front(operands.size() - 1));
  inferredReturnTypes.push_back(seqType);
  return success();
}

ParseResult SubstituteSequenceOp::parse(::mlir::OpAsmParser &parser,
                                        ::mlir::OperationState &result) {
  OpAsmParser::UnresolvedOperand sequenceRawOperand;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> replacementsOperands;
  Type sequenceRawType;

  if (parser.parseOperand(sequenceRawOperand) || parser.parseLParen())
    return failure();

  auto replacementsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(replacementsOperands) || parser.parseRParen() ||
      parser.parseColon() || parser.parseType(sequenceRawType) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (!isa<SequenceType>(sequenceRawType))
    return parser.emitError(parser.getNameLoc())
           << "'sequence' must be handle to a sequence or sequence family, but "
              "got "
           << sequenceRawType;

  if (parser.resolveOperand(sequenceRawOperand, sequenceRawType,
                            result.operands))
    return failure();

  if (parser.resolveOperands(replacementsOperands,
                             cast<SequenceType>(sequenceRawType)
                                 .getElementTypes()
                                 .take_front(replacementsOperands.size()),
                             replacementsOperandsLoc, result.operands))
    return failure();

  SmallVector<Type> inferredReturnTypes;
  if (failed(inferReturnTypes(
          parser.getContext(), result.location, result.operands,
          result.attributes.getDictionary(parser.getContext()),
          result.getRawProperties(), result.regions, inferredReturnTypes)))
    return failure();

  result.addTypes(inferredReturnTypes);
  return success();
}

void SubstituteSequenceOp::print(OpAsmPrinter &p) {
  p << ' ' << getSequence() << "(" << getReplacements()
    << ") : " << getSequence().getType();
  p.printOptionalAttrDict((*this)->getAttrs(), {});
}

//===----------------------------------------------------------------------===//
// InterleaveSequencesOp
//===----------------------------------------------------------------------===//

LogicalResult InterleaveSequencesOp::verify() {
  if (getSequences().empty())
    return emitOpError("must have at least one sequence in the list");

  return success();
}

OpFoldResult InterleaveSequencesOp::fold(FoldAdaptor adaptor) {
  if (getSequences().size() == 1)
    return getSequences()[0];

  return {};
}

//===----------------------------------------------------------------------===//
// SetCreateOp
//===----------------------------------------------------------------------===//

ParseResult SetCreateOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 16> operands;
  Type elemType;

  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(elemType))
    return failure();

  result.addTypes({SetType::get(result.getContext(), elemType)});

  for (auto operand : operands)
    if (parser.resolveOperand(operand, elemType, result.operands))
      return failure();

  return success();
}

void SetCreateOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printOperands(getElements());
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getSet().getType().getElementType();
}

LogicalResult SetCreateOp::verify() {
  if (getElements().size() > 0) {
    // We only need to check the first element because of the `SameTypeOperands`
    // trait.
    if (getElements()[0].getType() != getSet().getType().getElementType())
      return emitOpError() << "operand types must match set element type";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SetCartesianProductOp
//===----------------------------------------------------------------------===//

LogicalResult SetCartesianProductOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  if (operands.empty()) {
    if (loc)
      return mlir::emitError(*loc) << "at least one set must be provided";
    return failure();
  }

  SmallVector<Type> elementTypes;
  for (auto operand : operands)
    elementTypes.push_back(cast<SetType>(operand.getType()).getElementType());
  inferredReturnTypes.push_back(
      SetType::get(TupleType::get(context, elementTypes)));
  return success();
}

//===----------------------------------------------------------------------===//
// BagCreateOp
//===----------------------------------------------------------------------===//

ParseResult BagCreateOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 16> elementOperands,
      multipleOperands;
  Type elemType;

  if (!parser.parseOptionalLParen()) {
    while (true) {
      OpAsmParser::UnresolvedOperand elementOperand, multipleOperand;
      if (parser.parseOperand(multipleOperand) || parser.parseKeyword("x") ||
          parser.parseOperand(elementOperand))
        return failure();

      elementOperands.push_back(elementOperand);
      multipleOperands.push_back(multipleOperand);

      if (parser.parseOptionalComma()) {
        if (parser.parseRParen())
          return failure();
        break;
      }
    }
  }

  if (parser.parseColon() || parser.parseType(elemType) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  result.addTypes({BagType::get(result.getContext(), elemType)});

  for (auto operand : elementOperands)
    if (parser.resolveOperand(operand, elemType, result.operands))
      return failure();

  for (auto operand : multipleOperands)
    if (parser.resolveOperand(operand, IndexType::get(result.getContext()),
                              result.operands))
      return failure();

  return success();
}

void BagCreateOp::print(OpAsmPrinter &p) {
  p << " ";
  if (!getElements().empty())
    p << "(";
  llvm::interleaveComma(llvm::zip(getElements(), getMultiples()), p,
                        [&](auto elAndMultiple) {
                          auto [el, multiple] = elAndMultiple;
                          p << multiple << " x " << el;
                        });
  if (!getElements().empty())
    p << ")";

  p << " : " << getBag().getType().getElementType();
  p.printOptionalAttrDict((*this)->getAttrs());
}

LogicalResult BagCreateOp::verify() {
  if (!llvm::all_equal(getElements().getTypes()))
    return emitOpError() << "types of all elements must match";

  if (getElements().size() > 0)
    if (getElements()[0].getType() != getBag().getType().getElementType())
      return emitOpError() << "operand types must match bag element type";

  return success();
}

//===----------------------------------------------------------------------===//
// TupleCreateOp
//===----------------------------------------------------------------------===//

LogicalResult TupleCreateOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  if (operands.empty()) {
    if (loc)
      return mlir::emitError(*loc) << "empty tuples not allowed";
    return failure();
  }

  SmallVector<Type> elementTypes;
  for (auto operand : operands)
    elementTypes.push_back(operand.getType());
  inferredReturnTypes.push_back(TupleType::get(context, elementTypes));
  return success();
}

//===----------------------------------------------------------------------===//
// TupleExtractOp
//===----------------------------------------------------------------------===//

LogicalResult TupleExtractOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  assert(operands.size() == 1 && "must have exactly one operand");

  auto tupleTy = dyn_cast<TupleType>(operands[0].getType());
  size_t idx = properties.as<Properties *>()->getIndex().getInt();
  if (!tupleTy || tupleTy.getTypes().size() <= idx) {
    if (loc)
      return mlir::emitError(*loc)
             << "index (" << idx
             << ") must be smaller than number of elements in tuple ("
             << tupleTy.getTypes().size() << ")";
    return failure();
  }

  inferredReturnTypes.push_back(tupleTy.getTypes()[idx]);
  return success();
}

//===----------------------------------------------------------------------===//
// FixedRegisterOp
//===----------------------------------------------------------------------===//

LogicalResult FixedRegisterOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(
      properties.as<Properties *>()->getReg().getType());
  return success();
}

OpFoldResult FixedRegisterOp::fold(FoldAdaptor adaptor) { return getRegAttr(); }

//===----------------------------------------------------------------------===//
// VirtualRegisterOp
//===----------------------------------------------------------------------===//

LogicalResult VirtualRegisterOp::verify() {
  if (getAllowedRegs().empty())
    return emitOpError("must have at least one allowed register");

  if (llvm::any_of(getAllowedRegs(), [](Attribute attr) {
        return !isa<RegisterAttrInterface>(attr);
      }))
    return emitOpError("all elements must be of RegisterAttrInterface");

  if (!llvm::all_equal(
          llvm::map_range(getAllowedRegs().getAsRange<RegisterAttrInterface>(),
                          [](auto attr) { return attr.getType(); })))
    return emitOpError("all allowed registers must be of the same type");

  return success();
}

LogicalResult VirtualRegisterOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto allowedRegs = properties.as<Properties *>()->getAllowedRegs();
  if (allowedRegs.empty()) {
    if (loc)
      return mlir::emitError(*loc, "must have at least one allowed register");

    return failure();
  }

  auto regAttr = dyn_cast<RegisterAttrInterface>(allowedRegs[0]);
  if (!regAttr) {
    if (loc)
      return mlir::emitError(
          *loc, "allowed register attributes must be of RegisterAttrInterface");

    return failure();
  }
  inferredReturnTypes.push_back(regAttr.getType());
  return success();
}

//===----------------------------------------------------------------------===//
// ContextSwitchOp
//===----------------------------------------------------------------------===//

LogicalResult ContextSwitchOp::verify() {
  auto elementTypes = getSequence().getType().getElementTypes();
  if (elementTypes.size() != 3)
    return emitOpError("sequence type must have exactly 3 element types");

  if (getFrom().getType() != elementTypes[0])
    return emitOpError(
        "first sequence element type must match 'from' attribute type");

  if (getTo().getType() != elementTypes[1])
    return emitOpError(
        "second sequence element type must match 'to' attribute type");

  auto seqTy = dyn_cast<SequenceType>(elementTypes[2]);
  if (!seqTy || !seqTy.getElementTypes().empty())
    return emitOpError(
        "third sequence element type must be a fully substituted sequence");

  return success();
}

//===----------------------------------------------------------------------===//
// TestOp
//===----------------------------------------------------------------------===//

LogicalResult TestOp::verifyRegions() {
  if (!getTarget().entryTypesMatch(getBody()->getArgumentTypes()))
    return emitOpError("argument types must match dict entry types");

  return success();
}

ParseResult TestOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the name as a symbol.
  if (parser.parseSymbolName(
          result.getOrAddProperties<TestOp::Properties>().sym_name))
    return failure();

  // Parse the function signature.
  SmallVector<OpAsmParser::Argument> arguments;
  SmallVector<StringAttr> names;

  auto parseOneArgument = [&]() -> ParseResult {
    std::string name;
    if (parser.parseKeywordOrString(&name) || parser.parseEqual() ||
        parser.parseArgument(arguments.emplace_back(), /*allowType=*/true,
                             /*allowAttrs=*/true))
      return failure();

    names.push_back(StringAttr::get(result.getContext(), name));
    return success();
  };
  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                     parseOneArgument, " in argument list"))
    return failure();

  SmallVector<Type> argTypes;
  SmallVector<DictEntry> entries;
  SmallVector<Location> argLocs;
  argTypes.reserve(arguments.size());
  argLocs.reserve(arguments.size());
  for (auto [name, arg] : llvm::zip(names, arguments)) {
    argTypes.push_back(arg.type);
    argLocs.push_back(arg.sourceLoc ? *arg.sourceLoc : result.location);
    entries.push_back({name, arg.type});
  }
  auto emitError = [&]() -> InFlightDiagnostic {
    return parser.emitError(parser.getCurrentLocation());
  };
  Type type = DictType::getChecked(emitError, result.getContext(),
                                   ArrayRef<DictEntry>(entries));
  if (!type)
    return failure();
  result.getOrAddProperties<TestOp::Properties>().target = TypeAttr::get(type);

  auto loc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc)
               << "'" << result.name.getStringRef() << "' op ";
      })))
    return failure();

  std::unique_ptr<Region> bodyRegionRegion = std::make_unique<Region>();
  if (parser.parseRegion(*bodyRegionRegion, arguments))
    return failure();

  if (bodyRegionRegion->empty()) {
    bodyRegionRegion->emplaceBlock();
    bodyRegionRegion->addArguments(argTypes, argLocs);
  }
  result.addRegion(std::move(bodyRegionRegion));

  return success();
}

void TestOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymNameAttr().getValue());
  p << "(";
  SmallString<32> resultNameStr;
  llvm::interleaveComma(
      llvm::zip(getTarget().getEntries(), getBody()->getArguments()), p,
      [&](auto entryAndArg) {
        auto [entry, arg] = entryAndArg;
        p << entry.name.getValue() << " = ";
        p.printRegionArgument(arg);
      });
  p << ")";
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(), {getSymNameAttrName(), getTargetAttrName()});
  p << ' ';
  p.printRegion(getBodyRegion(), /*printEntryBlockArgs=*/false);
}

void TestOp::getAsmBlockArgumentNames(Region &region,
                                      OpAsmSetValueNameFn setNameFn) {
  for (auto [entry, arg] :
       llvm::zip(getTarget().getEntries(), region.getArguments()))
    setNameFn(arg, entry.name.getValue());
}

//===----------------------------------------------------------------------===//
// TargetOp
//===----------------------------------------------------------------------===//

LogicalResult TargetOp::verifyRegions() {
  if (!getTarget().entryTypesMatch(
          getBody()->getTerminator()->getOperandTypes()))
    return emitOpError("terminator operand types must match dict entry types");

  return success();
}

//===----------------------------------------------------------------------===//
// ArrayCreateOp
//===----------------------------------------------------------------------===//

LogicalResult ArrayCreateOp::verify() {
  if (!getElements().empty() &&
      getElements()[0].getType() != getType().getElementType())
    return emitOpError("operand types must match array element type, expected ")
           << getType().getElementType() << " but got "
           << getElements()[0].getType();

  return success();
}

ParseResult ArrayCreateOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  Type elementType;

  if (parser.parseOperandList(operands) || parser.parseColon() ||
      parser.parseType(elementType) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (failed(parser.resolveOperands(operands, elementType, result.operands)))
    return failure();

  result.addTypes(ArrayType::get(elementType));

  return success();
}

void ArrayCreateOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printOperands(getElements());
  p << " : " << getType().getElementType();
  p.printOptionalAttrDict((*this)->getAttrs(), {});
}

//===----------------------------------------------------------------------===//
// MemoryBlockDeclareOp
//===----------------------------------------------------------------------===//

LogicalResult MemoryBlockDeclareOp::verify() {
  if (getBaseAddress().getBitWidth() != getType().getAddressWidth())
    return emitOpError(
        "base address width must match memory block address width");

  if (getEndAddress().getBitWidth() != getType().getAddressWidth())
    return emitOpError(
        "end address width must match memory block address width");

  if (getBaseAddress().ugt(getEndAddress()))
    return emitOpError(
        "base address must be smaller than or equal to the end address");

  return success();
}

ParseResult MemoryBlockDeclareOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  MemoryBlockType memoryBlockType;
  APInt start, end;

  if (parser.parseLSquare())
    return failure();

  auto startLoc = parser.getCurrentLocation();
  if (parser.parseInteger(start))
    return failure();

  if (parser.parseMinus())
    return failure();

  auto endLoc = parser.getCurrentLocation();
  if (parser.parseInteger(end) || parser.parseRSquare() ||
      parser.parseColonType(memoryBlockType) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  auto width = memoryBlockType.getAddressWidth();
  auto adjustAPInt = [&](APInt value, llvm::SMLoc loc) -> FailureOr<APInt> {
    if (value.getBitWidth() > width) {
      if (!value.isIntN(width))
        return parser.emitError(
                   loc,
                   "address out of range for memory block with address width ")
               << width;

      return value.trunc(width);
    }

    if (value.getBitWidth() < width)
      return value.zext(width);

    return value;
  };

  auto startRes = adjustAPInt(start, startLoc);
  auto endRes = adjustAPInt(end, endLoc);
  if (failed(startRes) || failed(endRes))
    return failure();

  auto intType = IntegerType::get(result.getContext(), width);
  result.addAttribute(getBaseAddressAttrName(result.name),
                      IntegerAttr::get(intType, *startRes));
  result.addAttribute(getEndAddressAttrName(result.name),
                      IntegerAttr::get(intType, *endRes));

  result.addTypes(memoryBlockType);
  return success();
}

void MemoryBlockDeclareOp::print(OpAsmPrinter &p) {
  SmallVector<char> str;
  getBaseAddress().toString(str, 16, false, false, false);
  p << " [0x" << str;
  p << " - 0x";
  str.clear();
  getEndAddress().toString(str, 16, false, false, false);
  p << str << "] : " << getType();
  p.printOptionalAttrDict((*this)->getAttrs(),
                          {getBaseAddressAttrName(), getEndAddressAttrName()});
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "circt/Dialect/RTG/IR/RTG.cpp.inc"
