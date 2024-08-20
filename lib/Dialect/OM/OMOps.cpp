//===- OMOps.cpp - Object Model operation definitions ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Object Model operation definitions.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OM/OMOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/OM/OMUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;
using namespace circt::om;

//===----------------------------------------------------------------------===//
// Path Printers and Parsers
//===----------------------------------------------------------------------===//

static ParseResult parseBasePathString(OpAsmParser &parser, PathAttr &path) {
  auto *context = parser.getContext();
  auto loc = parser.getCurrentLocation();
  std::string rawPath;
  if (parser.parseString(&rawPath))
    return failure();
  if (parseBasePath(context, rawPath, path))
    return parser.emitError(loc, "invalid base path");
  return success();
}

static void printBasePathString(OpAsmPrinter &p, Operation *op, PathAttr path) {
  p << '\"';
  llvm::interleave(
      path, p,
      [&](const PathElement &elt) {
        p << elt.module.getValue() << '/' << elt.instance.getValue();
      },
      ":");
  p << '\"';
}

static ParseResult parsePathString(OpAsmParser &parser, PathAttr &path,
                                   StringAttr &module, StringAttr &ref,
                                   StringAttr &field) {

  auto *context = parser.getContext();
  auto loc = parser.getCurrentLocation();
  std::string rawPath;
  if (parser.parseString(&rawPath))
    return failure();
  if (parsePath(context, rawPath, path, module, ref, field))
    return parser.emitError(loc, "invalid path");
  return success();
}

static void printPathString(OpAsmPrinter &p, Operation *op, PathAttr path,
                            StringAttr module, StringAttr ref,
                            StringAttr field) {
  p << '\"';
  for (const auto &elt : path)
    p << elt.module.getValue() << '/' << elt.instance.getValue() << ':';
  if (!module.getValue().empty())
    p << module.getValue();
  if (!ref.getValue().empty())
    p << '>' << ref.getValue();
  if (!field.getValue().empty())
    p << field.getValue();
  p << '\"';
}

//===----------------------------------------------------------------------===//
// Shared definitions
//===----------------------------------------------------------------------===//
static ParseResult parseClassFieldsList(OpAsmParser &parser,
                                        SmallVectorImpl<Attribute> &fieldNames,
                                        SmallVectorImpl<Type> &fieldTypes) {

  llvm::StringMap<SMLoc> nameLocMap;
  auto parseElt = [&]() -> ParseResult {
    // Parse the field name.
    std::string fieldName;
    if (parser.parseKeywordOrString(&fieldName))
      return failure();
    SMLoc currLoc = parser.getCurrentLocation();
    if (nameLocMap.count(fieldName)) {
      parser.emitError(currLoc, "field \"")
          << fieldName << "\" is defined twice";
      parser.emitError(nameLocMap[fieldName]) << "previous definition is here";
      return failure();
    }
    nameLocMap[fieldName] = currLoc;
    fieldNames.push_back(StringAttr::get(parser.getContext(), fieldName));

    // Parse the field type.
    fieldTypes.emplace_back();
    if (parser.parseColonType(fieldTypes.back()))
      return failure();

    return success();
  };

  return parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                        parseElt);
}

static ParseResult parseClassLike(OpAsmParser &parser, OperationState &state) {
  // Parse the Class symbol name.
  StringAttr symName;
  if (parser.parseSymbolName(symName, mlir::SymbolTable::getSymbolAttrName(),
                             state.attributes))
    return failure();

  // Parse the formal parameters.
  SmallVector<OpAsmParser::Argument> args;
  if (parser.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/false))
    return failure();

  SmallVector<Type> fieldTypes;
  SmallVector<Attribute> fieldNames;
  if (succeeded(parser.parseOptionalArrow()))
    if (failed(parseClassFieldsList(parser, fieldNames, fieldTypes)))
      return failure();

  SmallVector<NamedAttribute> fieldTypesMap;
  if (!fieldNames.empty()) {
    for (auto [name, type] : zip(fieldNames, fieldTypes))
      fieldTypesMap.push_back(
          NamedAttribute(cast<StringAttr>(name), TypeAttr::get(type)));
  }
  auto *ctx = parser.getContext();
  state.addAttribute("fieldNames", mlir::ArrayAttr::get(ctx, fieldNames));
  state.addAttribute("fieldTypes",
                     mlir::DictionaryAttr::get(ctx, fieldTypesMap));

  // Parse the optional attribute dictionary.
  if (failed(parser.parseOptionalAttrDictWithKeyword(state.attributes)))
    return failure();

  // Parse the body.
  Region *region = state.addRegion();
  if (parser.parseRegion(*region, args))
    return failure();

  // If the region was empty, add an empty block so it's still a SizedRegion<1>.
  if (region->empty())
    region->emplaceBlock();

  // Remember the formal parameter names in an attribute.
  auto argNames = llvm::map_range(args, [&](OpAsmParser::Argument arg) {
    return StringAttr::get(parser.getContext(), arg.ssaName.name.drop_front());
  });
  state.addAttribute(
      "formalParamNames",
      ArrayAttr::get(parser.getContext(), SmallVector<Attribute>(argNames)));

  return success();
}

static void printClassLike(ClassLike classLike, OpAsmPrinter &printer) {
  // Print the Class symbol name.
  printer << " @";
  printer << classLike.getSymName();

  // Retrieve the formal parameter names and values.
  auto argNames = SmallVector<StringRef>(
      classLike.getFormalParamNames().getAsValueRange<StringAttr>());
  ArrayRef<BlockArgument> args = classLike.getBodyBlock()->getArguments();

  // Print the formal parameters.
  printer << '(';
  for (size_t i = 0, e = args.size(); i < e; ++i) {
    printer << '%' << argNames[i] << ": " << args[i].getType();
    if (i < e - 1)
      printer << ", ";
  }
  printer << ") ";

  ArrayRef<Attribute> fieldNames =
      cast<ArrayAttr>(classLike->getAttr("fieldNames")).getValue();

  if (!fieldNames.empty()) {
    printer << " -> (";
    for (size_t i = 0, e = fieldNames.size(); i < e; ++i) {
      if (i != 0)
        printer << ", ";
      StringAttr name = cast<StringAttr>(fieldNames[i]);
      printer.printKeywordOrString(name.getValue());
      printer << ": ";
      Type type = classLike.getFieldType(name).value();
      printer.printType(type);
    }
    printer << ") ";
  }

  // Print the optional attribute dictionary.
  SmallVector<StringRef> elidedAttrs{classLike.getSymNameAttrName(),
                                     classLike.getFormalParamNamesAttrName(),
                                     "fieldTypes", "fieldNames"};
  printer.printOptionalAttrDictWithKeyword(classLike.getOperation()->getAttrs(),
                                           elidedAttrs);

  // Print the body.
  printer.printRegion(classLike.getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}

LogicalResult verifyClassLike(ClassLike classLike) {
  // Verify the formal parameter names match up with the values.
  if (classLike.getFormalParamNames().size() !=
      classLike.getBodyBlock()->getArguments().size()) {
    auto error = classLike.emitOpError(
        "formal parameter name list doesn't match formal parameter value list");
    error.attachNote(classLike.getLoc())
        << "formal parameter names: " << classLike.getFormalParamNames();
    error.attachNote(classLike.getLoc())
        << "formal parameter values: "
        << classLike.getBodyBlock()->getArguments();
    return error;
  }

  return success();
}

void getClassLikeAsmBlockArgumentNames(ClassLike classLike, Region &region,
                                       OpAsmSetValueNameFn setNameFn) {
  // Retrieve the formal parameter names and values.
  auto argNames = SmallVector<StringRef>(
      classLike.getFormalParamNames().getAsValueRange<StringAttr>());
  ArrayRef<BlockArgument> args = classLike.getBodyBlock()->getArguments();

  // Use the formal parameter names as the SSA value names.
  for (size_t i = 0, e = args.size(); i < e; ++i)
    setNameFn(args[i], argNames[i]);
}

NamedAttribute makeFieldType(StringAttr name, Type type) {
  return NamedAttribute(name, TypeAttr::get(type));
}

NamedAttribute makeFieldIdx(MLIRContext *ctx, mlir::StringAttr name,
                            unsigned i) {
  return NamedAttribute(StringAttr(name),
                        mlir::IntegerAttr::get(mlir::IndexType::get(ctx), i));
}

std::optional<Type> getClassLikeFieldType(ClassLike classLike,
                                          StringAttr name) {
  DictionaryAttr fieldTypes = mlir::cast<DictionaryAttr>(
      classLike.getOperation()->getAttr("fieldTypes"));
  Attribute type = fieldTypes.get(name);
  if (!type)
    return std::nullopt;
  return cast<TypeAttr>(type).getValue();
}

void replaceClassLikeFieldTypes(ClassLike classLike,
                                AttrTypeReplacer &replacer) {
  classLike->setAttr("fieldTypes", cast<DictionaryAttr>(replacer.replace(
                                       classLike.getFieldTypes())));
}

ArrayAttr buildFieldNames(OpBuilder &odsBuilder, DictionaryAttr fieldTypes) {
  return odsBuilder.getArrayAttr(llvm::map_to_vector(
      fieldTypes.getValue(), [&](NamedAttribute field) -> Attribute {
        return cast<Attribute>(field.getName());
      }));
}

//===----------------------------------------------------------------------===//
// ClassOp
//===----------------------------------------------------------------------===//

ParseResult circt::om::ClassOp::parse(OpAsmParser &parser,
                                      OperationState &state) {
  return parseClassLike(parser, state);
}

circt::om::ClassOp circt::om::ClassOp::buildSimpleClassOp(
    OpBuilder &odsBuilder, Location loc, Twine name,
    ArrayRef<StringRef> formalParamNames, ArrayRef<StringRef> fieldNames,
    ArrayRef<Type> fieldTypes) {
  circt::om::ClassOp classOp = odsBuilder.create<circt::om::ClassOp>(
      loc, name, formalParamNames,
      odsBuilder.getDictionaryAttr(llvm::map_to_vector(
          llvm::zip(fieldNames, fieldTypes), [&](auto field) -> NamedAttribute {
            return NamedAttribute(odsBuilder.getStringAttr(std::get<0>(field)),
                                  TypeAttr::get(std::get<1>(field)));
          })));
  Block *body = &classOp.getRegion().emplaceBlock();
  auto prevLoc = odsBuilder.saveInsertionPoint();
  odsBuilder.setInsertionPointToEnd(body);
  odsBuilder.create<ClassFieldsOp>(
      loc, llvm::map_to_vector(fieldTypes, [&](Type type) -> Value {
        return body->addArgument(type, loc);
      }));
  odsBuilder.restoreInsertionPoint(prevLoc);

  return classOp;
}

void circt::om::ClassOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                               Twine name, ArrayRef<StringRef> formalParamNames,
                               DictionaryAttr fieldTypes) {
  build(odsBuilder, odsState, odsBuilder.getStringAttr(name),
        odsBuilder.getStrArrayAttr(formalParamNames),
        buildFieldNames(odsBuilder, fieldTypes), fieldTypes);
}

void circt::om::ClassOp::print(OpAsmPrinter &printer) {
  printClassLike(*this, printer);
}

LogicalResult circt::om::ClassOp::verify() { return verifyClassLike(*this); }

void circt::om::ClassOp::getAsmBlockArgumentNames(
    Region &region, OpAsmSetValueNameFn setNameFn) {
  getClassLikeAsmBlockArgumentNames(*this, region, setNameFn);
}

std::optional<mlir::Type>
circt::om::ClassOp::getFieldType(mlir::StringAttr field) {
  return getClassLikeFieldType(*this, field);
}

void circt::om::ClassOp::replaceFieldTypes(AttrTypeReplacer replacer) {
  replaceClassLikeFieldTypes(*this, replacer);
}

//===----------------------------------------------------------------------===//
// ClassExternOp
//===----------------------------------------------------------------------===//

ParseResult circt::om::ClassExternOp::parse(OpAsmParser &parser,
                                            OperationState &state) {
  return parseClassLike(parser, state);
}

void circt::om::ClassExternOp::build(OpBuilder &odsBuilder,
                                     OperationState &odsState, Twine name,
                                     ArrayRef<StringRef> formalParamNames,
                                     DictionaryAttr fieldTypes) {
  build(odsBuilder, odsState, odsBuilder.getStringAttr(name),
        odsBuilder.getStrArrayAttr(formalParamNames),
        buildFieldNames(odsBuilder, fieldTypes), fieldTypes);
}

void circt::om::ClassExternOp::print(OpAsmPrinter &printer) {
  printClassLike(*this, printer);
}

LogicalResult circt::om::ClassExternOp::verify() {
  if (failed(verifyClassLike(*this))) {
    return failure();
  }
  // Verify body is empty
  if (!this->getBodyBlock()->getOperations().empty()) {
    return this->emitOpError("external class body should be empty");
  }

  return success();
}

void circt::om::ClassExternOp::getAsmBlockArgumentNames(
    Region &region, OpAsmSetValueNameFn setNameFn) {
  getClassLikeAsmBlockArgumentNames(*this, region, setNameFn);
}

std::optional<mlir::Type>
circt::om::ClassExternOp::getFieldType(mlir::StringAttr field) {
  return getClassLikeFieldType(*this, field);
}

void circt::om::ClassExternOp::replaceFieldTypes(AttrTypeReplacer replacer) {
  replaceClassLikeFieldTypes(*this, replacer);
}

//===----------------------------------------------------------------------===//
// ObjectOp
//===----------------------------------------------------------------------===//

void circt::om::ObjectOp::build(::mlir::OpBuilder &odsBuilder,
                                ::mlir::OperationState &odsState,
                                om::ClassOp classOp,
                                ::mlir::ValueRange actualParams) {
  return build(odsBuilder, odsState,
               om::ClassType::get(odsBuilder.getContext(),
                                  mlir::FlatSymbolRefAttr::get(classOp)),
               classOp.getNameAttr(), actualParams);
}

LogicalResult
circt::om::ObjectOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verify the result type is the same as the referred-to class.
  StringAttr resultClassName = getResult().getType().getClassName().getAttr();
  StringAttr className = getClassNameAttr();
  if (resultClassName != className)
    return emitOpError("result type (")
           << resultClassName << ") does not match referred to class ("
           << className << ')';

  // Verify the referred to ClassOp exists.
  auto classDef = dyn_cast_or_null<ClassLike>(
      symbolTable.lookupNearestSymbolFrom(*this, className));
  if (!classDef)
    return emitOpError("refers to non-existant class (") << className << ')';

  auto actualTypes = getActualParams().getTypes();
  auto formalTypes = classDef.getBodyBlock()->getArgumentTypes();

  // Verify the actual parameter list matches the formal parameter list.
  if (actualTypes.size() != formalTypes.size()) {
    auto error = emitOpError(
        "actual parameter list doesn't match formal parameter list");
    error.attachNote(classDef.getLoc())
        << "formal parameters: " << classDef.getBodyBlock()->getArguments();
    error.attachNote(getLoc()) << "actual parameters: " << getActualParams();
    return error;
  }

  // Verify the actual parameter types match the formal parameter types.
  for (size_t i = 0, e = actualTypes.size(); i < e; ++i) {
    if (actualTypes[i] != formalTypes[i]) {
      return emitOpError("actual parameter type (")
             << actualTypes[i] << ") doesn't match formal parameter type ("
             << formalTypes[i] << ')';
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void circt::om::ConstantOp::build(::mlir::OpBuilder &odsBuilder,
                                  ::mlir::OperationState &odsState,
                                  ::mlir::TypedAttr constVal) {
  return build(odsBuilder, odsState, constVal.getType(), constVal);
}

OpFoldResult circt::om::ConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// ListCreateOp
//===----------------------------------------------------------------------===//

void circt::om::ListCreateOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printOperands(getInputs());
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getType().getElementType();
}

ParseResult circt::om::ListCreateOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 16> operands;
  Type elemType;

  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(elemType))
    return failure();
  result.addTypes({circt::om::ListType::get(elemType)});

  for (auto operand : operands)
    if (parser.resolveOperand(operand, elemType, result.operands))
      return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// TupleCreateOp
//===----------------------------------------------------------------------===//

LogicalResult TupleCreateOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties, RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
  ::llvm::SmallVector<Type> types;
  for (auto op : operands)
    types.push_back(op.getType());
  inferredReturnTypes.push_back(TupleType::get(context, types));
  return success();
}

//===----------------------------------------------------------------------===//
// TupleGetOp
//===----------------------------------------------------------------------===//

LogicalResult TupleGetOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties, RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
  auto idx = attributes.getAs<mlir::IntegerAttr>("index");
  if (operands.empty() || !idx)
    return failure();

  auto tupleTypes = cast<TupleType>(operands[0].getType()).getTypes();
  if (tupleTypes.size() <= idx.getValue().getLimitedValue()) {
    if (location)
      mlir::emitError(*location,
                      "tuple index out-of-bounds, must be less than ")
          << tupleTypes.size() << " but got "
          << idx.getValue().getLimitedValue();
    return failure();
  }

  inferredReturnTypes.push_back(tupleTypes[idx.getValue().getLimitedValue()]);
  return success();
}

//===----------------------------------------------------------------------===//
// MapCreateOp
//===----------------------------------------------------------------------===//

void circt::om::MapCreateOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printOperands(getInputs());
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << cast<circt::om::MapType>(getType()).getKeyType() << ", "
    << cast<circt::om::MapType>(getType()).getValueType();
}

ParseResult circt::om::MapCreateOp::parse(OpAsmParser &parser,
                                          OperationState &result) {
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 16> operands;
  Type elementType, valueType;

  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(elementType) || parser.parseComma() ||
      parser.parseType(valueType))
    return failure();
  result.addTypes({circt::om::MapType::get(elementType, valueType)});
  auto operandType =
      mlir::TupleType::get(valueType.getContext(), {elementType, valueType});

  for (auto operand : operands)
    if (parser.resolveOperand(operand, operandType, result.operands))
      return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// BasePathCreateOp
//===----------------------------------------------------------------------===//

LogicalResult
BasePathCreateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto hierPath = symbolTable.lookupNearestSymbolFrom<hw::HierPathOp>(
      *this, getTargetAttr());
  if (!hierPath)
    return emitOpError("invalid symbol reference");
  return success();
}

//===----------------------------------------------------------------------===//
// PathCreateOp
//===----------------------------------------------------------------------===//

LogicalResult
PathCreateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto hierPath = symbolTable.lookupNearestSymbolFrom<hw::HierPathOp>(
      *this, getTargetAttr());
  if (!hierPath)
    return emitOpError("invalid symbol reference");
  return success();
}

//===----------------------------------------------------------------------===//
// IntegerAddOp
//===----------------------------------------------------------------------===//

FailureOr<llvm::APSInt>
IntegerAddOp::evaluateIntegerOperation(const llvm::APSInt &lhs,
                                       const llvm::APSInt &rhs) {
  return success(lhs + rhs);
}

//===----------------------------------------------------------------------===//
// IntegerMulOp
//===----------------------------------------------------------------------===//

FailureOr<llvm::APSInt>
IntegerMulOp::evaluateIntegerOperation(const llvm::APSInt &lhs,
                                       const llvm::APSInt &rhs) {
  return success(lhs * rhs);
}

//===----------------------------------------------------------------------===//
// IntegerShrOp
//===----------------------------------------------------------------------===//

FailureOr<llvm::APSInt>
IntegerShrOp::evaluateIntegerOperation(const llvm::APSInt &lhs,
                                       const llvm::APSInt &rhs) {
  // Check non-negative constraint from operation semantics.
  if (!rhs.isNonNegative())
    return emitOpError("shift amount must be non-negative");
  // Check size constraint from implementation detail of using getExtValue.
  if (!rhs.isRepresentableByInt64())
    return emitOpError("shift amount must be representable in 64 bits");
  return success(lhs >> rhs.getExtValue());
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "circt/Dialect/OM/OM.cpp.inc"
