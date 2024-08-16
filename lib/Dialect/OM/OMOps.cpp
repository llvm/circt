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

  // Print the optional attribute dictionary.
  SmallVector<StringRef> elidedAttrs{classLike.getSymNameAttrName(),
                                     classLike.getFormalParamNamesAttrName()};
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

//===----------------------------------------------------------------------===//
// ClassOp
//===----------------------------------------------------------------------===//

ParseResult circt::om::ClassOp::parse(OpAsmParser &parser,
                                      OperationState &state) {
  return parseClassLike(parser, state);
}

void circt::om::ClassOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                               Twine name,
                               ArrayRef<StringRef> formalParamNames) {
  return build(odsBuilder, odsState, odsBuilder.getStringAttr(name),
               odsBuilder.getStrArrayAttr(formalParamNames));
}

circt::om::ClassOp circt::om::ClassOp::buildSimpleClassOp(
    OpBuilder &odsBuilder, Location loc, Twine name,
    ArrayRef<StringRef> formalParamNames, ArrayRef<StringRef> fieldNames,
    ArrayRef<Type> fieldTypes) {
  circt::om::ClassOp classOp = odsBuilder.create<circt::om::ClassOp>(
      loc, odsBuilder.getStringAttr(name),
      odsBuilder.getStrArrayAttr(formalParamNames));
  Block *body = &classOp.getRegion().emplaceBlock();
  auto prevLoc = odsBuilder.saveInsertionPoint();
  odsBuilder.setInsertionPointToEnd(body);
  SmallVector<Value> args =
      llvm::map_to_vector(fieldTypes, [&](Type type) -> Value {
        return body->addArgument(type, loc);
      });
  SmallVector<Attribute> fields =
      llvm::map_to_vector(fieldNames, [&](StringRef name) -> Attribute {
        return StringAttr::get(classOp.getContext(), name);
      });
  classOp.addFields(odsBuilder, loc, fields, args);
  odsBuilder.restoreInsertionPoint(prevLoc);

  return classOp;
}

void circt::om::ClassOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                               Twine name) {
  return build(odsBuilder, odsState, odsBuilder.getStringAttr(name),
               odsBuilder.getStrArrayAttr({}));
}

void circt::om::ClassOp::print(OpAsmPrinter &printer) {
  printClassLike(*this, printer);
}

LogicalResult circt::om::ClassOp::verify() { return verifyClassLike(*this); }

void circt::om::ClassOp::getAsmBlockArgumentNames(
    Region &region, OpAsmSetValueNameFn setNameFn) {
  getClassLikeAsmBlockArgumentNames(*this, region, setNameFn);
}

void circt::om::ClassOp::addFields(mlir::OpBuilder &builder, mlir::Location loc,
                                   llvm::ArrayRef<mlir::Attribute> fieldNames,
                                   llvm::ArrayRef<mlir::Value> fieldValues) {
  ClassFieldsOp op = builder.create<ClassFieldsOp>(loc, fieldValues);
  llvm::SmallVector<NamedAttribute> fieldTypes;
  for (auto [name, value] : llvm::zip(fieldNames, fieldValues)) {
    fieldTypes.push_back(mlir::NamedAttribute(
        cast<StringAttr>(name), mlir::TypeAttr::get(value.getType())));
  }
  op.getOperation()->setAttr(
      "fieldNames", mlir::ArrayAttr::get(this->getContext(), fieldNames));
  op.getOperation()->setAttr(
      "fieldTypes", mlir::DictionaryAttr::get(this->getContext(), fieldTypes));
}

void circt::om::ClassOp::addFields(mlir::OpBuilder &builder,
                                   llvm::ArrayRef<mlir::Location> locs,
                                   llvm::ArrayRef<mlir::Attribute> fieldNames,
                                   llvm::ArrayRef<mlir::Value> fieldValues) {
  this->addFields(builder, builder.getFusedLoc(locs), fieldNames, fieldValues);
}

void circt::om::ClassOp::addFields(mlir::OpBuilder &builder, mlir::Location loc,
                                   llvm::ArrayRef<llvm::StringRef> fieldNames,
                                   llvm::ArrayRef<mlir::Value> fieldValues) {
  this->addFields(builder, loc,
                  llvm::map_to_vector(fieldNames,
                                      [&](llvm::StringRef fieldName) {
                                        return cast<mlir::Attribute>(
                                            builder.getStringAttr(fieldName));
                                      }),
                  fieldValues);
}

//===----------------------------------------------------------------------===//
// ClassFieldsOpField
//===----------------------------------------------------------------------===//

struct FieldParse : OpAsmParser::Argument {
  StringAttr name;
};

static ParseResult parseFieldName(OpAsmParser &parser, StringAttr &name) {
  if (failed(parser.parseSymbolName(name)))
    return parser.emitError(parser.getCurrentLocation(), "expected field name");
  return success();
}

static ParseResult
parseField(OpAsmParser &parser,
           llvm::MapVector<StringAttr, SMLoc> &parsedFieldNames,

           FieldParse &result, bool hasOperand = true) {
  NamedAttrList attrs;
  if (parseFieldName(parser, result.name))
    return failure();
  SMLoc currLoc = parser.getCurrentLocation();
  if (parsedFieldNames.contains(result.name)) {
    parser.emitError(currLoc, "op field ")
        << result.name << " is defined twice";
    // TODO: Is there a way to attach a note to a parser error?
    parser.emitError(parsedFieldNames.lookup(result.name))
        << "previous definition is here";
    return failure();
  }
  parsedFieldNames[result.name] = currLoc;
  if (hasOperand && parser.parseOperand(result.ssaName))
    return failure();
  if (parser.parseColonType(result.type) ||
      parser.parseOptionalAttrDict(attrs) ||
      parser.parseOptionalLocationSpecifier(result.sourceLoc))
    return failure();
  result.attrs = attrs.getDictionary(parser.getContext());
  return success();
}

ParseResult buildFieldAttrs(OperationState &state, OpAsmParser &parser,
                     llvm::SmallVector<FieldParse> &parsedFields,
                     bool resolveOperands=false) {
  mlir::MLIRContext *ctx = parser.getContext();
  llvm::SmallVector<Attribute> fieldNames;
  llvm::SmallVector<NamedAttribute> fieldTypes;
  for (auto &field : parsedFields) {
    fieldTypes.push_back(mlir::NamedAttribute(mlir::StringAttr(field.name),
                                              mlir::TypeAttr::get(field.type)));
    fieldNames.push_back(field.name);
    if (resolveOperands &&
        parser.resolveOperand(field.ssaName, field.type, state.operands))
      return failure();
  }
  state.addAttribute("fieldTypes", mlir::DictionaryAttr::get(ctx, fieldTypes));
  state.addAttribute("fieldNames", mlir::ArrayAttr::get(ctx, fieldNames));
  return success();
}

ParseResult circt::om::ClassFieldsOp::parse(OpAsmParser &parser,
                                            OperationState &state) {
  llvm::SmallVector<FieldParse> parsedFields;
  llvm::MapVector<StringAttr, SMLoc> parsedFieldNames;
  auto parseOnePort = [&]() -> ParseResult {
    return parseField(parser, parsedFieldNames, parsedFields.emplace_back());
  };
  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                     parseOnePort, " in field list"))
    return failure();

  if (buildFieldAttrs(state, parser, parsedFields, true))
    return failure();

  // TODO: field attrs
  // TODO: field locs
  return success();
}

void circt::om::ClassFieldsOp::print(OpAsmPrinter &printer) {
  printer << "(";
  printer.increaseIndent();
  auto operands = this->getOperands();
  auto names = this->getFieldNames();
  for (unsigned i = 0; i < operands.size(); i++) {
    if (i > 0) {
      printer << ",";
    }
    printer.printNewline();
    printer.printSymbolName(cast<StringAttr>(names[i]));
    printer << " ";
    printer.printOperand(operands[i]);
    printer << " : ";
    printer.printType(operands[i].getType());
  }
  printer.decreaseIndent();
  if (!operands.empty())
    printer.printNewline();
  printer << ")";
  // printer.printRegion(this->getBody());
  // TODO: attrs
  // printer.printOptionalAttrDictWithKeyword(this->getOperation()->getAttrs());
}

//===----------------------------------------------------------------------===//
// ClassExternOp
//===----------------------------------------------------------------------===//

ParseResult circt::om::ClassExternOp::parse(OpAsmParser &parser,
                                            OperationState &state) {
  return parseClassLike(parser, state);
}

void circt::om::ClassExternOp::build(OpBuilder &odsBuilder,
                                     OperationState &odsState, Twine name) {
  return build(odsBuilder, odsState, odsBuilder.getStringAttr(name),
               odsBuilder.getStrArrayAttr({}));
}

void circt::om::ClassExternOp::build(OpBuilder &odsBuilder,
                                     OperationState &odsState, Twine name,
                                     ArrayRef<StringRef> formalParamNames) {
  return build(odsBuilder, odsState, odsBuilder.getStringAttr(name),
               odsBuilder.getStrArrayAttr(formalParamNames));
}

void circt::om::ClassExternOp::print(OpAsmPrinter &printer) {
  printClassLike(*this, printer);
}

LogicalResult circt::om::ClassExternOp::verify() {
  if (failed(verifyClassLike(*this))) {
    return failure();
  }

  // Verify that only external class field declarations are present in the body.
  for (auto &op : getOps())
    if (!isa<ClassExternFieldsOp>(op))
      return op.emitOpError("not allowed in external class");

  return success();
}

void circt::om::ClassExternOp::getAsmBlockArgumentNames(
    Region &region, OpAsmSetValueNameFn setNameFn) {
  getClassLikeAsmBlockArgumentNames(*this, region, setNameFn);
}

void circt::om::ClassExternOp::addFields(
    mlir::OpBuilder &builder, mlir::Location loc,
    llvm::ArrayRef<mlir::StringAttr> fieldNames,
    llvm::ArrayRef<mlir::Type> fieldTypes) {
  auto *op = builder.create<ClassExternFieldsOp>(loc).getOperation();
  // TODO: Merge with buildFieldAttrs code
  auto *ctx = builder.getContext();
  llvm::SmallVector<NamedAttribute> namedAttrs;
  llvm::SmallVector<Attribute> fieldAttrs;
  for (auto [name, type] : llvm::zip(fieldNames, fieldTypes)) {
    namedAttrs.push_back(mlir::NamedAttribute(mlir::StringAttr(name),
                                              mlir::TypeAttr::get(type)));
    fieldAttrs.push_back(cast<Attribute>(name));
  }
  op->setAttr("fieldNames", mlir::ArrayAttr::get(ctx, fieldAttrs));
  op->setAttr("fieldTypes", mlir::DictionaryAttr::get(ctx, namedAttrs));
}

void circt::om::ClassExternOp::addFields(
    mlir::OpBuilder &builder, llvm::ArrayRef<mlir::Location> locs,
    llvm::ArrayRef<mlir::StringAttr> fieldNames,
    llvm::ArrayRef<mlir::Type> fieldTypes) {
  this->addFields(builder, builder.getFusedLoc(locs), fieldNames, fieldTypes);
}

//===----------------------------------------------------------------------===//
// ClassExternFieldsOp
//===----------------------------------------------------------------------===//

ParseResult circt::om::ClassExternFieldsOp::parse(OpAsmParser &parser,
                                                  OperationState &state) {
  llvm::SmallVector<FieldParse> parsedFields;
  llvm::MapVector<StringAttr, SMLoc> parsedFieldNames;
  auto parseOnePort = [&]() -> ParseResult {
    return parseField(parser, parsedFieldNames, parsedFields.emplace_back(),
                      false);
  };
  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                     parseOnePort, " in field list"))
    return failure();

  if (buildFieldAttrs(state, parser, parsedFields))
    return failure();

  // TODO: field attrs
  // TODO: field locs
  return success();
}

void circt::om::ClassExternFieldsOp::print(OpAsmPrinter &printer) {
  printer << "(";
  printer.increaseIndent();
  mlir::ArrayAttr fieldNames =
      cast<ArrayAttr>(this->getOperation()->getAttr("fieldNames"));
  mlir::DictionaryAttr fieldTypes =
      cast<DictionaryAttr>(this->getOperation()->getAttr("fieldTypes"));
  for (unsigned i = 0; i < fieldNames.size(); i++) {
    auto name = cast<StringAttr>(fieldNames[i]).getValue();
    auto type = cast<TypeAttr>(fieldTypes.get(name)).getValue();
    if (i > 0) {
      printer << ",";
    }
    printer.printNewline();
    printer.printSymbolName(name);
    printer << " : ";
    printer.printType(type);
  }
  printer.decreaseIndent();
  if (!fieldNames.empty())
    printer.printNewline();
  printer << ")";
  // TODO: attrs
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
