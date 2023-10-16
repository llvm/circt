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
  for (auto [name, type] : llvm::zip(fieldNames, fieldTypes))
    odsBuilder.create<circt::om::ClassFieldOp>(loc, name,
                                               body->addArgument(type, loc));
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

//===----------------------------------------------------------------------===//
// ClassFieldOp
//===----------------------------------------------------------------------===//

Type circt::om::ClassFieldOp::getType() { return getValue().getType(); }

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
    if (!isa<ClassExternFieldOp>(op))
      return op.emitOpError("not allowed in external class");

  return success();
}

void circt::om::ClassExternOp::getAsmBlockArgumentNames(
    Region &region, OpAsmSetValueNameFn setNameFn) {
  getClassLikeAsmBlockArgumentNames(*this, region, setNameFn);
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
// ObjectFieldOp
//===----------------------------------------------------------------------===//

LogicalResult
circt::om::ObjectFieldOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Get the ObjectInstOp and the ClassLike it is an instance of.
  ObjectOp objectInst = getObject().getDefiningOp<ObjectOp>();
  ClassLike classDef = cast<ClassLike>(symbolTable.lookupNearestSymbolFrom(
      *this, objectInst.getClassNameAttr()));

  // Traverse the field path, verifying each field exists.
  ClassFieldLike finalField;
  auto fields = SmallVector<FlatSymbolRefAttr>(
      getFieldPath().getAsRange<FlatSymbolRefAttr>());
  for (size_t i = 0, e = fields.size(); i < e; ++i) {
    // Verify the field exists on the ClassOp.
    auto field = fields[i];
    ClassFieldLike fieldDef;
    classDef.walk([&](SymbolOpInterface symbol) {
      if (auto fieldLike = dyn_cast<ClassFieldLike>(symbol.getOperation())) {
        if (symbol.getNameAttr() == field.getAttr()) {
          fieldDef = fieldLike;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (!fieldDef) {
      auto error = emitOpError("referenced non-existant field ") << field;
      error.attachNote(classDef.getLoc()) << "class defined here";
      return error;
    }

    // If there are more fields, verify the current field is of ClassType, and
    // look up the ClassOp for that field.
    if (i < e - 1) {
      auto classType = fieldDef.getType().dyn_cast<ClassType>();
      if (!classType)
        return emitOpError("nested field access into ")
               << field << " requires a ClassType, but found "
               << fieldDef.getType();

      // The nested ClassOp must exist, since a field with ClassType must be
      // an ObjectInstOp, which already verifies the class exists.
      classDef = cast<ClassLike>(
          symbolTable.lookupNearestSymbolFrom(*this, classType.getClassName()));

      // Proceed to the next field in the path.
      continue;
    }

    // On the last iteration down the path, save the final field being accessed.
    finalField = fieldDef;
  }

  // Verify the accessed field type matches the result type.
  if (finalField.getType() != getResult().getType())
    return emitOpError("expected type ")
           << getResult().getType() << ", but accessed field has type "
           << finalField.getType();

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

  auto tupleTypes = operands[0].getType().cast<TupleType>().getTypes();
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
  p << " : " << getType().cast<circt::om::MapType>().getKeyType() << ", "
    << getType().cast<circt::om::MapType>().getValueType();
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
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "circt/Dialect/OM/OM.cpp.inc"
