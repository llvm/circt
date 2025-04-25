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
// Custom Printers and Parsers
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

static ParseResult parseFieldLocs(OpAsmParser &parser, ArrayAttr &fieldLocs) {
  if (parser.parseOptionalKeyword("field_locs"))
    return success();
  if (parser.parseLParen() || parser.parseAttribute(fieldLocs) ||
      parser.parseRParen()) {
    return failure();
  }
  return success();
}

static void printFieldLocs(OpAsmPrinter &printer, Operation *op,
                           ArrayAttr fieldLocs) {
  mlir::OpPrintingFlags flags;
  if (!flags.shouldPrintDebugInfo() || !fieldLocs)
    return;
  printer << "field_locs(";
  printer.printAttribute(fieldLocs);
  printer << ")";
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
      loc, odsBuilder.getStringAttr(name),
      odsBuilder.getStrArrayAttr(formalParamNames),
      odsBuilder.getStrArrayAttr(fieldNames),
      odsBuilder.getDictionaryAttr(llvm::map_to_vector(
          llvm::zip(fieldNames, fieldTypes), [&](auto field) -> NamedAttribute {
            return NamedAttribute(odsBuilder.getStringAttr(std::get<0>(field)),
                                  TypeAttr::get(std::get<1>(field)));
          })));
  Block *body = &classOp.getRegion().emplaceBlock();
  auto prevLoc = odsBuilder.saveInsertionPoint();
  odsBuilder.setInsertionPointToEnd(body);

  mlir::SmallVector<Attribute> locAttrs(fieldNames.size(), LocationAttr(loc));

  odsBuilder.create<ClassFieldsOp>(
      loc,
      llvm::map_to_vector(
          fieldTypes,
          [&](Type type) -> Value { return body->addArgument(type, loc); }),
      odsBuilder.getArrayAttr(locAttrs));

  odsBuilder.restoreInsertionPoint(prevLoc);

  return classOp;
}

void circt::om::ClassOp::print(OpAsmPrinter &printer) {
  printClassLike(*this, printer);
}

LogicalResult circt::om::ClassOp::verify() { return verifyClassLike(*this); }

LogicalResult circt::om::ClassOp::verifyRegions() {
  auto fieldsOp = cast<ClassFieldsOp>(this->getBodyBlock()->getTerminator());

  // The number of results matches the number of terminator operands.
  if (fieldsOp.getNumOperands() != this->getFieldNames().size()) {
    auto diag = this->emitOpError()
                << "returns '" << this->getFieldNames().size()
                << "' fields, but its terminator returned '"
                << fieldsOp.getNumOperands() << "' fields";
    return diag.attachNote(fieldsOp.getLoc()) << "see terminator:";
  }

  // The type of each result matches the corresponding terminator operand type.
  auto types = this->getFieldTypes();
  for (auto [fieldName, terminatorOperandType] :
       llvm::zip(this->getFieldNames(), fieldsOp.getOperandTypes())) {

    if (terminatorOperandType ==
        cast<TypeAttr>(types.get(cast<StringAttr>(fieldName))).getValue())
      continue;

    auto diag = this->emitOpError()
                << "returns different field types than its terminator";
    return diag.attachNote(fieldsOp.getLoc()) << "see terminator:";
  }

  return success();
}

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

void circt::om::ClassOp::updateFields(
    mlir::ArrayRef<mlir::Location> newLocations,
    mlir::ArrayRef<mlir::Value> newValues,
    mlir::ArrayRef<mlir::Attribute> newNames) {

  auto fieldsOp = getFieldsOp();
  assert(fieldsOp && "The fields op should exist");
  // Get field names.
  SmallVector<Attribute> names(getFieldNamesAttr().getAsRange<StringAttr>());
  // Get the field types.
  SmallVector<NamedAttribute> fieldTypes(getFieldTypesAttr().getValue());
  // Get the field values.
  SmallVector<Value> fieldVals(fieldsOp.getFields());
  // Get the field locations.
  Location fieldOpLoc = fieldsOp->getLoc();

  // Extract the locations per field.
  SmallVector<Location> locations;
  if (auto fl = dyn_cast<FusedLoc>(fieldOpLoc)) {
    auto metadataArr = dyn_cast<ArrayAttr>(fl.getMetadata());
    assert(metadataArr && "Expected the metadata for the fused location");
    auto r = metadataArr.getAsRange<LocationAttr>();
    locations.append(r.begin(), r.end());
  } else {
    // Assume same loc for every field.
    locations.append(names.size(), fieldOpLoc);
  }

  // Append the new names, locations and values.
  names.append(newNames.begin(), newNames.end());
  locations.append(newLocations.begin(), newLocations.end());
  fieldVals.append(newValues.begin(), newValues.end());

  // Construct the new field types from values and names.
  for (auto [v, n] : llvm::zip(newValues, newNames))
    fieldTypes.emplace_back(
        NamedAttribute(llvm::cast<StringAttr>(n), TypeAttr::get(v.getType())));

  // Keep the locations as array on the metadata.
  SmallVector<Attribute> locationsAttr;
  llvm::for_each(locations, [&](Location &l) {
    locationsAttr.push_back(cast<Attribute>(l));
  });

  ImplicitLocOpBuilder builder(getLoc(), *this);
  // Update the field names attribute.
  setFieldNamesAttr(builder.getArrayAttr(names));
  // Update the fields type attribute.
  setFieldTypesAttr(builder.getDictionaryAttr(fieldTypes));
  fieldsOp.getFieldsMutable().assign(fieldVals);
  // Update the location.
  fieldsOp->setLoc(builder.getFusedLoc(
      locations, ArrayAttr::get(getContext(), locationsAttr)));
}

void circt::om::ClassOp::addNewFieldsOp(mlir::OpBuilder &builder,
                                        mlir::ArrayRef<Location> locs,
                                        mlir::ArrayRef<Value> values) {
  // Store the original locations as a metadata array so that unique locations
  // are preserved as a mapping from field index to location
  assert(locs.size() == values.size() && "Expected a location per value");
  mlir::SmallVector<Attribute> locAttrs;
  for (auto loc : locs) {
    locAttrs.push_back(cast<Attribute>(LocationAttr(loc)));
  }
  // Also store the locations incase there's some other analysis that might
  // be able to use the default FusedLoc representation.
  builder.create<ClassFieldsOp>(builder.getFusedLoc(locs), values,
                                builder.getArrayAttr(locAttrs));
}

mlir::Location circt::om::ClassOp::getFieldLocByIndex(size_t i) {
  auto fieldLocs = this->getFieldsOp().getFieldLocs();
  if (!fieldLocs.has_value())
    return UnknownLoc::get(this->getContext());
  assert(i < fieldLocs.value().size() &&
         "field index too large for location array");
  return cast<LocationAttr>(fieldLocs.value()[i]);
}

//===----------------------------------------------------------------------===//
// ClassExternOp
//===----------------------------------------------------------------------===//

ParseResult circt::om::ClassExternOp::parse(OpAsmParser &parser,
                                            OperationState &state) {
  return parseClassLike(parser, state);
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
// ClassFieldsOp
//===----------------------------------------------------------------------===//
//
LogicalResult circt::om::ClassFieldsOp::verify() {
  auto fieldLocs = this->getFieldLocs();
  if (fieldLocs.has_value()) {
    auto fieldLocsVal = fieldLocs.value();
    if (fieldLocsVal.size() != this->getFields().size()) {
      auto error = this->emitOpError("size of field_locs (")
                   << fieldLocsVal.size()
                   << ") does not match number of fields ("
                   << this->getFields().size() << ")";
    }
  }
  return success();
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
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);
  auto idx = adaptor.getIndexAttr();
  if (operands.empty() || !idx)
    return failure();

  auto tupleTypes = cast<TupleType>(adaptor.getInput().getType()).getTypes();
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
// IntegerShlOp
//===----------------------------------------------------------------------===//

FailureOr<llvm::APSInt>
IntegerShlOp::evaluateIntegerOperation(const llvm::APSInt &lhs,
                                       const llvm::APSInt &rhs) {
  // Check non-negative constraint from operation semantics.
  if (!rhs.isNonNegative())
    return emitOpError("shift amount must be non-negative");
  // Check size constraint from implementation detail of using getExtValue.
  if (!rhs.isRepresentableByInt64())
    return emitOpError("shift amount must be representable in 64 bits");
  return success(lhs << rhs.getExtValue());
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "circt/Dialect/OM/OM.cpp.inc"
