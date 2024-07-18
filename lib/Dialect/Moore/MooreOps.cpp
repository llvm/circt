//===- MooreOps.cpp - Implement the Moore operations ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Moore dialect operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/HW/CustomDirectiveImpl.h"
#include "circt/Dialect/HW/ModuleImplementation.h"
#include "circt/Support/CustomDirectiveImpl.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::moore;
using namespace mlir;

//===----------------------------------------------------------------------===//
// SVModuleOp
//===----------------------------------------------------------------------===//

void SVModuleOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       llvm::StringRef name, hw::ModuleType type) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getModuleTypeAttrName(state.name), TypeAttr::get(type));
  state.addRegion();
}

void SVModuleOp::print(OpAsmPrinter &p) {
  p << " ";

  // Print the visibility of the module.
  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibility = (*this)->getAttrOfType<StringAttr>(visibilityAttrName))
    p << visibility.getValue() << ' ';

  p.printSymbolName(SymbolTable::getSymbolName(*this).getValue());
  hw::module_like_impl::printModuleSignatureNew(p, getBodyRegion(),
                                                getModuleType(), {}, {});
  p << " ";
  p.printRegion(getBodyRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);

  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                     getAttributeNames());
}

ParseResult SVModuleOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the visibility attribute.
  (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  // Parse the module name.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, getSymNameAttrName(result.name),
                             result.attributes))
    return failure();

  // Parse the ports.
  SmallVector<hw::module_like_impl::PortParse> ports;
  TypeAttr modType;
  if (failed(
          hw::module_like_impl::parseModuleSignature(parser, ports, modType)))
    return failure();
  result.addAttribute(getModuleTypeAttrName(result.name), modType);

  // Parse the attributes.
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  // Add the entry block arguments.
  SmallVector<OpAsmParser::Argument, 4> entryArgs;
  for (auto &port : ports)
    if (port.direction != hw::ModulePort::Direction::Output)
      entryArgs.push_back(port);

  // Parse the optional function body.
  auto &bodyRegion = *result.addRegion();
  if (parser.parseRegion(bodyRegion, entryArgs))
    return failure();

  ensureTerminator(bodyRegion, parser.getBuilder(), result.location);
  return success();
}

void SVModuleOp::getAsmBlockArgumentNames(mlir::Region &region,
                                          mlir::OpAsmSetValueNameFn setNameFn) {
  if (&region != &getBodyRegion())
    return;
  auto moduleType = getModuleType();
  for (auto [index, arg] : llvm::enumerate(region.front().getArguments()))
    setNameFn(arg, moduleType.getInputNameAttr(index));
}

OutputOp SVModuleOp::getOutputOp() {
  return cast<OutputOp>(getBody()->getTerminator());
}

OperandRange SVModuleOp::getOutputs() { return getOutputOp().getOperands(); }

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

LogicalResult OutputOp::verify() {
  auto module = getParentOp();

  // Check that the number of operands matches the number of output ports.
  auto outputTypes = module.getModuleType().getOutputTypes();
  if (outputTypes.size() != getNumOperands())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing module @"
           << module.getSymName() << " has " << outputTypes.size()
           << " outputs";

  // Check that the operand types match the output ports.
  for (unsigned i = 0, e = outputTypes.size(); i != e; ++i)
    if (outputTypes[i] != getOperand(i).getType())
      return emitOpError() << "operand " << i << " (" << getOperand(i).getType()
                           << ") does not match output type (" << outputTypes[i]
                           << ") of module @" << module.getSymName();

  return success();
}

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

LogicalResult InstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Resolve the target symbol.
  auto *symbol =
      symbolTable.lookupNearestSymbolFrom(*this, getModuleNameAttr());
  if (!symbol)
    return emitOpError("references unknown symbol @") << getModuleName();

  // Check that the symbol is a SVModuleOp.
  auto module = dyn_cast<SVModuleOp>(symbol);
  if (!module)
    return emitOpError("must reference a 'moore.module', but @")
           << getModuleName() << " is a '" << symbol->getName() << "'";

  // Check that the input ports match.
  auto moduleType = module.getModuleType();
  auto inputTypes = moduleType.getInputTypes();

  if (inputTypes.size() != getNumOperands())
    return emitOpError("has ")
           << getNumOperands() << " operands, but target module @"
           << module.getSymName() << " has " << inputTypes.size() << " inputs";

  for (unsigned i = 0, e = inputTypes.size(); i != e; ++i)
    if (inputTypes[i] != getOperand(i).getType())
      return emitOpError() << "operand " << i << " (" << getOperand(i).getType()
                           << ") does not match input type (" << inputTypes[i]
                           << ") of module @" << module.getSymName();

  // Check that the output ports match.
  auto outputTypes = moduleType.getOutputTypes();

  if (outputTypes.size() != getNumResults())
    return emitOpError("has ")
           << getNumOperands() << " results, but target module @"
           << module.getSymName() << " has " << outputTypes.size()
           << " outputs";

  for (unsigned i = 0, e = outputTypes.size(); i != e; ++i)
    if (outputTypes[i] != getResult(i).getType())
      return emitOpError() << "result " << i << " (" << getResult(i).getType()
                           << ") does not match output type (" << outputTypes[i]
                           << ") of module @" << module.getSymName();

  return success();
}

void InstanceOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getInstanceNameAttr());
  p << " ";
  p.printAttributeWithoutType(getModuleNameAttr());
  printInputPortList(p, getOperation(), getInputs(), getInputs().getTypes(),
                     getInputNames());
  p << " -> ";
  printOutputPortList(p, getOperation(), getOutputs().getTypes(),
                      getOutputNames());
  p.printOptionalAttrDict(getOperation()->getAttrs(), getAttributeNames());
}

ParseResult InstanceOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the instance name.
  StringAttr instanceName;
  if (parser.parseAttribute(instanceName, "instanceName", result.attributes))
    return failure();

  // Parse the module name.
  FlatSymbolRefAttr moduleName;
  if (parser.parseAttribute(moduleName, "moduleName", result.attributes))
    return failure();

  // Parse the input port list.
  auto loc = parser.getCurrentLocation();
  SmallVector<OpAsmParser::UnresolvedOperand> inputs;
  SmallVector<Type> types;
  ArrayAttr names;
  if (parseInputPortList(parser, inputs, types, names))
    return failure();
  if (parser.resolveOperands(inputs, types, loc, result.operands))
    return failure();
  result.addAttribute("inputNames", names);

  // Parse `->`.
  if (parser.parseArrow())
    return failure();

  // Parse the output port list.
  types.clear();
  if (parseOutputPortList(parser, types, names))
    return failure();
  result.addAttribute("outputNames", names);
  result.addTypes(types);

  // Parse the attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

void InstanceOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  SmallString<32> name;
  name += getInstanceName();
  name += '.';
  auto baseLen = name.size();

  for (auto [result, portName] :
       llvm::zip(getOutputs(), getOutputNames().getAsRange<StringAttr>())) {
    if (!portName || portName.empty())
      continue;
    name.resize(baseLen);
    name += portName.getValue();
    setNameFn(result, name);
  }
}

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

void VariableOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  if (getName() && !getName()->empty())
    setNameFn(getResult(), *getName());
}

llvm::SmallVector<MemorySlot> VariableOp::getPromotableSlots() {
  if (isa<SVModuleOp>(getOperation()->getParentOp()))
    return {};
  return {MemorySlot{getResult(), getType()}};
}

Value VariableOp::getDefaultValue(const MemorySlot &slot, OpBuilder &builder) {
  if (auto value = getInitial())
    return value;
  return builder.create<ConstantOp>(
      getLoc(),
      cast<moore::IntType>(cast<RefType>(slot.elemType).getNestedType()), 0);
}

void VariableOp::handleBlockArgument(const MemorySlot &slot,
                                     BlockArgument argument,
                                     OpBuilder &builder) {}

::std::optional<::mlir::PromotableAllocationOpInterface>
VariableOp::handlePromotionComplete(const MemorySlot &slot, Value defaultValue,
                                    OpBuilder &builder) {
  if (defaultValue.use_empty())
    defaultValue.getDefiningOp()->erase();
  this->erase();
  return std::nullopt;
}

LogicalResult VariableOp::canonicalize(VariableOp op,
                                       ::mlir::PatternRewriter &rewriter) {
  Value initial;
  for (auto *user : op->getUsers())
    if (isa<ContinuousAssignOp>(user) &&
        (user->getOperand(0) == op.getResult())) {
      // Don't canonicalize the multiple continuous assignment to the same
      // variable.
      if (initial)
        return failure();
      initial = user->getOperand(1);
    }

  if (initial) {
    rewriter.replaceOpWithNewOp<AssignedVarOp>(op, op.getType(),
                                               op.getNameAttr(), initial);
    return success();
  }

  return failure();
}

SmallVector<DestructurableMemorySlot> VariableOp::getDestructurableSlots() {
  if (isa<SVModuleOp>(getOperation()->getParentOp()))
    return {};

  auto refType = getType();
  auto destructurable = llvm::dyn_cast<DestructurableTypeInterface>(refType);
  if (!destructurable)
    return {};

  auto destructuredType = destructurable.getSubelementIndexMap();
  if (!destructuredType)
    return {};

  return {DestructurableMemorySlot{{getResult(), refType}, *destructuredType}};
}

DenseMap<Attribute, MemorySlot> VariableOp::destructure(
    const DestructurableMemorySlot &slot,
    const SmallPtrSetImpl<Attribute> &usedIndices, OpBuilder &builder,
    SmallVectorImpl<DestructurableAllocationOpInterface> &newAllocators) {
  assert(slot.ptr == getResult());
  builder.setInsertionPointAfter(*this);

  auto destructurableType = cast<DestructurableTypeInterface>(getType());
  DenseMap<Attribute, MemorySlot> slotMap;
  for (Attribute index : usedIndices) {
    auto elemType = cast<RefType>(destructurableType.getTypeAtIndex(index));
    assert(elemType && "used index must exist");
    auto varOp = builder.create<VariableOp>(getLoc(), elemType,
                                            cast<StringAttr>(index), Value());
    newAllocators.push_back(varOp);
    slotMap.try_emplace<MemorySlot>(index, {varOp.getResult(), elemType});
  }

  return slotMap;
}

std::optional<DestructurableAllocationOpInterface>
VariableOp::handleDestructuringComplete(const DestructurableMemorySlot &slot,
                                        OpBuilder &builder) {
  assert(slot.ptr == getResult());
  this->erase();
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// NetOp
//===----------------------------------------------------------------------===//

void NetOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  if (getName() && !getName()->empty())
    setNameFn(getResult(), *getName());
}

//===----------------------------------------------------------------------===//
// AssignedVarOp
//===----------------------------------------------------------------------===//

void AssignedVarOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  if (getName() && !getName()->empty())
    setNameFn(getResult(), *getName());
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getValueAttr());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  p << " : ";
  p.printStrippedAttrOrType(getType());
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the constant value without bit width.
  APInt value;
  auto valueLoc = parser.getCurrentLocation();

  if (parser.parseInteger(value) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon())
    return failure();

  // Parse the result type.
  IntType type;
  if (parser.parseCustomTypeWithFallback(type))
    return failure();

  // Extend or truncate the constant value to match the size of the type.
  if (type.getWidth() > value.getBitWidth()) {
    // sext is always safe here, even for unsigned values, because the
    // parseOptionalInteger method will return something with a zero in the
    // top bits if it is a positive number.
    value = value.sext(type.getWidth());
  } else if (type.getWidth() < value.getBitWidth()) {
    // The parser can return an unnecessarily wide result with leading
    // zeros. This isn't a problem, but truncating off bits is bad.
    unsigned neededBits =
        value.isNegative() ? value.getSignificantBits() : value.getActiveBits();
    if (type.getWidth() < neededBits)
      return parser.emitError(valueLoc,
                              "constant out of range for result type ")
             << type;
    value = value.trunc(type.getWidth());
  }

  // Build the attribute and op.
  auto attrType = IntegerType::get(parser.getContext(), type.getWidth());
  auto attrValue = IntegerAttr::get(attrType, value);

  result.addAttribute("value", attrValue);
  result.addTypes(type);
  return success();
}

LogicalResult ConstantOp::verify() {
  auto attrWidth = getValue().getBitWidth();
  auto typeWidth = getType().getWidth();
  if (attrWidth != typeWidth)
    return emitError("attribute width ")
           << attrWidth << " does not match return type's width " << typeWidth;
  return success();
}

void ConstantOp::build(OpBuilder &builder, OperationState &result, IntType type,
                       const APInt &value) {
  assert(type.getWidth() == value.getBitWidth() &&
         "APInt width must match type width");
  build(builder, result, type,
        builder.getIntegerAttr(builder.getIntegerType(type.getWidth()), value));
}

/// This builder allows construction of small signed integers like 0, 1, -1
/// matching a specified MLIR type. This shouldn't be used for general constant
/// folding because it only works with values that can be expressed in an
/// `int64_t`.
void ConstantOp::build(OpBuilder &builder, OperationState &result, IntType type,
                       int64_t value) {
  build(builder, result, type,
        APInt(type.getWidth(), (uint64_t)value, /*isSigned=*/true));
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// NamedConstantOp
//===----------------------------------------------------------------------===//

void NamedConstantOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), getName());
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  Domain domain = Domain::TwoValued;
  unsigned width = 0;
  for (auto operand : operands) {
    auto type = cast<IntType>(operand.getType());
    if (type.getDomain() == Domain::FourValued)
      domain = Domain::FourValued;
    width += type.getWidth();
  }
  results.push_back(IntType::get(context, width, domain));
  return success();
}

//===----------------------------------------------------------------------===//
// ConcatRefOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatRefOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  Domain domain = Domain::TwoValued;
  unsigned width = 0;
  for (auto operand : operands) {
    auto type = cast<IntType>(cast<RefType>(operand.getType()).getNestedType());
    if (type.getDomain() == Domain::FourValued)
      domain = Domain::FourValued;
    width += type.getWidth();
  }
  results.push_back(RefType::get(IntType::get(context, width, domain)));
  return success();
}

//===----------------------------------------------------------------------===//
// StructCreateOp
//===----------------------------------------------------------------------===//

LogicalResult StructCreateOp::verify() {
  /// checks if the types of the inputs are exactly equal to the types of the
  /// result struct fields
  return TypeSwitch<Type, LogicalResult>(getType().getNestedType())
      .Case<StructType, UnpackedStructType>([this](auto &type) {
        auto members = type.getMembers();
        auto inputs = getInput();
        if (inputs.size() != members.size())
          return failure();
        for (size_t i = 0; i < members.size(); i++) {
          auto memberType = cast<UnpackedType>(members[i].type);
          auto inputType = inputs[i].getType();
          if (inputType != memberType) {
            emitOpError("input types must match struct field types and orders");
            return failure();
          }
        }
        return success();
      })
      .Default([this](auto &) {
        emitOpError("Result type must be StructType or UnpackedStructType");
        return failure();
      });
}

OpFoldResult StructCreateOp::fold(FoldAdaptor adaptor) {
  auto inputs = adaptor.getInput();

  if (llvm::any_of(inputs, [](Attribute attr) { return !attr; }))
    return {};

  auto members = TypeSwitch<Type, ArrayRef<StructLikeMember>>(
                     cast<RefType>(getType()).getNestedType())
                     .Case<StructType, UnpackedStructType>(
                         [](auto &type) { return type.getMembers(); })
                     .Default([](auto) { return std::nullopt; });
  SmallVector<NamedAttribute> namedInputs;
  for (auto [input, member] : llvm::zip(inputs, members))
    namedInputs.push_back(NamedAttribute(member.name, input));

  return DictionaryAttr::get(getContext(), namedInputs);
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

LogicalResult StructExtractOp::verify() {
  /// checks if the type of the result match field type in this struct
  return TypeSwitch<Type, LogicalResult>(getInput().getType().getNestedType())
      .Case<StructType, UnpackedStructType>([this](auto &type) {
        auto members = type.getMembers();
        auto filedName = getFieldName();
        auto resultType = getType();
        for (const auto &member : members) {
          if (member.name == filedName && member.type == resultType) {
            return success();
          }
        }
        emitOpError("result type must match struct field type");
        return failure();
      })
      .Default([this](auto &) {
        emitOpError("input type must be StructType or UnpackedStructType");
        return failure();
      });
}

bool StructExtractOp::canRewire(const DestructurableMemorySlot &slot,
                                SmallPtrSetImpl<Attribute> &usedIndices,
                                SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                                const DataLayout &dataLayout) {
  if (slot.ptr == getInput()) {
    usedIndices.insert(getFieldNameAttr());
    return true;
  }
  return false;
}

DeletionKind StructExtractOp::rewire(const DestructurableMemorySlot &slot,
                                     DenseMap<Attribute, MemorySlot> &subslots,
                                     OpBuilder &builder,
                                     const DataLayout &dataLayout) {
  auto index = getFieldNameAttr();
  const auto &memorySlot = subslots.at(index);
  auto readOp = builder.create<moore::ReadOp>(
      getLoc(), cast<RefType>(memorySlot.elemType).getNestedType(),
      memorySlot.ptr);
  replaceAllUsesWith(readOp.getResult());
  getInputMutable().drop();
  erase();
  return DeletionKind::Keep;
}

OpFoldResult StructExtractOp::fold(FoldAdaptor adaptor) {
  if (auto constOperand = adaptor.getInput()) {
    auto operandAttr = llvm::cast<DictionaryAttr>(constOperand);
    for (const auto &ele : operandAttr)
      if (ele.getName() == getFieldNameAttr())
        return ele.getValue();
  }

  if (auto structInject = getInput().getDefiningOp<StructInjectOp>())
    return structInject.getFieldNameAttr() == getFieldNameAttr()
               ? structInject.getNewValue()
               : Value();
  if (auto structCreate = getInput().getDefiningOp<StructCreateOp>()) {
    auto ind = TypeSwitch<Type, std::optional<uint32_t>>(
                   getInput().getType().getNestedType())
                   .Case<StructType, UnpackedStructType>([this](auto &type) {
                     return type.getFieldIndex(getFieldNameAttr());
                   })
                   .Default([](auto &) { return std::nullopt; });
    return ind.has_value() ? structCreate->getOperand(ind.value()) : Value();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// StructExtractRefOp
//===----------------------------------------------------------------------===//

LogicalResult StructExtractRefOp::verify() {
  /// checks if the type of the result match field type in this struct
  return TypeSwitch<Type, LogicalResult>(getInput().getType().getNestedType())
      .Case<StructType, UnpackedStructType>([this](auto &type) {
        auto members = type.getMembers();
        auto filedName = getFieldName();
        auto resultType = getType().getNestedType();
        for (const auto &member : members) {
          if (member.name == filedName && member.type == resultType) {
            return success();
          }
        }
        emitOpError("result type must match struct field type");
        return failure();
      })
      .Default([this](auto &) {
        emitOpError("input type must be refrence of StructType or "
                    "UnpackedStructType");
        return failure();
      });
}

bool StructExtractRefOp::canRewire(
    const DestructurableMemorySlot &slot,
    SmallPtrSetImpl<Attribute> &usedIndices,
    SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
  if (slot.ptr != getInput())
    return false;
  auto index = getFieldNameAttr();
  if (!index || !slot.subelementTypes.contains(index))
    return false;
  usedIndices.insert(index);
  return true;
}

DeletionKind
StructExtractRefOp::rewire(const DestructurableMemorySlot &slot,
                           DenseMap<Attribute, MemorySlot> &subslots,
                           OpBuilder &builder, const DataLayout &dataLayout) {
  auto index = getFieldNameAttr();
  const MemorySlot &memorySlot = subslots.at(index);
  replaceAllUsesWith(memorySlot.ptr);
  getInputMutable().drop();
  erase();
  return DeletionKind::Keep;
}

//===----------------------------------------------------------------------===//
// StructInjectOp
//===----------------------------------------------------------------------===//

LogicalResult StructInjectOp::verify() {
  /// checks if the type of the new value match field type in this struct
  return TypeSwitch<Type, LogicalResult>(getInput().getType().getNestedType())
      .Case<StructType, UnpackedStructType>([this](auto &type) {
        auto members = type.getMembers();
        auto filedName = getFieldName();
        auto newValueType = getNewValue().getType();
        for (const auto &member : members) {
          if (member.name == filedName && member.type == newValueType) {
            return success();
          }
        }
        emitOpError("new value type must match struct field type");
        return failure();
      })
      .Default([this](auto &) {
        emitOpError("input type must be StructType or UnpackedStructType");
        return failure();
      });
}

void StructInjectOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printOperand(getInput());
  p << ", " << getFieldNameAttr() << ", ";
  p.printOperand(getNewValue());
  p << " : " << getInput().getType();
}

ParseResult StructInjectOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
  OpAsmParser::UnresolvedOperand operand, val;
  StringAttr fieldName;
  Type declType;

  if (parser.parseOperand(operand) || parser.parseComma() ||
      parser.parseAttribute(fieldName) || parser.parseComma() ||
      parser.parseOperand(val) || parser.parseColonType(declType))
    return failure();

  return TypeSwitch<Type, ParseResult>(cast<RefType>(declType).getNestedType())
      .Case<StructType, UnpackedStructType>([&parser, &result, &declType,
                                             &fieldName, &operand, &val,
                                             &inputOperandsLoc](auto &type) {
        auto members = type.getMembers();
        Type fieldType;
        for (const auto &member : members)
          if (member.name == fieldName)
            fieldType = member.type;
        if (!fieldType) {
          parser.emitError(parser.getNameLoc(),
                           "field name '" + fieldName.getValue() +
                               "' not found in struct type");
          return failure();
        }

        auto fieldNameAttr =
            StringAttr::get(parser.getContext(), Twine(fieldName));
        result.addAttribute("fieldName", fieldNameAttr);
        result.addTypes(declType);
        if (parser.resolveOperands({operand, val}, {declType, fieldType},
                                   inputOperandsLoc, result.operands))
          return failure();

        return success();
      })
      .Default([&parser, &inputOperandsLoc](auto &) {
        return parser.emitError(inputOperandsLoc,
                                "invalid kind of type specified");
      });
}

OpFoldResult StructInjectOp::fold(FoldAdaptor adaptor) {
  auto input = adaptor.getInput();
  auto newValue = adaptor.getNewValue();
  if (!input || !newValue)
    return {};
  SmallVector<NamedAttribute> array;
  llvm::copy(cast<DictionaryAttr>(input), std::back_inserter(array));
  for (auto &ele : array) {
    if (ele.getName() == getFieldName())
      ele.setValue(newValue);
  }
  return DictionaryAttr::get(getContext(), array);
}

LogicalResult StructInjectOp::canonicalize(StructInjectOp op,
                                           PatternRewriter &rewriter) {
  // Canonicalize multiple injects into a create op and eliminate overwrites.
  SmallPtrSet<Operation *, 4> injects;
  DenseMap<StringAttr, Value> fields;

  // Chase a chain of injects. Bail out if cycles are present.
  StructInjectOp inject = op;
  Value input;
  do {
    if (!injects.insert(inject).second)
      return failure();

    fields.try_emplace(inject.getFieldNameAttr(), inject.getNewValue());
    input = inject.getInput();
    inject = input.getDefiningOp<StructInjectOp>();
  } while (inject);
  assert(input && "missing input to inject chain");

  auto members = TypeSwitch<Type, ArrayRef<StructLikeMember>>(
                     cast<RefType>(op.getType()).getNestedType())
                     .Case<StructType, UnpackedStructType>(
                         [](auto &type) { return type.getMembers(); })
                     .Default([](auto) { return std::nullopt; });

  // If the inject chain sets all fields, canonicalize to create.
  if (fields.size() == members.size()) {
    SmallVector<Value> createFields;
    for (const auto &member : members) {
      auto it = fields.find(member.name);
      assert(it != fields.end() && "missing field");
      createFields.push_back(it->second);
    }
    op.getInputMutable();
    rewriter.replaceOpWithNewOp<StructCreateOp>(op, op.getType(), createFields);
    return success();
  }

  // Nothing to canonicalize, only the original inject in the chain.
  if (injects.size() == fields.size())
    return failure();

  // Eliminate overwrites. The hash map contains the last write to each field.
  for (const auto &member : members) {
    auto it = fields.find(member.name);
    if (it == fields.end())
      continue;
    input = rewriter.create<StructInjectOp>(op.getLoc(), op.getType(), input,
                                            member.name, it->second);
  }

  rewriter.replaceOp(op, input);
  return success();
}

//===----------------------------------------------------------------------===//
// UnionCreateOp
//===----------------------------------------------------------------------===//

LogicalResult UnionCreateOp::verify() {
  /// checks if the types of the input is exactly equal to the union field
  /// type
  return TypeSwitch<Type, LogicalResult>(getType())
      .Case<UnionType, UnpackedUnionType>([this](auto &type) {
        auto members = type.getMembers();
        auto resultType = getType();
        auto fieldName = getFieldName();
        for (const auto &member : members)
          if (member.name == fieldName && member.type == resultType)
            return success();
        emitOpError("input type must match the union field type");
        return failure();
      })
      .Default([this](auto &) {
        emitOpError("input type must be UnionType or UnpackedUnionType");
        return failure();
      });
}

//===----------------------------------------------------------------------===//
// UnionExtractOp
//===----------------------------------------------------------------------===//

LogicalResult UnionExtractOp::verify() {
  /// checks if the types of the input is exactly equal to the one of the
  /// types of the result union fields
  return TypeSwitch<Type, LogicalResult>(getInput().getType())
      .Case<UnionType, UnpackedUnionType>([this](auto &type) {
        auto members = type.getMembers();
        auto fieldName = getFieldName();
        auto resultType = getType();
        for (const auto &member : members)
          if (member.name == fieldName && member.type == resultType)
            return success();
        emitOpError("result type must match the union field type");
        return failure();
      })
      .Default([this](auto &) {
        emitOpError("input type must be UnionType or UnpackedUnionType");
        return failure();
      });
}

//===----------------------------------------------------------------------===//
// UnionExtractOp
//===----------------------------------------------------------------------===//

LogicalResult UnionExtractRefOp::verify() {
  /// checks if the types of the result is exactly equal to the type of the
  /// refe union field
  return TypeSwitch<Type, LogicalResult>(getInput().getType().getNestedType())
      .Case<UnionType, UnpackedUnionType>([this](auto &type) {
        auto members = type.getMembers();
        auto fieldName = getFieldName();
        auto resultType = getType().getNestedType();
        for (const auto &member : members)
          if (member.name == fieldName && member.type == resultType)
            return success();
        emitOpError("result type must match the union field type");
        return failure();
      })
      .Default([this](auto &) {
        emitOpError("input type must be UnionType or UnpackedUnionType");
        return failure();
      });
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult YieldOp::verify() {
  // Check that YieldOp's parent operation is ConditionalOp.
  auto cond = dyn_cast<ConditionalOp>(*(*this).getParentOp());
  if (!cond) {
    emitOpError("must have a conditional parent");
    return failure();
  }

  // Check that the operand matches the parent operation's result.
  auto condType = cond.getType();
  auto yieldType = getOperand().getType();
  if (condType != yieldType) {
    emitOpError("yield type must match conditional. Expected ")
        << condType << ", but got " << yieldType << ".";
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConversionOp
//===----------------------------------------------------------------------===//

OpFoldResult ConversionOp::fold(FoldAdaptor adaptor) {
  // Fold away no-op casts.
  if (getInput().getType() == getResult().getType())
    return getInput();
  return {};
}

//===----------------------------------------------------------------------===//
// BoolCastOp
//===----------------------------------------------------------------------===//

OpFoldResult BoolCastOp::fold(FoldAdaptor adaptor) {
  // Fold away no-op casts.
  if (getInput().getType() == getResult().getType())
    return getInput();
  return {};
}

//===----------------------------------------------------------------------===//
// BlockingAssignOp
//===----------------------------------------------------------------------===//

bool BlockingAssignOp::loadsFrom(const MemorySlot &slot) { return false; }

bool BlockingAssignOp::storesTo(const MemorySlot &slot) {
  return getDst() == slot.ptr;
}

Value BlockingAssignOp::getStored(const MemorySlot &slot, OpBuilder &builder,
                                  Value reachingDef,
                                  const DataLayout &dataLayout) {
  return getSrc();
}

bool BlockingAssignOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {

  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getDst() == slot.ptr &&
         getSrc() != slot.ptr &&
         getSrc().getType() == cast<RefType>(slot.elemType).getNestedType();
}

DeletionKind BlockingAssignOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, Value reachingDefinition,
    const DataLayout &dataLayout) {
  return DeletionKind::Delete;
}

//===----------------------------------------------------------------------===//
// ReadOp
//===----------------------------------------------------------------------===//

bool ReadOp::loadsFrom(const MemorySlot &slot) {
  return getOperand() == slot.ptr;
}

bool ReadOp::storesTo(const MemorySlot &slot) { return false; }

Value ReadOp::getStored(const MemorySlot &slot, OpBuilder &builder,
                        Value reachingDef, const DataLayout &dataLayout) {
  llvm_unreachable("getStored should not be called on ReadOp");
}

bool ReadOp::canUsesBeRemoved(const MemorySlot &slot,
                              const SmallPtrSetImpl<OpOperand *> &blockingUses,
                              SmallVectorImpl<OpOperand *> &newBlockingUses,
                              const DataLayout &dataLayout) {

  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getOperand() == slot.ptr &&
         getResult().getType() == cast<RefType>(slot.elemType).getNestedType();
}

DeletionKind
ReadOp::removeBlockingUses(const MemorySlot &slot,
                           const SmallPtrSetImpl<OpOperand *> &blockingUses,
                           OpBuilder &builder, Value reachingDefinition,
                           const DataLayout &dataLayout) {
  getResult().replaceAllUsesWith(reachingDefinition);
  return DeletionKind::Delete;
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Moore/Moore.cpp.inc"
#include "circt/Dialect/Moore/MooreEnums.cpp.inc"
