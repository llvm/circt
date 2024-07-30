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
  if (getInitial())
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
  if (getInitial())
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
  assert(!getInitial());
  builder.setInsertionPointAfter(*this);

  auto destructurableType = cast<DestructurableTypeInterface>(getType());
  DenseMap<Attribute, MemorySlot> slotMap;
  for (Attribute index : usedIndices) {
    auto elemType = cast<RefType>(destructurableType.getTypeAtIndex(index));
    assert(elemType && "used index must exist");
    StringAttr varName;
    if (auto name = getName(); name && !name->empty())
      varName = StringAttr::get(
          getContext(), (*name) + "." + cast<StringAttr>(index).getValue());
    auto varOp =
        builder.create<VariableOp>(getLoc(), elemType, varName, Value());
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

static std::optional<uint32_t> getStructFieldIndex(Type type, StringAttr name) {
  if (auto structType = dyn_cast<StructType>(type))
    return structType.getFieldIndex(name);
  if (auto structType = dyn_cast<UnpackedStructType>(type))
    return structType.getFieldIndex(name);
  assert(0 && "expected StructType or UnpackedStructType");
  return {};
}

static ArrayRef<StructLikeMember> getStructMembers(Type type) {
  if (auto structType = dyn_cast<StructType>(type))
    return structType.getMembers();
  if (auto structType = dyn_cast<UnpackedStructType>(type))
    return structType.getMembers();
  assert(0 && "expected StructType or UnpackedStructType");
  return {};
}

static UnpackedType getStructFieldType(Type type, StringAttr name) {
  if (auto index = getStructFieldIndex(type, name))
    return getStructMembers(type)[*index].type;
  return {};
}

LogicalResult StructCreateOp::verify() {
  auto members = getStructMembers(getType());

  // Check that the number of operands matches the number of struct fields.
  if (getFields().size() != members.size())
    return emitOpError() << "has " << getFields().size()
                         << " operands, but result type requires "
                         << members.size();

  // Check that the operand types match the struct field types.
  for (auto [index, pair] : llvm::enumerate(llvm::zip(getFields(), members))) {
    auto [value, member] = pair;
    if (value.getType() != member.type)
      return emitOpError() << "operand #" << index << " has type "
                           << value.getType() << ", but struct field "
                           << member.name << " requires " << member.type;
  }
  return success();
}

OpFoldResult StructCreateOp::fold(FoldAdaptor adaptor) {
  SmallVector<NamedAttribute> fields;
  for (auto [member, field] :
       llvm::zip(getStructMembers(getType()), adaptor.getFields())) {
    if (!field)
      return {};
    fields.push_back(NamedAttribute(member.name, field));
  }
  return DictionaryAttr::get(getContext(), fields);
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

LogicalResult StructExtractOp::verify() {
  auto type = getStructFieldType(getInput().getType(), getFieldNameAttr());
  if (!type)
    return emitOpError() << "extracts field " << getFieldNameAttr()
                         << " which does not exist in " << getInput().getType();
  if (type != getType())
    return emitOpError() << "result type " << getType()
                         << " must match struct field type " << type;
  return success();
}

OpFoldResult StructExtractOp::fold(FoldAdaptor adaptor) {
  // Extract on a constant struct input.
  if (auto fields = dyn_cast_or_null<DictionaryAttr>(adaptor.getInput()))
    if (auto value = fields.get(getFieldNameAttr()))
      return value;

  // extract(inject(s, "field", v), "field") -> v
  if (auto inject = getInput().getDefiningOp<StructInjectOp>()) {
    if (inject.getFieldNameAttr() == getFieldNameAttr())
      return inject.getNewValue();
    return {};
  }

  // extract(create({"field": v, ...}), "field") -> v
  if (auto create = getInput().getDefiningOp<StructCreateOp>()) {
    if (auto index = getStructFieldIndex(create.getType(), getFieldNameAttr()))
      return create.getFields()[*index];
    return {};
  }

  return {};
}

//===----------------------------------------------------------------------===//
// StructExtractRefOp
//===----------------------------------------------------------------------===//

LogicalResult StructExtractRefOp::verify() {
  auto type = getStructFieldType(
      cast<RefType>(getInput().getType()).getNestedType(), getFieldNameAttr());
  if (!type)
    return emitOpError() << "extracts field " << getFieldNameAttr()
                         << " which does not exist in " << getInput().getType();
  if (type != getType().getNestedType())
    return emitOpError() << "result ref of type " << getType().getNestedType()
                         << " must match struct field type " << type;
  return success();
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
  auto type = getStructFieldType(getInput().getType(), getFieldNameAttr());
  if (!type)
    return emitOpError() << "injects field " << getFieldNameAttr()
                         << " which does not exist in " << getInput().getType();
  if (type != getNewValue().getType())
    return emitOpError() << "injected value " << getNewValue().getType()
                         << " must match struct field type " << type;
  return success();
}

OpFoldResult StructInjectOp::fold(FoldAdaptor adaptor) {
  auto input = adaptor.getInput();
  auto newValue = adaptor.getNewValue();
  if (!input || !newValue)
    return {};
  NamedAttrList fields(cast<DictionaryAttr>(input));
  fields.set(getFieldNameAttr(), newValue);
  return fields.getDictionary(getContext());
}

LogicalResult StructInjectOp::canonicalize(StructInjectOp op,
                                           PatternRewriter &rewriter) {
  auto members = getStructMembers(op.getType());

  // Chase a chain of `struct_inject` ops, with an optional final
  // `struct_create`, and take note of the values assigned to each field.
  SmallPtrSet<Operation *, 4> injectOps;
  DenseMap<StringAttr, Value> fieldValues;
  Value input = op;
  while (auto injectOp = input.getDefiningOp<StructInjectOp>()) {
    if (!injectOps.insert(injectOp).second)
      return failure();
    fieldValues.insert({injectOp.getFieldNameAttr(), injectOp.getNewValue()});
    input = injectOp.getInput();
  }
  if (auto createOp = input.getDefiningOp<StructCreateOp>())
    for (auto [value, member] : llvm::zip(createOp.getFields(), members))
      fieldValues.insert({member.name, value});

  // If the inject chain sets all fields, canonicalize to a `struct_create`.
  if (fieldValues.size() == members.size()) {
    SmallVector<Value> values;
    values.reserve(fieldValues.size());
    for (auto member : members)
      values.push_back(fieldValues.lookup(member.name));
    rewriter.replaceOpWithNewOp<StructCreateOp>(op, op.getType(), values);
    return success();
  }

  // If each inject op in the chain assigned to a unique field, there is nothing
  // to canonicalize.
  if (injectOps.size() == fieldValues.size())
    return failure();

  // Otherwise we can eliminate overwrites by creating new injects. The hash map
  // of field values contains the last assigned value for each field.
  for (auto member : members)
    if (auto value = fieldValues.lookup(member.name))
      input = rewriter.create<StructInjectOp>(op.getLoc(), op.getType(), input,
                                              member.name, value);
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
