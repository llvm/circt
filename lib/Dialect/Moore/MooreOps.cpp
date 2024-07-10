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
  setNameFn(getResult(), getName());
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
    rewriter.replaceOpWithNewOp<AssignedVarOp>(op, op.getType(), op.getName(),
                                               initial);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// NetOp
//===----------------------------------------------------------------------===//

void NetOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), getName());
}

//===----------------------------------------------------------------------===//
// AssignedVarOp
//===----------------------------------------------------------------------===//

void AssignedVarOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), getName());
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
