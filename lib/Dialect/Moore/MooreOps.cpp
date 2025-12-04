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
#include "circt/Dialect/Moore/MooreAttributes.h"
#include "circt/Support/CustomDirectiveImpl.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>

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

LogicalResult VariableOp::canonicalize(VariableOp op,
                                       PatternRewriter &rewriter) {
  // If the variable is embedded in an SSACFG region, move the initial value
  // into an assignment immediately after the variable op. This allows the
  // mem2reg pass which cannot handle variables with initial values.
  auto initial = op.getInitial();
  if (initial && mlir::mayHaveSSADominance(*op->getParentRegion())) {
    rewriter.modifyOpInPlace(op, [&] { op.getInitialMutable().clear(); });
    rewriter.setInsertionPointAfter(op);
    BlockingAssignOp::create(rewriter, initial.getLoc(), op, initial);
    return success();
  }

  // Check if the variable has one unique continuous assignment to it, all other
  // uses are reads, and that all uses are in the same block as the variable
  // itself.
  auto *block = op->getBlock();
  ContinuousAssignOp uniqueAssignOp;
  for (auto *user : op->getUsers()) {
    // Ensure that all users of the variable are in the same block.
    if (user->getBlock() != block)
      return failure();

    // Ensure there is at most one unique continuous assignment to the variable.
    if (auto assignOp = dyn_cast<ContinuousAssignOp>(user)) {
      if (uniqueAssignOp)
        return failure();
      uniqueAssignOp = assignOp;
      continue;
    }

    // Ensure all other users are reads.
    if (!isa<ReadOp>(user))
      return failure();
  }
  if (!uniqueAssignOp)
    return failure();

  // If the original variable had a name, create an `AssignedVariableOp` as a
  // replacement. Otherwise substitute the assigned value directly.
  Value assignedValue = uniqueAssignOp.getSrc();
  if (auto name = op.getNameAttr(); name && !name.empty())
    assignedValue = AssignedVariableOp::create(rewriter, op.getLoc(), name,
                                               uniqueAssignOp.getSrc());

  // Remove the assign op and replace all reads with the new assigned var op.
  rewriter.eraseOp(uniqueAssignOp);
  for (auto *user : llvm::make_early_inc_range(op->getUsers())) {
    auto readOp = cast<ReadOp>(user);
    rewriter.replaceOp(readOp, assignedValue);
  }

  // Remove the original variable.
  rewriter.eraseOp(op);
  return success();
}

SmallVector<MemorySlot> VariableOp::getPromotableSlots() {
  // We cannot promote variables with an initial value, since that value may not
  // dominate the location where the default value needs to be constructed.
  if (mlir::mayBeGraphRegion(*getOperation()->getParentRegion()) ||
      getInitial())
    return {};

  // Ensure that `getDefaultValue` can conjure up a default value for the
  // variable's type.
  if (!isa<PackedType>(getType().getNestedType()))
    return {};

  return {MemorySlot{getResult(), getType().getNestedType()}};
}

Value VariableOp::getDefaultValue(const MemorySlot &slot, OpBuilder &builder) {
  auto packedType = dyn_cast<PackedType>(slot.elemType);
  if (!packedType)
    return {};
  auto bitWidth = packedType.getBitSize();
  if (!bitWidth)
    return {};
  auto fvint = packedType.getDomain() == Domain::FourValued
                   ? FVInt::getAllX(*bitWidth)
                   : FVInt::getZero(*bitWidth);
  Value value = ConstantOp::create(
      builder, getLoc(),
      IntType::get(getContext(), *bitWidth, packedType.getDomain()), fvint);
  if (value.getType() != packedType)
    ConversionOp::create(builder, getLoc(), packedType, value);
  return value;
}

void VariableOp::handleBlockArgument(const MemorySlot &slot,
                                     BlockArgument argument,
                                     OpBuilder &builder) {}

std::optional<mlir::PromotableAllocationOpInterface>
VariableOp::handlePromotionComplete(const MemorySlot &slot, Value defaultValue,
                                    OpBuilder &builder) {
  if (defaultValue && defaultValue.use_empty())
    defaultValue.getDefiningOp()->erase();
  this->erase();
  return {};
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
        VariableOp::create(builder, getLoc(), elemType, varName, Value());
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

LogicalResult NetOp::canonicalize(NetOp op, PatternRewriter &rewriter) {
  bool modified = false;

  // Check if the net has one unique continuous assignment to it, and
  // additionally if all other users are reads.
  auto *block = op->getBlock();
  ContinuousAssignOp uniqueAssignOp;
  bool allUsesAreReads = true;
  for (auto *user : op->getUsers()) {
    // Ensure that all users of the net are in the same block.
    if (user->getBlock() != block)
      return failure();

    // Ensure there is at most one unique continuous assignment to the net.
    if (auto assignOp = dyn_cast<ContinuousAssignOp>(user)) {
      if (uniqueAssignOp)
        return failure();
      uniqueAssignOp = assignOp;
      continue;
    }

    // Ensure all other users are reads.
    if (!isa<ReadOp>(user))
      allUsesAreReads = false;
  }

  // If there was one unique assignment, and the `NetOp` does not yet have an
  // assigned value set, fold the assignment into the net.
  if (uniqueAssignOp && !op.getAssignment()) {
    rewriter.modifyOpInPlace(
        op, [&] { op.getAssignmentMutable().assign(uniqueAssignOp.getSrc()); });
    rewriter.eraseOp(uniqueAssignOp);
    modified = true;
    uniqueAssignOp = {};
  }

  // If all users of the net op are reads, and any potential unique assignment
  // has been folded into the net op itself, directly replace the reads with the
  // net's assigned value.
  if (!uniqueAssignOp && allUsesAreReads && op.getAssignment()) {
    // If the original net had a name, create an `AssignedVariableOp` as a
    // replacement. Otherwise substitute the assigned value directly.
    auto assignedValue = op.getAssignment();
    if (auto name = op.getNameAttr(); name && !name.empty())
      assignedValue = AssignedVariableOp::create(rewriter, op.getLoc(), name,
                                                 assignedValue);

    // Replace all reads with the new assigned var op and remove the original
    // net op.
    for (auto *user : llvm::make_early_inc_range(op->getUsers())) {
      auto readOp = cast<ReadOp>(user);
      rewriter.replaceOp(readOp, assignedValue);
    }
    rewriter.eraseOp(op);
    modified = true;
  }

  return success(modified);
}

//===----------------------------------------------------------------------===//
// AssignedVariableOp
//===----------------------------------------------------------------------===//

void AssignedVariableOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  if (getName() && !getName()->empty())
    setNameFn(getResult(), *getName());
}

LogicalResult AssignedVariableOp::canonicalize(AssignedVariableOp op,
                                               PatternRewriter &rewriter) {
  // Eliminate chained variables with the same name.
  // var(name, var(name, x)) -> var(name, x)
  if (auto otherOp = op.getInput().getDefiningOp<AssignedVariableOp>()) {
    if (otherOp.getNameAttr() == op.getNameAttr()) {
      rewriter.replaceOp(op, otherOp);
      return success();
    }
  }

  // Eliminate variables that alias an input port of the same name.
  if (auto blockArg = dyn_cast<BlockArgument>(op.getInput())) {
    if (auto moduleOp =
            dyn_cast<SVModuleOp>(blockArg.getOwner()->getParentOp())) {
      auto moduleType = moduleOp.getModuleType();
      auto portName = moduleType.getInputNameAttr(blockArg.getArgNumber());
      if (portName == op.getNameAttr()) {
        rewriter.replaceOp(op, blockArg);
        return success();
      }
    }
  }

  // Eliminate variables that feed an output port of the same name.
  for (auto &use : op->getUses()) {
    auto *useOwner = use.getOwner();
    if (auto outputOp = dyn_cast<OutputOp>(useOwner)) {
      if (auto moduleOp = dyn_cast<SVModuleOp>(outputOp->getParentOp())) {
        auto moduleType = moduleOp.getModuleType();
        auto portName = moduleType.getOutputNameAttr(use.getOperandNumber());
        if (portName == op.getNameAttr()) {
          rewriter.replaceOp(op, op.getInput());
          return success();
        }
      } else
        break;
    }
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// GlobalVariableOp
//===----------------------------------------------------------------------===//

LogicalResult GlobalVariableOp::verifyRegions() {
  if (auto *block = getInitBlock()) {
    auto &terminator = block->back();
    if (!isa<YieldOp>(terminator))
      return emitOpError() << "must have a 'moore.yield' terminator";
  }
  return success();
}

Block *GlobalVariableOp::getInitBlock() {
  if (getInitRegion().empty())
    return nullptr;
  return &getInitRegion().front();
}

//===----------------------------------------------------------------------===//
// GetGlobalVariableOp
//===----------------------------------------------------------------------===//

LogicalResult
GetGlobalVariableOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Resolve the target symbol.
  auto *symbol =
      symbolTable.lookupNearestSymbolFrom(*this, getGlobalNameAttr());
  if (!symbol)
    return emitOpError() << "references unknown symbol " << getGlobalNameAttr();

  // Check that the symbol is a global variable.
  auto var = dyn_cast<GlobalVariableOp>(symbol);
  if (!var)
    return emitOpError() << "must reference a 'moore.global_variable', but "
                         << getGlobalNameAttr() << " is a '"
                         << symbol->getName() << "'";

  // Check that the types match.
  auto expType = var.getType();
  auto actType = getType().getNestedType();
  if (expType != actType)
    return emitOpError() << "returns a " << actType << " reference, but "
                         << getGlobalNameAttr() << " is of type " << expType;

  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  printFVInt(p, getValue());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  p << " : ";
  p.printStrippedAttrOrType(getType());
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the constant value.
  FVInt value;
  auto valueLoc = parser.getCurrentLocation();
  if (parseFVInt(parser, value))
    return failure();

  // Parse any optional attributes and the `:`.
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon())
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
      return parser.emitError(valueLoc)
             << "value requires " << neededBits
             << " bits, but result type only has " << type.getWidth();
    value = value.trunc(type.getWidth());
  }

  // If the constant contains any X or Z bits, the result type must be
  // four-valued.
  if (value.hasUnknown() && type.getDomain() != Domain::FourValued)
    return parser.emitError(valueLoc)
           << "value contains X or Z bits, but result type " << type
           << " only allows two-valued bits";

  // Build the attribute and op.
  auto attrValue = FVIntegerAttr::get(parser.getContext(), value);
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
                       const FVInt &value) {
  assert(type.getWidth() == value.getBitWidth() &&
         "FVInt width must match type width");
  build(builder, result, type, FVIntegerAttr::get(builder.getContext(), value));
}

void ConstantOp::build(OpBuilder &builder, OperationState &result, IntType type,
                       const APInt &value) {
  assert(type.getWidth() == value.getBitWidth() &&
         "APInt width must match type width");
  build(builder, result, type, FVInt(value));
}

/// This builder allows construction of small signed integers like 0, 1, -1
/// matching a specified MLIR type. This shouldn't be used for general constant
/// folding because it only works with values that can be expressed in an
/// `int64_t`.
void ConstantOp::build(OpBuilder &builder, OperationState &result, IntType type,
                       int64_t value, bool isSigned) {
  build(builder, result, type,
        APInt(type.getWidth(), (uint64_t)value, isSigned));
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// ConstantTimeOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantTimeOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// ConstantRealOp
//===----------------------------------------------------------------------===//

LogicalResult ConstantRealOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  ConstantRealOp::Adaptor adaptor(operands, attrs, properties);
  results.push_back(RealType::get(
      context, static_cast<RealWidth>(
                   adaptor.getValueAttr().getType().getIntOrFloatBitWidth())));
  return success();
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
  for (Value operand : operands) {
    UnpackedType nestedType = cast<RefType>(operand.getType()).getNestedType();
    PackedType packedType = dyn_cast<PackedType>(nestedType);

    if (!packedType) {
      return failure();
    }

    if (packedType.getDomain() == Domain::FourValued)
      domain = Domain::FourValued;

    // getBitSize() for PackedType returns an optional, so we must check it.
    std::optional<int> bitSize = packedType.getBitSize();
    if (!bitSize) {
      return failure();
    }
    width += *bitSize;
  }
  results.push_back(RefType::get(IntType::get(context, width, domain)));
  return success();
}

//===----------------------------------------------------------------------===//
// ArrayCreateOp
//===----------------------------------------------------------------------===//

static std::pair<unsigned, UnpackedType> getArrayElements(Type type) {
  if (auto arrayType = dyn_cast<ArrayType>(type))
    return {arrayType.getSize(), arrayType.getElementType()};
  if (auto arrayType = dyn_cast<UnpackedArrayType>(type))
    return {arrayType.getSize(), arrayType.getElementType()};
  assert(0 && "expected ArrayType or UnpackedArrayType");
  return {};
}

LogicalResult ArrayCreateOp::verify() {
  auto [size, elementType] = getArrayElements(getType());

  // Check that the number of operands matches the array size.
  if (getElements().size() != size)
    return emitOpError() << "has " << getElements().size()
                         << " operands, but result type requires " << size;

  // Check that the operand types match the array element type. We only need to
  // check one of the operands, since the `SameTypeOperands` trait ensures all
  // operands have the same type.
  if (size > 0) {
    auto value = getElements()[0];
    if (value.getType() != elementType)
      return emitOpError() << "operands have type " << value.getType()
                           << ", but array requires " << elementType;
  }
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
      input = StructInjectOp::create(rewriter, op.getLoc(), op.getType(), input,
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
  Type expType;
  auto *parentOp = getOperation()->getParentOp();
  if (auto cond = dyn_cast<ConditionalOp>(parentOp)) {
    expType = cond.getType();
  } else if (auto varOp = dyn_cast<GlobalVariableOp>(parentOp)) {
    expType = varOp.getType();
  } else {
    llvm_unreachable("all in ParentOneOf handled");
  }

  auto actType = getOperand().getType();
  if (expType != actType) {
    return emitOpError() << "yields " << actType << ", but parent expects "
                         << expType;
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

  // Convert domains of constant integer inputs.
  auto intInput = dyn_cast_or_null<FVIntegerAttr>(adaptor.getInput());
  auto fromIntType = dyn_cast<IntType>(getInput().getType());
  auto toIntType = dyn_cast<IntType>(getResult().getType());
  if (intInput && fromIntType && toIntType &&
      fromIntType.getWidth() == toIntType.getWidth()) {
    // If we are going *to* a four-valued type, simply pass through the
    // constant.
    if (toIntType.getDomain() == Domain::FourValued)
      return intInput;

    // Otherwise map all unknown bits to zero (the default in SystemVerilog) and
    // return a new constant.
    return FVIntegerAttr::get(getContext(), intInput.getValue().toAPInt(false));
  }

  return {};
}

//===----------------------------------------------------------------------===//
// LogicToIntOp
//===----------------------------------------------------------------------===//

OpFoldResult LogicToIntOp::fold(FoldAdaptor adaptor) {
  // logic_to_int(int_to_logic(x)) -> x
  if (auto reverseOp = getInput().getDefiningOp<IntToLogicOp>())
    return reverseOp.getInput();

  // Map all unknown bits to zero (the default in SystemVerilog) and return a
  // new constant.
  if (auto intInput = dyn_cast_or_null<FVIntegerAttr>(adaptor.getInput()))
    return FVIntegerAttr::get(getContext(), intInput.getValue().toAPInt(false));

  return {};
}

//===----------------------------------------------------------------------===//
// IntToLogicOp
//===----------------------------------------------------------------------===//

OpFoldResult IntToLogicOp::fold(FoldAdaptor adaptor) {
  // Cannot fold int_to_logic(logic_to_int(x)) -> x since that would lose
  // information.

  // Simply pass through constants.
  if (auto intInput = dyn_cast_or_null<FVIntegerAttr>(adaptor.getInput()))
    return intInput;

  return {};
}

//===----------------------------------------------------------------------===//
// TimeToLogicOp
//===----------------------------------------------------------------------===//

OpFoldResult TimeToLogicOp::fold(FoldAdaptor adaptor) {
  // time_to_logic(logic_to_time(x)) -> x
  if (auto reverseOp = getInput().getDefiningOp<LogicToTimeOp>())
    return reverseOp.getInput();

  // Convert constants.
  if (auto attr = dyn_cast_or_null<IntegerAttr>(adaptor.getInput()))
    return FVIntegerAttr::get(getContext(), attr.getValue());

  return {};
}

//===----------------------------------------------------------------------===//
// LogicToTimeOp
//===----------------------------------------------------------------------===//

OpFoldResult LogicToTimeOp::fold(FoldAdaptor adaptor) {
  // logic_to_time(time_to_logic(x)) -> x
  if (auto reverseOp = getInput().getDefiningOp<TimeToLogicOp>())
    return reverseOp.getInput();

  // Convert constants.
  if (auto attr = dyn_cast_or_null<FVIntegerAttr>(adaptor.getInput()))
    return IntegerAttr::get(getContext(), APSInt(attr.getValue().toAPInt(false),
                                                 /*isUnsigned=*/true));

  return {};
}

//===----------------------------------------------------------------------===//
// TruncOp
//===----------------------------------------------------------------------===//

OpFoldResult TruncOp::fold(FoldAdaptor adaptor) {
  // Truncate constants.
  if (auto intAttr = dyn_cast_or_null<FVIntegerAttr>(adaptor.getInput())) {
    auto width = getType().getWidth();
    return FVIntegerAttr::get(getContext(), intAttr.getValue().trunc(width));
  }

  return {};
}

//===----------------------------------------------------------------------===//
// ZExtOp
//===----------------------------------------------------------------------===//

OpFoldResult ZExtOp::fold(FoldAdaptor adaptor) {
  // Zero-extend constants.
  if (auto intAttr = dyn_cast_or_null<FVIntegerAttr>(adaptor.getInput())) {
    auto width = getType().getWidth();
    return FVIntegerAttr::get(getContext(), intAttr.getValue().zext(width));
  }

  return {};
}

//===----------------------------------------------------------------------===//
// SExtOp
//===----------------------------------------------------------------------===//

OpFoldResult SExtOp::fold(FoldAdaptor adaptor) {
  // Sign-extend constants.
  if (auto intAttr = dyn_cast_or_null<FVIntegerAttr>(adaptor.getInput())) {
    auto width = getType().getWidth();
    return FVIntegerAttr::get(getContext(), intAttr.getValue().sext(width));
  }

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
         getSrc() != slot.ptr && getSrc().getType() == slot.elemType;
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
  return getInput() == slot.ptr;
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
         getResult().getType() == slot.elemType;
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
// PowSOp
//===----------------------------------------------------------------------===//

static OpFoldResult powCommonFolding(MLIRContext *ctxt, Attribute lhs,
                                     Attribute rhs) {
  auto lhsValue = dyn_cast_or_null<FVIntegerAttr>(lhs);
  if (lhsValue && lhsValue.getValue() == 1)
    return lhs;

  auto rhsValue = dyn_cast_or_null<FVIntegerAttr>(rhs);
  if (rhsValue && rhsValue.getValue().isZero())
    return FVIntegerAttr::get(ctxt,
                              FVInt(rhsValue.getValue().getBitWidth(), 1));

  return {};
}

OpFoldResult PowSOp::fold(FoldAdaptor adaptor) {
  return powCommonFolding(getContext(), adaptor.getLhs(), adaptor.getRhs());
}

LogicalResult PowSOp::canonicalize(PowSOp op, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  auto intType = cast<IntType>(op.getRhs().getType());
  if (auto baseOp = op.getLhs().getDefiningOp<ConstantOp>()) {
    if (baseOp.getValue() == 2) {
      Value constOne = ConstantOp::create(rewriter, loc, intType, 1);
      Value constZero = ConstantOp::create(rewriter, loc, intType, 0);
      Value shift = ShlOp::create(rewriter, loc, constOne, op.getRhs());
      Value isNegative = SltOp::create(rewriter, loc, op.getRhs(), constZero);
      auto condOp = rewriter.replaceOpWithNewOp<ConditionalOp>(
          op, op.getLhs().getType(), isNegative);
      Block *thenBlock = rewriter.createBlock(&condOp.getTrueRegion());
      rewriter.setInsertionPointToStart(thenBlock);
      YieldOp::create(rewriter, loc, constZero);
      Block *elseBlock = rewriter.createBlock(&condOp.getFalseRegion());
      rewriter.setInsertionPointToStart(elseBlock);
      YieldOp::create(rewriter, loc, shift);
      return success();
    }
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// PowUOp
//===----------------------------------------------------------------------===//

OpFoldResult PowUOp::fold(FoldAdaptor adaptor) {
  return powCommonFolding(getContext(), adaptor.getLhs(), adaptor.getRhs());
}

LogicalResult PowUOp::canonicalize(PowUOp op, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  auto intType = cast<IntType>(op.getRhs().getType());
  if (auto baseOp = op.getLhs().getDefiningOp<ConstantOp>()) {
    if (baseOp.getValue() == 2) {
      Value constOne = ConstantOp::create(rewriter, loc, intType, 1);
      rewriter.replaceOpWithNewOp<ShlOp>(op, constOne, op.getRhs());
      return success();
    }
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  if (auto intAttr = dyn_cast_or_null<FVIntegerAttr>(adaptor.getRhs()))
    if (intAttr.getValue().isZero())
      return getLhs();

  return {};
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<FVIntegerAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_or_null<FVIntegerAttr>(adaptor.getRhs());
  if (lhs && rhs)
    return FVIntegerAttr::get(getContext(), lhs.getValue() * rhs.getValue());
  return {};
}

//===----------------------------------------------------------------------===//
// DivUOp
//===----------------------------------------------------------------------===//

OpFoldResult DivUOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<FVIntegerAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_or_null<FVIntegerAttr>(adaptor.getRhs());
  if (lhs && rhs)
    return FVIntegerAttr::get(getContext(),
                              lhs.getValue().udiv(rhs.getValue()));
  return {};
}

//===----------------------------------------------------------------------===//
// DivSOp
//===----------------------------------------------------------------------===//

OpFoldResult DivSOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<FVIntegerAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_or_null<FVIntegerAttr>(adaptor.getRhs());
  if (lhs && rhs)
    return FVIntegerAttr::get(getContext(),
                              lhs.getValue().sdiv(rhs.getValue()));
  return {};
}

//===----------------------------------------------------------------------===//
// Classes
//===----------------------------------------------------------------------===//

LogicalResult ClassDeclOp::verify() {
  mlir::Region &body = getBody();
  if (body.empty())
    return mlir::success();

  auto &block = body.front();
  for (mlir::Operation &op : block) {

    // allow only property and method decls and terminator
    if (llvm::isa<circt::moore::ClassPropertyDeclOp,
                  circt::moore::ClassMethodDeclOp>(&op))
      continue;

    return emitOpError()
           << "body may only contain 'moore.class.propertydecl' operations";
  }
  return mlir::success();
}

LogicalResult ClassNewOp::verify() {
  // The result is constrained to ClassHandleType in ODS, so this cast should be
  // safe.
  auto handleTy = cast<ClassHandleType>(getResult().getType());
  mlir::SymbolRefAttr classSym = handleTy.getClassSym();
  if (!classSym)
    return emitOpError("result type is missing a class symbol");

  // Resolve the referenced symbol starting from the nearest symbol table.
  mlir::Operation *sym =
      mlir::SymbolTable::lookupNearestSymbolFrom(getOperation(), classSym);
  if (!sym)
    return emitOpError("referenced class symbol `")
           << classSym << "` was not found";

  if (!llvm::isa<ClassDeclOp>(sym))
    return emitOpError("symbol `")
           << classSym << "` does not name a `moore.class.classdecl`";

  return mlir::success();
}

void ClassNewOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Always allocates heap memory.
  effects.emplace_back(MemoryEffects::Allocate::get());
}

LogicalResult
ClassUpcastOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // 1) Type checks.
  auto srcTy = dyn_cast<ClassHandleType>(getOperand().getType());
  if (!srcTy)
    return emitOpError() << "operand must be !moore.class<...>; got "
                         << getOperand().getType();

  auto dstTy = dyn_cast<ClassHandleType>(getResult().getType());
  if (!dstTy)
    return emitOpError() << "result must be !moore.class<...>; got "
                         << getResult().getType();

  if (srcTy == dstTy)
    return success();

  auto *op = getOperation();

  auto *srcDeclOp =
      symbolTable.lookupNearestSymbolFrom(op, srcTy.getClassSym());
  auto *dstDeclOp =
      symbolTable.lookupNearestSymbolFrom(op, dstTy.getClassSym());
  if (!srcDeclOp || !dstDeclOp)
    return emitOpError() << "failed to resolve class symbol(s): src="
                         << srcTy.getClassSym()
                         << ", dst=" << dstTy.getClassSym();

  auto srcDecl = dyn_cast<ClassDeclOp>(srcDeclOp);
  auto dstDecl = dyn_cast<ClassDeclOp>(dstDeclOp);
  if (!srcDecl || !dstDecl)
    return emitOpError()
           << "symbol(s) do not name `moore.class.classdecl` ops: src="
           << srcTy.getClassSym() << ", dst=" << dstTy.getClassSym();

  auto cur = srcDecl;
  while (cur) {
    if (cur == dstDecl)
      return success(); // legal upcast: dst is src or an ancestor

    auto baseSym = cur.getBaseAttr();
    if (!baseSym)
      break;

    auto *baseOp = symbolTable.lookupNearestSymbolFrom(op, baseSym);
    cur = llvm::dyn_cast_or_null<ClassDeclOp>(baseOp);
  }

  return emitOpError() << "cannot upcast from " << srcTy.getClassSym() << " to "
                       << dstTy.getClassSym()
                       << " (destination is not a base class)";
}

LogicalResult
ClassPropertyRefOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // The operand is constrained to ClassHandleType in ODS; unwrap it.
  Type instTy = getInstance().getType();
  auto handleTy = dyn_cast<moore::ClassHandleType>(instTy);
  if (!handleTy)
    return emitOpError() << "instance must be a !moore.class<@C> value, got "
                         << instTy;

  // Extract the referenced class symbol from the handle type.
  SymbolRefAttr classSym = handleTy.getClassSym();
  if (!classSym)
    return emitOpError("instance type is missing a class symbol");

  // Resolve the class symbol starting from the nearest symbol table.
  Operation *clsSym =
      symbolTable.lookupNearestSymbolFrom(getOperation(), classSym);
  if (!clsSym)
    return emitOpError("referenced class symbol `")
           << classSym << "` was not found";
  auto classDecl = dyn_cast<ClassDeclOp>(clsSym);
  if (!classDecl)
    return emitOpError("symbol `")
           << classSym << "` does not name a `moore.class.classdecl`";

  // Look up the field symbol inside the class declaration's symbol table.
  FlatSymbolRefAttr fieldSym = getPropertyAttr();
  if (!fieldSym)
    return emitOpError("missing field symbol");

  Operation *fldSym = symbolTable.lookupSymbolIn(classDecl, fieldSym.getAttr());
  if (!fldSym)
    return emitOpError("no field `") << fieldSym << "` in class " << classSym;

  auto fieldDecl = dyn_cast<ClassPropertyDeclOp>(fldSym);
  if (!fieldDecl)
    return emitOpError("symbol `")
           << fieldSym << "` is not a `moore.class.propertydecl`";

  // Result must be !moore.ref<T> where T matches the field's declared type.
  auto resRefTy = cast<RefType>(getPropertyRef().getType());
  if (!resRefTy)
    return emitOpError("result must be a !moore.ref<T>");

  Type expectedElemTy = fieldDecl.getPropertyType();
  if (resRefTy.getNestedType() != expectedElemTy)
    return emitOpError("result element type (")
           << resRefTy.getNestedType() << ") does not match field type ("
           << expectedElemTy << ")";

  return success();
}

LogicalResult
VTableLoadMethodOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *op = getOperation();

  auto object = getObject();
  auto implSym = object.getType().getClassSym();

  // Check that classdecl of class handle exists
  Operation *implOp = symbolTable.lookupNearestSymbolFrom(op, implSym);
  if (!implOp)
    return emitOpError() << "implementing class " << implSym << " not found";
  auto implClass = cast<moore::ClassDeclOp>(implOp);

  StringAttr methodName = getMethodSymAttr().getLeafReference();
  if (!methodName || methodName.getValue().empty())
    return emitOpError() << "empty method name";

  moore::ClassDeclOp cursor = implClass;
  Operation *methodDeclOp = nullptr;

  // Find method in class decl or parents' class decl
  while (cursor && !methodDeclOp) {
    methodDeclOp = symbolTable.lookupSymbolIn(cursor, methodName);
    if (methodDeclOp)
      break;
    SymbolRefAttr baseSym = cursor.getBaseAttr();
    if (!baseSym)
      break;
    Operation *baseOp = symbolTable.lookupNearestSymbolFrom(op, baseSym);
    cursor = baseOp ? cast<moore::ClassDeclOp>(baseOp) : moore::ClassDeclOp();
  }

  if (!methodDeclOp)
    return emitOpError() << "no method `" << methodName << "` found in "
                         << implClass.getSymName() << " or its bases";

  // Make sure method decl is a ClassMethodDeclOp
  auto methodDecl = dyn_cast<moore::ClassMethodDeclOp>(methodDeclOp);
  if (!methodDecl)
    return emitOpError() << "`" << methodName
                         << "` is not a method declaration";

  // Make sure method signature matches
  auto resFnTy = cast<FunctionType>(getResult().getType());
  auto declFnTy = cast<FunctionType>(methodDecl.getFunctionType());
  if (resFnTy != declFnTy)
    return emitOpError() << "result type " << resFnTy
                         << " does not match method erased ABI " << declFnTy;

  return success();
}

LogicalResult VTableOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *self = getOperation();

  // sym_name's root must be a ClassDeclOp
  SymbolRefAttr name = getSymNameAttr();
  if (!name)
    return emitOpError("requires 'sym_name' SymbolRefAttr");

  // Root symbol must resolve (from the nearest symbol table) to a ClassDeclOp.
  Operation *rootDef = symbolTable.lookupNearestSymbolFrom(
      self, SymbolRefAttr::get(name.getRootReference()));
  if (!rootDef)
    return emitOpError() << "cannot resolve root class symbol '"
                         << name.getRootReference() << "' for sym_name "
                         << name;

  if (!isa<ClassDeclOp>(rootDef))
    return emitOpError()
           << "root of sym_name must name a 'moore.class.classdecl', got "
           << name;

  // All good.
  return success();
}

LogicalResult VTableOp::verifyRegions() {
  // Ensure only allowed ops appear inside.
  for (Operation &op : getBody().front()) {
    if (!isa<VTableOp, VTableEntryOp>(op))
      return emitOpError(
          "body may only contain 'moore.vtable' or 'moore.vtable_entry' ops");
  }
  return mlir::success();
}

LogicalResult
VTableEntryOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *self = getOperation();

  // 'target' must exist and resolve from the top-level symbol table of a func
  // op
  SymbolRefAttr target = getTargetAttr();
  func::FuncOp def =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(self, target);
  if (!def)
    return emitOpError()
           << "cannot resolve target symbol to a function operation " << target;

  // VTableEntries may only exist in VTables.
  if (!isa<VTableOp>(self->getParentOp()))
    return emitOpError("must be nested directly inside a 'moore.vtable' op");

  Operation *currentOp = self;
  VTableOp currentVTable;
  bool defined = false;

  // Walk up the VTable tree and check whether the corresponding classDeclOp
  // declares a method with the same implementation. Further checks all the way
  // up the tree if another classdeclop overrides the implementation.
  // The entry is correct iff the impl matches the most derived classdeclop's
  // methoddeclop implementing the virtual method.
  while (auto parentOp = dyn_cast<VTableOp>(currentOp->getParentOp())) {
    currentOp = parentOp;
    currentVTable = cast<VTableOp>(currentOp);

    auto classSymName = currentVTable.getSymName();
    ClassDeclOp parentClassDecl =
        symbolTable.lookupNearestSymbolFrom<ClassDeclOp>(
            parentOp, classSymName.getRootReference());
    assert(parentClassDecl && "VTableOp must point to a classdeclop");

    for (auto method : parentClassDecl.getBody().getOps<ClassMethodDeclOp>()) {
      // A virtual interface declaration. Ignore.
      if (!method.getImpl())
        continue;

      // A matching definition.
      if (method.getSymName() == getName() && method.getImplAttr() == target)
        defined = true;

      // All definitions of the same method up the tree must be the same as the
      // current definition, there is no shadowing.
      // Hence, if we encounter a methoddeclop that has the same name but a
      // different implementation that means this vtableentry should point to
      // the op's implementation - that's an error.
      else if (method.getSymName() == getName() &&
               method.getImplAttr() != target && defined)
        return emitOpError() << "Target " << target
                             << " should be overridden by " << classSymName;
    }
  }
  if (!defined)
    return emitOpError()
           << "Parent class does not point to any implementation!";

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Moore/Moore.cpp.inc"
#include "circt/Dialect/Moore/MooreEnums.cpp.inc"
