//===- HWOps.cpp - Implement the HW operations ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the HW ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/CustomDirectiveImpl.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWInstanceImplementation.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/HWVisitors.h"
#include "circt/Dialect/HW/ModuleImplementation.h"
#include "circt/Support/CustomDirectiveImpl.h"
#include "circt/Support/Namespace.h"
#include "circt/Support/Naming.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"

using namespace circt;
using namespace hw;
using mlir::TypedAttr;

/// Flip a port direction.
ModulePort::Direction hw::flip(ModulePort::Direction direction) {
  switch (direction) {
  case ModulePort::Direction::Input:
    return ModulePort::Direction::Output;
  case ModulePort::Direction::Output:
    return ModulePort::Direction::Input;
  case ModulePort::Direction::InOut:
    return ModulePort::Direction::InOut;
  }
  llvm_unreachable("unknown PortDirection");
}

bool hw::isValidIndexBitWidth(Value index, Value array) {
  hw::ArrayType arrayType =
      dyn_cast<hw::ArrayType>(hw::getCanonicalType(array.getType()));
  assert(arrayType && "expected array type");
  unsigned indexWidth = index.getType().getIntOrFloatBitWidth();
  auto requiredWidth = llvm::Log2_64_Ceil(arrayType.getNumElements());
  return requiredWidth == 0 ? (indexWidth == 0 || indexWidth == 1)
                            : indexWidth == requiredWidth;
}

/// Return true if the specified operation is a combinational logic op.
bool hw::isCombinational(Operation *op) {
  struct IsCombClassifier : public TypeOpVisitor<IsCombClassifier, bool> {
    bool visitInvalidTypeOp(Operation *op) { return false; }
    bool visitUnhandledTypeOp(Operation *op) { return true; }
  };

  return (op->getDialect() && op->getDialect()->getNamespace() == "comb") ||
         IsCombClassifier().dispatchTypeOpVisitor(op);
}

static Value foldStructExtract(Operation *inputOp, uint32_t fieldIndex) {
  // A struct extract of a struct create -> corresponding struct create operand.
  if (auto structCreate = dyn_cast_or_null<StructCreateOp>(inputOp)) {
    return structCreate.getOperand(fieldIndex);
  }

  // Extracting injected field -> corresponding field
  if (auto structInject = dyn_cast_or_null<StructInjectOp>(inputOp)) {
    if (structInject.getFieldIndex() != fieldIndex)
      return {};
    return structInject.getNewValue();
  }
  return {};
}

static ArrayAttr arrayOrEmpty(mlir::MLIRContext *context,
                              ArrayRef<Attribute> attrs) {
  if (attrs.empty())
    return ArrayAttr::get(context, {});
  bool empty = true;
  for (auto a : attrs)
    if (a && !cast<DictionaryAttr>(a).empty()) {
      empty = false;
      break;
    }
  if (empty)
    return ArrayAttr::get(context, {});
  return ArrayAttr::get(context, attrs);
}

/// Get a special name to use when printing the entry block arguments of the
/// region contained by an operation in this dialect.
static void getAsmBlockArgumentNamesImpl(mlir::Region &region,
                                         OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;
  // Assign port names to the bbargs.
  auto module = cast<HWModuleOp>(region.getParentOp());

  auto *block = &region.front();
  for (size_t i = 0, e = block->getNumArguments(); i != e; ++i) {
    auto name = module.getInputName(i);
    // Let mlir deterministically convert names to valid identifiers
    setNameFn(block->getArgument(i), name);
  }
}

enum class Delimiter {
  None,
  Paren,               // () enclosed list
  OptionalLessGreater, // <> enclosed list or absent
};

/// Check parameter specified by `value` to see if it is valid according to the
/// module's parameters.  If not, emit an error to the diagnostic provided as an
/// argument to the lambda 'instanceError' and return failure, otherwise return
/// success.
///
/// If `disallowParamRefs` is true, then parameter references are not allowed.
LogicalResult hw::checkParameterInContext(
    Attribute value, ArrayAttr moduleParameters,
    const instance_like_impl::EmitErrorFn &instanceError,
    bool disallowParamRefs) {
  // Literals are always ok.  Their types are already known to match
  // expectations.
  if (isa<IntegerAttr>(value) || isa<FloatAttr>(value) ||
      isa<StringAttr>(value) || isa<ParamVerbatimAttr>(value))
    return success();

  // Check both subexpressions of an expression.
  if (auto expr = dyn_cast<ParamExprAttr>(value)) {
    for (auto op : expr.getOperands())
      if (failed(checkParameterInContext(op, moduleParameters, instanceError,
                                         disallowParamRefs)))
        return failure();
    return success();
  }

  // Parameter references need more analysis to make sure they are valid within
  // this module.
  if (auto parameterRef = dyn_cast<ParamDeclRefAttr>(value)) {
    auto nameAttr = parameterRef.getName();

    // Don't allow references to parameters from the default values of a
    // parameter list.
    if (disallowParamRefs) {
      instanceError([&](auto &diag) {
        diag << "parameter " << nameAttr
             << " cannot be used as a default value for a parameter";
        return false;
      });
      return failure();
    }

    // Find the corresponding attribute in the module.
    for (auto param : moduleParameters) {
      auto paramAttr = cast<ParamDeclAttr>(param);
      if (paramAttr.getName() != nameAttr)
        continue;

      // If the types match then the reference is ok.
      if (paramAttr.getType() == parameterRef.getType())
        return success();

      instanceError([&](auto &diag) {
        diag << "parameter " << nameAttr << " used with type "
             << parameterRef.getType() << "; should have type "
             << paramAttr.getType();
        return true;
      });
      return failure();
    }

    instanceError([&](auto &diag) {
      diag << "use of unknown parameter " << nameAttr;
      return true;
    });
    return failure();
  }

  instanceError([&](auto &diag) {
    diag << "invalid parameter value " << value;
    return false;
  });
  return failure();
}

/// Check parameter specified by `value` to see if it is valid within the scope
/// of the specified module `module`.  If not, emit an error at the location of
/// `usingOp` and return failure, otherwise return success.  If `usingOp` is
/// null, then no diagnostic is generated.
///
/// If `disallowParamRefs` is true, then parameter references are not allowed.
LogicalResult hw::checkParameterInContext(Attribute value, Operation *module,
                                          Operation *usingOp,
                                          bool disallowParamRefs) {
  instance_like_impl::EmitErrorFn emitError =
      [&](const std::function<bool(InFlightDiagnostic &)> &fn) {
        if (usingOp) {
          auto diag = usingOp->emitOpError();
          if (fn(diag))
            diag.attachNote(module->getLoc()) << "module declared here";
        }
      };

  return checkParameterInContext(value,
                                 module->getAttrOfType<ArrayAttr>("parameters"),
                                 emitError, disallowParamRefs);
}

/// Return true if the specified attribute tree is made up of nodes that are
/// valid in a parameter expression.
bool hw::isValidParameterExpression(Attribute attr, Operation *module) {
  return succeeded(checkParameterInContext(attr, module, nullptr, false));
}

HWModulePortAccessor::HWModulePortAccessor(Location loc,
                                           const ModulePortInfo &info,
                                           Region &bodyRegion)
    : info(info) {
  inputArgs.resize(info.sizeInputs());
  for (auto [i, barg] : llvm::enumerate(bodyRegion.getArguments())) {
    inputIdx[info.at(i).name.str()] = i;
    inputArgs[i] = barg;
  }

  outputOperands.resize(info.sizeOutputs());
  for (auto [i, outputInfo] : llvm::enumerate(info.getOutputs())) {
    outputIdx[outputInfo.name.str()] = i;
  }
}

void HWModulePortAccessor::setOutput(unsigned i, Value v) {
  assert(outputOperands.size() > i && "invalid output index");
  assert(outputOperands[i] == Value() && "output already set");
  outputOperands[i] = v;
}

Value HWModulePortAccessor::getInput(unsigned i) {
  assert(inputArgs.size() > i && "invalid input index");
  return inputArgs[i];
}
Value HWModulePortAccessor::getInput(StringRef name) {
  return getInput(inputIdx.find(name.str())->second);
}
void HWModulePortAccessor::setOutput(StringRef name, Value v) {
  setOutput(outputIdx.find(name.str())->second, v);
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttribute(getValueAttr());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  IntegerAttr valueAttr;

  if (parser.parseAttribute(valueAttr, "value", result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  result.addTypes(valueAttr.getType());
  return success();
}

LogicalResult ConstantOp::verify() {
  // If the result type has a bitwidth, then the attribute must match its width.
  if (getValue().getBitWidth() != cast<IntegerType>(getType()).getWidth())
    return emitError(
        "hw.constant attribute bitwidth doesn't match return type");

  return success();
}

/// Build a ConstantOp from an APInt, infering the result type from the
/// width of the APInt.
void ConstantOp::build(OpBuilder &builder, OperationState &result,
                       const APInt &value) {

  auto type = IntegerType::get(builder.getContext(), value.getBitWidth());
  auto attr = builder.getIntegerAttr(type, value);
  return build(builder, result, type, attr);
}

/// Build a ConstantOp from an APInt, infering the result type from the
/// width of the APInt.
void ConstantOp::build(OpBuilder &builder, OperationState &result,
                       IntegerAttr value) {
  return build(builder, result, value.getType(), value);
}

/// This builder allows construction of small signed integers like 0, 1, -1
/// matching a specified MLIR IntegerType.  This shouldn't be used for general
/// constant folding because it only works with values that can be expressed in
/// an int64_t.  Use APInt's instead.
void ConstantOp::build(OpBuilder &builder, OperationState &result, Type type,
                       int64_t value) {
  auto numBits = cast<IntegerType>(type).getWidth();
  build(builder, result, APInt(numBits, (uint64_t)value, /*isSigned=*/true));
}

void ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  auto intTy = getType();
  auto intCst = getValue();

  // Sugar i1 constants with 'true' and 'false'.
  if (cast<IntegerType>(intTy).getWidth() == 1)
    return setNameFn(getResult(), intCst.isZero() ? "false" : "true");

  // Otherwise, build a complex name with the value and type.
  SmallVector<char, 32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << 'c' << intCst << '_' << intTy;
  setNameFn(getResult(), specialName.str());
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// WireOp
//===----------------------------------------------------------------------===//

/// Check whether an operation has any additional attributes set beyond its
/// standard list of attributes returned by `getAttributeNames`.
template <class Op>
static bool hasAdditionalAttributes(Op op,
                                    ArrayRef<StringRef> ignoredAttrs = {}) {
  auto names = op.getAttributeNames();
  llvm::SmallDenseSet<StringRef> nameSet;
  nameSet.reserve(names.size() + ignoredAttrs.size());
  nameSet.insert(names.begin(), names.end());
  nameSet.insert(ignoredAttrs.begin(), ignoredAttrs.end());
  return llvm::any_of(op->getAttrs(), [&](auto namedAttr) {
    return !nameSet.contains(namedAttr.getName());
  });
}

void WireOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // If the wire has an optional 'name' attribute, use it.
  auto nameAttr = (*this)->getAttrOfType<StringAttr>("name");
  if (nameAttr && !nameAttr.getValue().empty())
    setNameFn(getResult(), nameAttr.getValue());
}

std::optional<size_t> WireOp::getTargetResultIndex() { return 0; }

OpFoldResult WireOp::fold(FoldAdaptor adaptor) {
  // If the wire has no additional attributes, no name, and no symbol, just
  // forward its input.
  if (!hasAdditionalAttributes(*this, {"sv.namehint"}) && !getNameAttr() &&
      !getInnerSymAttr())
    return getInput();
  return {};
}

LogicalResult WireOp::canonicalize(WireOp wire, PatternRewriter &rewriter) {
  // Block if the wire has any attributes.
  if (hasAdditionalAttributes(wire, {"sv.namehint"}))
    return failure();

  // If the wire has a symbol, then we can't delete it.
  if (wire.getInnerSymAttr())
    return failure();

  // If the wire has a name or an `sv.namehint` attribute, propagate it as an
  // `sv.namehint` to the expression.
  if (auto *inputOp = wire.getInput().getDefiningOp())
    if (auto name = chooseName(wire, inputOp))
      rewriter.modifyOpInPlace(inputOp,
                               [&] { inputOp->setAttr("sv.namehint", name); });

  rewriter.replaceOp(wire, wire.getInput());
  return success();
}

//===----------------------------------------------------------------------===//
// AggregateConstantOp
//===----------------------------------------------------------------------===//

static LogicalResult checkAttributes(Operation *op, Attribute attr, Type type) {
  // If this is a type alias, get the underlying type.
  if (auto typeAlias = dyn_cast<TypeAliasType>(type))
    type = typeAlias.getCanonicalType();

  if (auto structType = dyn_cast<StructType>(type)) {
    auto arrayAttr = dyn_cast<ArrayAttr>(attr);
    if (!arrayAttr)
      return op->emitOpError("expected array attribute for constant of type ")
             << type;
    if (structType.getElements().size() != arrayAttr.size())
      return op->emitOpError("array attribute (")
             << arrayAttr.size() << ") has wrong size for struct constant ("
             << structType.getElements().size() << ")";

    for (auto [attr, fieldInfo] :
         llvm::zip(arrayAttr.getValue(), structType.getElements())) {
      if (failed(checkAttributes(op, attr, fieldInfo.type)))
        return failure();
    }
  } else if (auto arrayType = dyn_cast<ArrayType>(type)) {
    auto arrayAttr = dyn_cast<ArrayAttr>(attr);
    if (!arrayAttr)
      return op->emitOpError("expected array attribute for constant of type ")
             << type;
    if (arrayType.getNumElements() != arrayAttr.size())
      return op->emitOpError("array attribute (")
             << arrayAttr.size() << ") has wrong size for array constant ("
             << arrayType.getNumElements() << ")";

    auto elementType = arrayType.getElementType();
    for (auto attr : arrayAttr.getValue()) {
      if (failed(checkAttributes(op, attr, elementType)))
        return failure();
    }
  } else if (auto arrayType = dyn_cast<UnpackedArrayType>(type)) {
    auto arrayAttr = dyn_cast<ArrayAttr>(attr);
    if (!arrayAttr)
      return op->emitOpError("expected array attribute for constant of type ")
             << type;
    auto elementType = arrayType.getElementType();
    if (arrayType.getNumElements() != arrayAttr.size())
      return op->emitOpError("array attribute (")
             << arrayAttr.size()
             << ") has wrong size for unpacked array constant ("
             << arrayType.getNumElements() << ")";

    for (auto attr : arrayAttr.getValue()) {
      if (failed(checkAttributes(op, attr, elementType)))
        return failure();
    }
  } else if (auto enumType = dyn_cast<EnumType>(type)) {
    auto stringAttr = dyn_cast<StringAttr>(attr);
    if (!stringAttr)
      return op->emitOpError("expected string attribute for constant of type ")
             << type;
  } else if (auto intType = dyn_cast<IntegerType>(type)) {
    // Check the attribute kind is correct.
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    if (!intAttr)
      return op->emitOpError("expected integer attribute for constant of type ")
             << type;
    // Check the bitwidth is correct.
    if (intAttr.getValue().getBitWidth() != intType.getWidth())
      return op->emitOpError("hw.constant attribute bitwidth "
                             "doesn't match return type");
  } else {
    return op->emitOpError("unknown element type") << type;
  }
  return success();
}

LogicalResult AggregateConstantOp::verify() {
  return checkAttributes(*this, getFieldsAttr(), getType());
}

OpFoldResult AggregateConstantOp::fold(FoldAdaptor) { return getFieldsAttr(); }

//===----------------------------------------------------------------------===//
// ParamValueOp
//===----------------------------------------------------------------------===//

static ParseResult parseParamValue(OpAsmParser &p, Attribute &value,
                                   Type &resultType) {
  if (p.parseType(resultType) || p.parseEqual() ||
      p.parseAttribute(value, resultType))
    return failure();
  return success();
}

static void printParamValue(OpAsmPrinter &p, Operation *, Attribute value,
                            Type resultType) {
  p << resultType << " = ";
  p.printAttributeWithoutType(value);
}

LogicalResult ParamValueOp::verify() {
  // Check that the attribute expression is valid in this module.
  return checkParameterInContext(
      getValue(), (*this)->getParentOfType<hw::HWModuleOp>(), *this);
}

OpFoldResult ParamValueOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "hw.param.value has no operands");
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// HWModuleOp
//===----------------------------------------------------------------------===/

/// Return true if isAnyModule or instance.
bool hw::isAnyModuleOrInstance(Operation *moduleOrInstance) {
  return isa<HWModuleLike, InstanceOp>(moduleOrInstance);
}

/// Return the signature for a module as a function type from the module itself
/// or from an hw::InstanceOp.
FunctionType hw::getModuleType(Operation *moduleOrInstance) {
  return TypeSwitch<Operation *, FunctionType>(moduleOrInstance)
      .Case<InstanceOp, InstanceChoiceOp>([](auto instance) {
        SmallVector<Type> inputs(instance->getOperandTypes());
        SmallVector<Type> results(instance->getResultTypes());
        return FunctionType::get(instance->getContext(), inputs, results);
      })
      .Case<HWModuleLike>(
          [](auto mod) { return mod.getHWModuleType().getFuncType(); })
      .Default([](Operation *op) {
        return cast<FunctionType>(
            cast<mlir::FunctionOpInterface>(op).getFunctionType());
      });
}

/// Return the name to use for the Verilog module that we're referencing
/// here.  This is typically the symbol, but can be overridden with the
/// verilogName attribute.
StringAttr hw::getVerilogModuleNameAttr(Operation *module) {
  auto nameAttr = module->getAttrOfType<StringAttr>("verilogName");
  if (nameAttr)
    return nameAttr;

  return module->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
}

template <typename ModuleTy>
static void
buildModule(OpBuilder &builder, OperationState &result, StringAttr name,
            const ModulePortInfo &ports, ArrayAttr parameters,
            ArrayRef<NamedAttribute> attributes, StringAttr comment) {
  using namespace mlir::function_interface_impl;

  // Add an attribute for the name.
  result.addAttribute(SymbolTable::getSymbolAttrName(), name);

  SmallVector<Attribute> perPortAttrs;
  SmallVector<ModulePort> portTypes;

  for (auto elt : ports) {
    portTypes.push_back(elt);
    llvm::SmallVector<NamedAttribute> portAttrs;
    if (elt.attrs)
      llvm::copy(elt.attrs, std::back_inserter(portAttrs));
    perPortAttrs.push_back(builder.getDictionaryAttr(portAttrs));
  }

  // Allow clients to pass in null for the parameters list.
  if (!parameters)
    parameters = builder.getArrayAttr({});

  // Record the argument and result types as an attribute.
  auto type = ModuleType::get(builder.getContext(), portTypes);
  result.addAttribute(ModuleTy::getModuleTypeAttrName(result.name),
                      TypeAttr::get(type));
  result.addAttribute("per_port_attrs",
                      arrayOrEmpty(builder.getContext(), perPortAttrs));
  result.addAttribute("parameters", parameters);
  if (!comment)
    comment = builder.getStringAttr("");
  result.addAttribute("comment", comment);
  result.addAttributes(attributes);
  result.addRegion();
}

/// Internal implementation of argument/result insertion and removal on modules.
static void modifyModuleArgs(
    MLIRContext *context, ArrayRef<std::pair<unsigned, PortInfo>> insertArgs,
    ArrayRef<unsigned> removeArgs, ArrayRef<Attribute> oldArgNames,
    ArrayRef<Type> oldArgTypes, ArrayRef<Attribute> oldArgAttrs,
    ArrayRef<Location> oldArgLocs, SmallVector<Attribute> &newArgNames,
    SmallVector<Type> &newArgTypes, SmallVector<Attribute> &newArgAttrs,
    SmallVector<Location> &newArgLocs, Block *body = nullptr) {

#ifndef NDEBUG
  // Check that the `insertArgs` and `removeArgs` indices are in ascending
  // order.
  assert(llvm::is_sorted(insertArgs,
                         [](auto &a, auto &b) { return a.first < b.first; }) &&
         "insertArgs must be in ascending order");
  assert(llvm::is_sorted(removeArgs, [](auto &a, auto &b) { return a < b; }) &&
         "removeArgs must be in ascending order");
#endif

  auto oldArgCount = oldArgTypes.size();
  auto newArgCount = oldArgCount + insertArgs.size() - removeArgs.size();
  assert((int)newArgCount >= 0);

  newArgNames.reserve(newArgCount);
  newArgTypes.reserve(newArgCount);
  newArgAttrs.reserve(newArgCount);
  newArgLocs.reserve(newArgCount);

  auto exportPortAttrName = StringAttr::get(context, "hw.exportPort");
  auto emptyDictAttr = DictionaryAttr::get(context, {});
  auto unknownLoc = UnknownLoc::get(context);

  BitVector erasedIndices;
  if (body)
    erasedIndices.resize(oldArgCount + insertArgs.size());

  for (unsigned argIdx = 0, idx = 0; argIdx <= oldArgCount; ++argIdx, ++idx) {
    // Insert new ports at this position.
    while (!insertArgs.empty() && insertArgs[0].first == argIdx) {
      auto port = insertArgs[0].second;
      if (port.dir == ModulePort::Direction::InOut &&
          !isa<InOutType>(port.type))
        port.type = InOutType::get(port.type);
      auto sym = port.getSym();
      Attribute attr =
          (sym && !sym.empty())
              ? DictionaryAttr::get(context, {{exportPortAttrName, sym}})
              : emptyDictAttr;
      newArgNames.push_back(port.name);
      newArgTypes.push_back(port.type);
      newArgAttrs.push_back(attr);
      insertArgs = insertArgs.drop_front();
      LocationAttr loc = port.loc ? port.loc : unknownLoc;
      newArgLocs.push_back(loc);
      if (body)
        body->insertArgument(idx++, port.type, loc);
    }
    if (argIdx == oldArgCount)
      break;

    // Migrate the old port at this position.
    bool removed = false;
    while (!removeArgs.empty() && removeArgs[0] == argIdx) {
      removeArgs = removeArgs.drop_front();
      removed = true;
    }

    if (removed) {
      if (body)
        erasedIndices.set(idx);
    } else {
      newArgNames.push_back(oldArgNames[argIdx]);
      newArgTypes.push_back(oldArgTypes[argIdx]);
      newArgAttrs.push_back(oldArgAttrs.empty() ? emptyDictAttr
                                                : oldArgAttrs[argIdx]);
      newArgLocs.push_back(oldArgLocs[argIdx]);
    }
  }

  if (body)
    body->eraseArguments(erasedIndices);

  assert(newArgNames.size() == newArgCount);
  assert(newArgTypes.size() == newArgCount);
  assert(newArgAttrs.size() == newArgCount);
  assert(newArgLocs.size() == newArgCount);
}

/// Insert and remove ports of a module. The insertion and removal indices must
/// be in ascending order. The indices refer to the port positions before any
/// insertion or removal occurs. Ports inserted at the same index will appear in
/// the module in the same order as they were listed in the `insert*` array.
///
/// The operation must be any of the module-like operations.
///
/// This is marked deprecated as it's only used from HandshakeToHW and
/// PortConverter and is likely broken and not currently tested.  Users of this
/// are still written dealing with input and output ports separately, which is
/// an old and broken style.
[[deprecated]] static void
modifyModulePorts(Operation *op,
                  ArrayRef<std::pair<unsigned, PortInfo>> insertInputs,
                  ArrayRef<std::pair<unsigned, PortInfo>> insertOutputs,
                  ArrayRef<unsigned> removeInputs,
                  ArrayRef<unsigned> removeOutputs, Block *body = nullptr) {
  auto moduleOp = cast<HWModuleLike>(op);
  auto *context = moduleOp.getContext();

  // Dig up the old argument and result data.
  auto oldArgNames = moduleOp.getInputNames();
  auto oldArgTypes = moduleOp.getInputTypes();
  auto oldArgAttrs = moduleOp.getAllInputAttrs();
  auto oldArgLocs = moduleOp.getInputLocs();

  auto oldResultNames = moduleOp.getOutputNames();
  auto oldResultTypes = moduleOp.getOutputTypes();
  auto oldResultAttrs = moduleOp.getAllOutputAttrs();
  auto oldResultLocs = moduleOp.getOutputLocs();

  // Modify the ports.
  SmallVector<Attribute> newArgNames, newResultNames;
  SmallVector<Type> newArgTypes, newResultTypes;
  SmallVector<Attribute> newArgAttrs, newResultAttrs;
  SmallVector<Location> newArgLocs, newResultLocs;

  modifyModuleArgs(context, insertInputs, removeInputs, oldArgNames,
                   oldArgTypes, oldArgAttrs, oldArgLocs, newArgNames,
                   newArgTypes, newArgAttrs, newArgLocs, body);

  modifyModuleArgs(context, insertOutputs, removeOutputs, oldResultNames,
                   oldResultTypes, oldResultAttrs, oldResultLocs,
                   newResultNames, newResultTypes, newResultAttrs,
                   newResultLocs);

  // Update the module operation types and attributes.
  auto fnty = FunctionType::get(context, newArgTypes, newResultTypes);
  auto modty = detail::fnToMod(fnty, newArgNames, newResultNames);
  moduleOp.setHWModuleType(modty);
  moduleOp.setAllInputAttrs(newArgAttrs);
  moduleOp.setAllOutputAttrs(newResultAttrs);

  newArgLocs.append(newResultLocs.begin(), newResultLocs.end());
  moduleOp.setAllPortLocs(newArgLocs);
}

void HWModuleOp::build(OpBuilder &builder, OperationState &result,
                       StringAttr name, const ModulePortInfo &ports,
                       ArrayAttr parameters,
                       ArrayRef<NamedAttribute> attributes, StringAttr comment,
                       bool shouldEnsureTerminator) {
  buildModule<HWModuleOp>(builder, result, name, ports, parameters, attributes,
                          comment);

  // Create a region and a block for the body.
  auto *bodyRegion = result.regions[0].get();
  Block *body = new Block();
  bodyRegion->push_back(body);

  // Add arguments to the body block.
  auto unknownLoc = builder.getUnknownLoc();
  for (auto port : ports.getInputs()) {
    auto loc = port.loc ? Location(port.loc) : unknownLoc;
    auto type = port.type;
    if (port.isInOut() && !isa<InOutType>(type))
      type = InOutType::get(type);
    body->addArgument(type, loc);
  }

  // Add result ports attribute.
  auto unknownLocAttr = cast<LocationAttr>(unknownLoc);
  SmallVector<Attribute> resultLocs;
  for (auto port : ports.getOutputs())
    resultLocs.push_back(port.loc ? port.loc : unknownLocAttr);
  result.addAttribute("result_locs", builder.getArrayAttr(resultLocs));

  if (shouldEnsureTerminator)
    HWModuleOp::ensureTerminator(*bodyRegion, builder, result.location);
}

void HWModuleOp::build(OpBuilder &builder, OperationState &result,
                       StringAttr name, ArrayRef<PortInfo> ports,
                       ArrayAttr parameters,
                       ArrayRef<NamedAttribute> attributes,
                       StringAttr comment) {
  build(builder, result, name, ModulePortInfo(ports), parameters, attributes,
        comment);
}

void HWModuleOp::build(OpBuilder &builder, OperationState &odsState,
                       StringAttr name, const ModulePortInfo &ports,
                       HWModuleBuilder modBuilder, ArrayAttr parameters,
                       ArrayRef<NamedAttribute> attributes,
                       StringAttr comment) {
  build(builder, odsState, name, ports, parameters, attributes, comment,
        /*shouldEnsureTerminator=*/false);
  auto *bodyRegion = odsState.regions[0].get();
  OpBuilder::InsertionGuard guard(builder);
  auto accessor = HWModulePortAccessor(odsState.location, ports, *bodyRegion);
  builder.setInsertionPointToEnd(&bodyRegion->front());
  modBuilder(builder, accessor);
  // Create output operands.
  llvm::SmallVector<Value> outputOperands = accessor.getOutputOperands();
  builder.create<hw::OutputOp>(odsState.location, outputOperands);
}

void HWModuleOp::modifyPorts(
    ArrayRef<std::pair<unsigned, PortInfo>> insertInputs,
    ArrayRef<std::pair<unsigned, PortInfo>> insertOutputs,
    ArrayRef<unsigned> eraseInputs, ArrayRef<unsigned> eraseOutputs) {
  modifyModulePorts(*this, insertInputs, insertOutputs, eraseInputs,
                    eraseOutputs);
}

/// Return the name to use for the Verilog module that we're referencing
/// here.  This is typically the symbol, but can be overridden with the
/// verilogName attribute.
StringAttr HWModuleExternOp::getVerilogModuleNameAttr() {
  if (auto vName = getVerilogNameAttr())
    return vName;

  return (*this)->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
}

StringAttr HWModuleGeneratedOp::getVerilogModuleNameAttr() {
  if (auto vName = getVerilogNameAttr()) {
    return vName;
  }
  return (*this)->getAttrOfType<StringAttr>(
      ::mlir::SymbolTable::getSymbolAttrName());
}

void HWModuleExternOp::build(OpBuilder &builder, OperationState &result,
                             StringAttr name, const ModulePortInfo &ports,
                             StringRef verilogName, ArrayAttr parameters,
                             ArrayRef<NamedAttribute> attributes) {
  buildModule<HWModuleExternOp>(builder, result, name, ports, parameters,
                                attributes, {});

  // Add the port locations.
  LocationAttr unknownLoc = builder.getUnknownLoc();
  SmallVector<Attribute> portLocs;
  for (auto elt : ports)
    portLocs.push_back(elt.loc ? elt.loc : unknownLoc);
  result.addAttribute("port_locs", builder.getArrayAttr(portLocs));

  if (!verilogName.empty())
    result.addAttribute("verilogName", builder.getStringAttr(verilogName));
}

void HWModuleExternOp::build(OpBuilder &builder, OperationState &result,
                             StringAttr name, ArrayRef<PortInfo> ports,
                             StringRef verilogName, ArrayAttr parameters,
                             ArrayRef<NamedAttribute> attributes) {
  build(builder, result, name, ModulePortInfo(ports), verilogName, parameters,
        attributes);
}

void HWModuleExternOp::modifyPorts(
    ArrayRef<std::pair<unsigned, PortInfo>> insertInputs,
    ArrayRef<std::pair<unsigned, PortInfo>> insertOutputs,
    ArrayRef<unsigned> eraseInputs, ArrayRef<unsigned> eraseOutputs) {
  modifyModulePorts(*this, insertInputs, insertOutputs, eraseInputs,
                    eraseOutputs);
}

void HWModuleExternOp::appendOutputs(
    ArrayRef<std::pair<StringAttr, Value>> outputs) {}

void HWModuleGeneratedOp::build(OpBuilder &builder, OperationState &result,
                                FlatSymbolRefAttr genKind, StringAttr name,
                                const ModulePortInfo &ports,
                                StringRef verilogName, ArrayAttr parameters,
                                ArrayRef<NamedAttribute> attributes) {
  buildModule<HWModuleGeneratedOp>(builder, result, name, ports, parameters,
                                   attributes, {});
  // Add the port locations.
  LocationAttr unknownLoc = builder.getUnknownLoc();
  SmallVector<Attribute> portLocs;
  for (auto elt : ports)
    portLocs.push_back(elt.loc ? elt.loc : unknownLoc);
  result.addAttribute("port_locs", builder.getArrayAttr(portLocs));

  result.addAttribute("generatorKind", genKind);
  if (!verilogName.empty())
    result.addAttribute("verilogName", builder.getStringAttr(verilogName));
}

void HWModuleGeneratedOp::build(OpBuilder &builder, OperationState &result,
                                FlatSymbolRefAttr genKind, StringAttr name,
                                ArrayRef<PortInfo> ports, StringRef verilogName,
                                ArrayAttr parameters,
                                ArrayRef<NamedAttribute> attributes) {
  build(builder, result, genKind, name, ModulePortInfo(ports), verilogName,
        parameters, attributes);
}

void HWModuleGeneratedOp::modifyPorts(
    ArrayRef<std::pair<unsigned, PortInfo>> insertInputs,
    ArrayRef<std::pair<unsigned, PortInfo>> insertOutputs,
    ArrayRef<unsigned> eraseInputs, ArrayRef<unsigned> eraseOutputs) {
  modifyModulePorts(*this, insertInputs, insertOutputs, eraseInputs,
                    eraseOutputs);
}

void HWModuleGeneratedOp::appendOutputs(
    ArrayRef<std::pair<StringAttr, Value>> outputs) {}

static bool hasAttribute(StringRef name, ArrayRef<NamedAttribute> attrs) {
  for (auto &argAttr : attrs)
    if (argAttr.getName() == name)
      return true;
  return false;
}

template <typename ModuleTy>
static ParseResult parseHWModuleOp(OpAsmParser &parser,
                                   OperationState &result) {

  using namespace mlir::function_interface_impl;
  auto builder = parser.getBuilder();
  auto loc = parser.getCurrentLocation();

  // Parse the visibility attribute.
  (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the generator information.
  FlatSymbolRefAttr kindAttr;
  if constexpr (std::is_same_v<ModuleTy, HWModuleGeneratedOp>) {
    if (parser.parseComma() ||
        parser.parseAttribute(kindAttr, "generatorKind", result.attributes)) {
      return failure();
    }
  }

  // Parse the parameters.
  ArrayAttr parameters;
  if (parseOptionalParameterList(parser, parameters))
    return failure();

  SmallVector<module_like_impl::PortParse> ports;
  TypeAttr modType;
  if (failed(module_like_impl::parseModuleSignature(parser, ports, modType)))
    return failure();

  // Parse the attribute dict.
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  if (hasAttribute("parameters", result.attributes)) {
    parser.emitError(loc, "explicit `parameters` attributes not allowed");
    return failure();
  }

  result.addAttribute("parameters", parameters);
  result.addAttribute(ModuleTy::getModuleTypeAttrName(result.name), modType);

  // Convert the specified array of dictionary attrs (which may have null
  // entries) to an ArrayAttr of dictionaries.
  SmallVector<Attribute> attrs;
  for (auto &port : ports)
    attrs.push_back(port.attrs ? port.attrs : builder.getDictionaryAttr({}));
  // Add the attributes to the ports.
  auto nonEmptyAttrsFn = [](Attribute attr) {
    return attr && !cast<DictionaryAttr>(attr).empty();
  };
  if (llvm::any_of(attrs, nonEmptyAttrsFn))
    result.addAttribute(ModuleTy::getPerPortAttrsAttrName(result.name),
                        builder.getArrayAttr(attrs));

  // Add the port locations.
  auto unknownLoc = builder.getUnknownLoc();
  auto nonEmptyLocsFn = [unknownLoc](Attribute attr) {
    return attr && cast<Location>(attr) != unknownLoc;
  };
  SmallVector<Attribute> locs;
  StringAttr portLocsAttrName;
  if constexpr (std::is_same_v<ModuleTy, HWModuleOp>) {
    // Plain modules only store the output port locations, as the input port
    // locations will be stored in the basic block arguments.
    portLocsAttrName = ModuleTy::getResultLocsAttrName(result.name);
    for (auto &port : ports)
      if (port.direction == ModulePort::Direction::Output)
        locs.push_back(port.sourceLoc ? Location(*port.sourceLoc) : unknownLoc);
  } else {
    // All other modules store all port locations in a single array.
    portLocsAttrName = ModuleTy::getPortLocsAttrName(result.name);
    for (auto &port : ports)
      locs.push_back(port.sourceLoc ? Location(*port.sourceLoc) : unknownLoc);
  }
  if (llvm::any_of(locs, nonEmptyLocsFn))
    result.addAttribute(portLocsAttrName, builder.getArrayAttr(locs));

  // Add the entry block arguments.
  SmallVector<OpAsmParser::Argument, 4> entryArgs;
  for (auto &port : ports)
    if (port.direction != ModulePort::Direction::Output)
      entryArgs.push_back(port);

  // Parse the optional function body.
  auto *body = result.addRegion();
  if (std::is_same_v<ModuleTy, HWModuleOp>) {
    if (parser.parseRegion(*body, entryArgs))
      return failure();

    HWModuleOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  }
  return success();
}

ParseResult HWModuleOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseHWModuleOp<HWModuleOp>(parser, result);
}

ParseResult HWModuleExternOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  return parseHWModuleOp<HWModuleExternOp>(parser, result);
}

ParseResult HWModuleGeneratedOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return parseHWModuleOp<HWModuleGeneratedOp>(parser, result);
}

FunctionType getHWModuleOpType(Operation *op) {
  if (auto mod = dyn_cast<HWModuleLike>(op))
    return mod.getHWModuleType().getFuncType();
  return cast<FunctionType>(
      cast<mlir::FunctionOpInterface>(op).getFunctionType());
}

template <typename ModuleTy>
static void printModuleOp(OpAsmPrinter &p, ModuleTy mod) {
  p << ' ';
  // Print the visibility of the module.
  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibility = mod.getOperation()->template getAttrOfType<StringAttr>(
          visibilityAttrName))
    p << visibility.getValue() << ' ';

  // Print the operation and the function name.
  p.printSymbolName(SymbolTable::getSymbolName(mod.getOperation()).getValue());
  if (auto gen = dyn_cast<HWModuleGeneratedOp>(mod.getOperation())) {
    p << ", ";
    p.printSymbolName(gen.getGeneratorKind());
  }

  // Print the parameter list if present.
  printOptionalParameterList(p, mod.getOperation(), mod.getParameters());

  module_like_impl::printModuleSignatureNew(p, mod);

  SmallVector<StringRef, 3> omittedAttrs;
  if (isa<HWModuleGeneratedOp>(mod.getOperation()))
    omittedAttrs.push_back("generatorKind");
  if constexpr (std::is_same_v<ModuleTy, HWModuleOp>)
    omittedAttrs.push_back(mod.getResultLocsAttrName());
  else
    omittedAttrs.push_back(mod.getPortLocsAttrName());
  omittedAttrs.push_back(mod.getModuleTypeAttrName());
  omittedAttrs.push_back(mod.getPerPortAttrsAttrName());
  omittedAttrs.push_back(mod.getParametersAttrName());
  omittedAttrs.push_back(visibilityAttrName);
  if (auto cmt =
          mod.getOperation()->template getAttrOfType<StringAttr>("comment"))
    if (cmt.getValue().empty())
      omittedAttrs.push_back("comment");

  mlir::function_interface_impl::printFunctionAttributes(p, mod.getOperation(),
                                                         omittedAttrs);
}

void HWModuleExternOp::print(OpAsmPrinter &p) { printModuleOp(p, *this); }
void HWModuleGeneratedOp::print(OpAsmPrinter &p) { printModuleOp(p, *this); }

void HWModuleOp::print(OpAsmPrinter &p) {
  printModuleOp(p, *this);

  // Print the body if this is not an external function.
  Region &body = getBody();
  if (!body.empty()) {
    p << " ";
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

static LogicalResult verifyModuleCommon(HWModuleLike module) {
  assert(isa<HWModuleLike>(module) &&
         "verifier hook should only be called on modules");

  SmallPtrSet<Attribute, 4> paramNames;

  // Check parameter default values are sensible.
  for (auto param : module->getAttrOfType<ArrayAttr>("parameters")) {
    auto paramAttr = cast<ParamDeclAttr>(param);

    // Check that we don't have any redundant parameter names.  These are
    // resolved by string name: reuse of the same name would cause ambiguities.
    if (!paramNames.insert(paramAttr.getName()).second)
      return module->emitOpError("parameter ")
             << paramAttr << " has the same name as a previous parameter";

    // Default values are allowed to be missing, check them if present.
    auto value = paramAttr.getValue();
    if (!value)
      continue;

    auto typedValue = dyn_cast<TypedAttr>(value);
    if (!typedValue)
      return module->emitOpError("parameter ")
             << paramAttr << " should have a typed value; has value " << value;

    if (typedValue.getType() != paramAttr.getType())
      return module->emitOpError("parameter ")
             << paramAttr << " should have type " << paramAttr.getType()
             << "; has type " << typedValue.getType();

    // Verify that this is a valid parameter value, disallowing parameter
    // references.  We could allow parameters to refer to each other in the
    // future with lexical ordering if there is a need.
    if (failed(checkParameterInContext(value, module, module,
                                       /*disallowParamRefs=*/true)))
      return failure();
  }
  return success();
}

LogicalResult HWModuleOp::verify() {
  if (failed(verifyModuleCommon(*this)))
    return failure();

  auto type = getModuleType();
  auto *body = getBodyBlock();

  // Verify the number of block arguments.
  auto numInputs = type.getNumInputs();
  if (body->getNumArguments() != numInputs)
    return emitOpError("entry block must have")
           << numInputs << " arguments to match module signature";

  return success();
}

LogicalResult HWModuleExternOp::verify() { return verifyModuleCommon(*this); }

std::pair<StringAttr, BlockArgument>
HWModuleOp::insertInput(unsigned index, StringAttr name, Type ty) {
  // Find a unique name for the wire.
  Namespace ns;
  auto ports = getPortList();
  for (auto port : ports)
    ns.newName(port.name.getValue());
  auto nameAttr = StringAttr::get(getContext(), ns.newName(name.getValue()));

  Block *body = getBodyBlock();

  // Create a new port for the host clock.
  PortInfo port;
  port.name = nameAttr;
  port.dir = ModulePort::Direction::Input;
  port.type = ty;
  modifyModulePorts(getOperation(), {std::make_pair(index, port)}, {}, {}, {},
                    body);

  // Add a new argument.
  return {nameAttr, body->getArgument(index)};
}

void HWModuleOp::insertOutputs(unsigned index,
                               ArrayRef<std::pair<StringAttr, Value>> outputs) {

  auto output = cast<OutputOp>(getBodyBlock()->getTerminator());
  assert(index <= output->getNumOperands() && "invalid output index");

  // Rewrite the port list of the module.
  SmallVector<std::pair<unsigned, PortInfo>> indexedNewPorts;
  for (auto &[name, value] : outputs) {
    PortInfo port;
    port.name = name;
    port.dir = ModulePort::Direction::Output;
    port.type = value.getType();
    indexedNewPorts.emplace_back(index, port);
  }
  modifyModulePorts(getOperation(), {}, indexedNewPorts, {}, {},
                    getBodyBlock());

  // Rewrite the output op.
  for (auto &[name, value] : outputs)
    output->insertOperands(index++, value);
}

void HWModuleOp::appendOutputs(ArrayRef<std::pair<StringAttr, Value>> outputs) {
  return insertOutputs(getNumOutputPorts(), outputs);
}

void HWModuleOp::getAsmBlockArgumentNames(mlir::Region &region,
                                          mlir::OpAsmSetValueNameFn setNameFn) {
  getAsmBlockArgumentNamesImpl(region, setNameFn);
}

void HWModuleExternOp::getAsmBlockArgumentNames(
    mlir::Region &region, mlir::OpAsmSetValueNameFn setNameFn) {
  getAsmBlockArgumentNamesImpl(region, setNameFn);
}

template <typename ModTy>
static SmallVector<Location> getAllPortLocs(ModTy module) {
  auto locs = module.getPortLocs();
  if (locs) {
    SmallVector<Location> retval;
    retval.reserve(locs->size());
    for (auto l : *locs)
      retval.push_back(cast<Location>(l));
    // Either we have a length of 0 or the correct length
    assert(!locs->size() || locs->size() == module.getNumPorts());
    return retval;
  }
  return SmallVector<Location>(module.getNumPorts(),
                               UnknownLoc::get(module.getContext()));
}

SmallVector<Location> HWModuleOp::getAllPortLocs() {
  SmallVector<Location> portLocs;
  portLocs.reserve(getNumPorts());
  auto resultLocs = getResultLocsAttr();
  unsigned inputCount = 0;
  auto modType = getModuleType();
  auto unknownLoc = UnknownLoc::get(getContext());
  auto *body = getBodyBlock();
  for (unsigned i = 0, e = getNumPorts(); i < e; ++i) {
    if (modType.isOutput(i)) {
      auto loc = resultLocs
                     ? cast<Location>(
                           resultLocs.getValue()[portLocs.size() - inputCount])
                     : unknownLoc;
      portLocs.push_back(loc);
    } else {
      auto loc = body ? body->getArgument(inputCount).getLoc() : unknownLoc;
      portLocs.push_back(loc);
      ++inputCount;
    }
  }
  return portLocs;
}

SmallVector<Location> HWModuleExternOp::getAllPortLocs() {
  return ::getAllPortLocs(*this);
}

SmallVector<Location> HWModuleGeneratedOp::getAllPortLocs() {
  return ::getAllPortLocs(*this);
}

void HWModuleOp::setAllPortLocsAttrs(ArrayRef<Attribute> locs) {
  SmallVector<Attribute> resultLocs;
  unsigned inputCount = 0;
  auto modType = getModuleType();
  auto *body = getBodyBlock();
  for (unsigned i = 0, e = getNumPorts(); i < e; ++i) {
    if (modType.isOutput(i))
      resultLocs.push_back(locs[i]);
    else
      body->getArgument(inputCount++).setLoc(cast<Location>(locs[i]));
  }
  setResultLocsAttr(ArrayAttr::get(getContext(), resultLocs));
}

void HWModuleExternOp::setAllPortLocsAttrs(ArrayRef<Attribute> locs) {
  setPortLocsAttr(ArrayAttr::get(getContext(), locs));
}

void HWModuleGeneratedOp::setAllPortLocsAttrs(ArrayRef<Attribute> locs) {
  setPortLocsAttr(ArrayAttr::get(getContext(), locs));
}

template <typename ModTy>
static void setAllPortNames(ArrayRef<Attribute> names, ModTy module) {
  auto numInputs = module.getNumInputPorts();
  SmallVector<Attribute> argNames(names.begin(), names.begin() + numInputs);
  SmallVector<Attribute> resNames(names.begin() + numInputs, names.end());
  auto oldType = module.getModuleType();
  SmallVector<ModulePort> newPorts(oldType.getPorts().begin(),
                                   oldType.getPorts().end());
  for (size_t i = 0UL, e = newPorts.size(); i != e; ++i)
    newPorts[i].name = cast<StringAttr>(names[i]);
  auto newType = ModuleType::get(module.getContext(), newPorts);
  module.setModuleType(newType);
}

void HWModuleOp::setAllPortNames(ArrayRef<Attribute> names) {
  ::setAllPortNames(names, *this);
}

void HWModuleExternOp::setAllPortNames(ArrayRef<Attribute> names) {
  ::setAllPortNames(names, *this);
}

void HWModuleGeneratedOp::setAllPortNames(ArrayRef<Attribute> names) {
  ::setAllPortNames(names, *this);
}

ArrayRef<Attribute> HWModuleOp::getAllPortAttrs() {
  auto attrs = getPerPortAttrs();
  if (attrs && !attrs->empty())
    return attrs->getValue();
  return {};
}

ArrayRef<Attribute> HWModuleExternOp::getAllPortAttrs() {
  auto attrs = getPerPortAttrs();
  if (attrs && !attrs->empty())
    return attrs->getValue();
  return {};
}

ArrayRef<Attribute> HWModuleGeneratedOp::getAllPortAttrs() {
  auto attrs = getPerPortAttrs();
  if (attrs && !attrs->empty())
    return attrs->getValue();
  return {};
}

void HWModuleOp::setAllPortAttrs(ArrayRef<Attribute> attrs) {
  setPerPortAttrsAttr(arrayOrEmpty(getContext(), attrs));
}

void HWModuleExternOp::setAllPortAttrs(ArrayRef<Attribute> attrs) {
  setPerPortAttrsAttr(arrayOrEmpty(getContext(), attrs));
}

void HWModuleGeneratedOp::setAllPortAttrs(ArrayRef<Attribute> attrs) {
  setPerPortAttrsAttr(arrayOrEmpty(getContext(), attrs));
}

void HWModuleOp::removeAllPortAttrs() {
  setPerPortAttrsAttr(ArrayAttr::get(getContext(), {}));
}

void HWModuleExternOp::removeAllPortAttrs() {
  setPerPortAttrsAttr(ArrayAttr::get(getContext(), {}));
}

void HWModuleGeneratedOp::removeAllPortAttrs() {
  setPerPortAttrsAttr(ArrayAttr::get(getContext(), {}));
}

// This probably does really unexpected stuff when you change the number of

template <typename ModTy>
static void setHWModuleType(ModTy &mod, ModuleType type) {
  auto argAttrs = mod.getAllInputAttrs();
  auto resAttrs = mod.getAllOutputAttrs();
  mod.setModuleTypeAttr(TypeAttr::get(type));
  unsigned newNumArgs = type.getNumInputs();
  unsigned newNumResults = type.getNumOutputs();

  auto emptyDict = DictionaryAttr::get(mod.getContext());
  argAttrs.resize(newNumArgs, emptyDict);
  resAttrs.resize(newNumResults, emptyDict);

  SmallVector<Attribute> attrs;
  attrs.append(argAttrs.begin(), argAttrs.end());
  attrs.append(resAttrs.begin(), resAttrs.end());

  if (attrs.empty())
    return mod.removeAllPortAttrs();
  mod.setAllPortAttrs(attrs);
}

void HWModuleOp::setHWModuleType(ModuleType type) {
  return ::setHWModuleType(*this, type);
}

void HWModuleExternOp::setHWModuleType(ModuleType type) {
  return ::setHWModuleType(*this, type);
}

void HWModuleGeneratedOp::setHWModuleType(ModuleType type) {
  return ::setHWModuleType(*this, type);
}

/// Lookup the generator for the symbol.  This returns null on
/// invalid IR.
Operation *HWModuleGeneratedOp::getGeneratorKindOp() {
  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  return topLevelModuleOp.lookupSymbol(getGeneratorKind());
}

LogicalResult
HWModuleGeneratedOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto *referencedKind =
      symbolTable.lookupNearestSymbolFrom(*this, getGeneratorKindAttr());

  if (referencedKind == nullptr)
    return emitError("Cannot find generator definition '")
           << getGeneratorKind() << "'";

  if (!isa<HWGeneratorSchemaOp>(referencedKind))
    return emitError("Symbol resolved to '")
           << referencedKind->getName()
           << "' which is not a HWGeneratorSchemaOp";

  auto referencedKindOp = dyn_cast<HWGeneratorSchemaOp>(referencedKind);
  auto paramRef = referencedKindOp.getRequiredAttrs();
  auto dict = (*this)->getAttrDictionary();
  for (auto str : paramRef) {
    auto strAttr = dyn_cast<StringAttr>(str);
    if (!strAttr)
      return emitError("Unknown attribute type, expected a string");
    if (!dict.get(strAttr.getValue()))
      return emitError("Missing attribute '") << strAttr.getValue() << "'";
  }

  return success();
}

LogicalResult HWModuleGeneratedOp::verify() {
  return verifyModuleCommon(*this);
}

void HWModuleGeneratedOp::getAsmBlockArgumentNames(
    mlir::Region &region, mlir::OpAsmSetValueNameFn setNameFn) {
  getAsmBlockArgumentNamesImpl(region, setNameFn);
}

LogicalResult HWModuleOp::verifyBody() { return success(); }

template <typename ModuleTy>
static SmallVector<PortInfo> getPortList(ModuleTy &mod) {
  auto modTy = mod.getHWModuleType();
  auto emptyDict = DictionaryAttr::get(mod.getContext());
  SmallVector<PortInfo> retval;
  auto locs = mod.getAllPortLocs();
  for (unsigned i = 0, e = modTy.getNumPorts(); i < e; ++i) {
    LocationAttr loc = locs[i];
    DictionaryAttr attrs =
        dyn_cast_or_null<DictionaryAttr>(mod.getPortAttrs(i));
    if (!attrs)
      attrs = emptyDict;
    retval.push_back({modTy.getPorts()[i],
                      modTy.isOutput(i) ? modTy.getOutputIdForPortId(i)
                                        : modTy.getInputIdForPortId(i),
                      attrs, loc});
  }
  return retval;
}

template <typename ModuleTy>
static PortInfo getPort(ModuleTy &mod, size_t idx) {
  auto modTy = mod.getHWModuleType();
  auto emptyDict = DictionaryAttr::get(mod.getContext());
  LocationAttr loc = mod.getPortLoc(idx);
  DictionaryAttr attrs =
      dyn_cast_or_null<DictionaryAttr>(mod.getPortAttrs(idx));
  if (!attrs)
    attrs = emptyDict;
  return {modTy.getPorts()[idx],
          modTy.isOutput(idx) ? modTy.getOutputIdForPortId(idx)
                              : modTy.getInputIdForPortId(idx),
          attrs, loc};
}

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

/// Create a instance that refers to a known module.
void InstanceOp::build(OpBuilder &builder, OperationState &result,
                       Operation *module, StringAttr name,
                       ArrayRef<Value> inputs, ArrayAttr parameters,
                       InnerSymAttr innerSym) {
  if (!parameters)
    parameters = builder.getArrayAttr({});

  auto mod = cast<hw::HWModuleLike>(module);
  auto argNames = builder.getArrayAttr(mod.getInputNames());
  auto resultNames = builder.getArrayAttr(mod.getOutputNames());

  // Try to resolve the parameterized module type. If failed, use the module's
  // parmeterized type. If the client doesn't fix this error, the verifier will
  // fail.
  ModuleType modType = mod.getHWModuleType();
  FailureOr<ModuleType> resolvedModType = modType.resolveParametricTypes(
      parameters, result.location, /*emitErrors=*/false);
  if (succeeded(resolvedModType))
    modType = *resolvedModType;
  FunctionType funcType = resolvedModType->getFuncType();
  build(builder, result, funcType.getResults(), name,
        FlatSymbolRefAttr::get(SymbolTable::getSymbolName(module)), inputs,
        argNames, resultNames, parameters, innerSym);
}

std::optional<size_t> InstanceOp::getTargetResultIndex() {
  // Inner symbols on instance operations target the op not any result.
  return std::nullopt;
}

LogicalResult InstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return instance_like_impl::verifyInstanceOfHWModule(
      *this, getModuleNameAttr(), getInputs(), getResultTypes(), getArgNames(),
      getResultNames(), getParameters(), symbolTable);
}

LogicalResult InstanceOp::verify() {
  auto module = (*this)->getParentOfType<HWModuleOp>();
  if (!module)
    return success();

  auto moduleParameters = module->getAttrOfType<ArrayAttr>("parameters");
  instance_like_impl::EmitErrorFn emitError =
      [&](const std::function<bool(InFlightDiagnostic &)> &fn) {
        auto diag = emitOpError();
        if (fn(diag))
          diag.attachNote(module->getLoc()) << "module declared here";
      };
  return instance_like_impl::verifyParameterStructure(
      getParameters(), moduleParameters, emitError);
}

ParseResult InstanceOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr instanceNameAttr;
  InnerSymAttr innerSym;
  FlatSymbolRefAttr moduleNameAttr;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputsOperands;
  SmallVector<Type, 1> inputsTypes, allResultTypes;
  ArrayAttr argNames, resultNames, parameters;
  auto noneType = parser.getBuilder().getType<NoneType>();

  if (parser.parseAttribute(instanceNameAttr, noneType, "instanceName",
                            result.attributes))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("sym"))) {
    // Parsing an optional symbol name doesn't fail, so no need to check the
    // result.
    if (parser.parseCustomAttributeWithFallback(innerSym))
      return failure();
    result.addAttribute(InnerSymbolTable::getInnerSymbolAttrName(), innerSym);
  }

  llvm::SMLoc parametersLoc, inputsOperandsLoc;
  if (parser.parseAttribute(moduleNameAttr, noneType, "moduleName",
                            result.attributes) ||
      parser.getCurrentLocation(&parametersLoc) ||
      parseOptionalParameterList(parser, parameters) ||
      parseInputPortList(parser, inputsOperands, inputsTypes, argNames) ||
      parser.resolveOperands(inputsOperands, inputsTypes, inputsOperandsLoc,
                             result.operands) ||
      parser.parseArrow() ||
      parseOutputPortList(parser, allResultTypes, resultNames) ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  result.addAttribute("argNames", argNames);
  result.addAttribute("resultNames", resultNames);
  result.addAttribute("parameters", parameters);
  result.addTypes(allResultTypes);
  return success();
}

void InstanceOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printAttributeWithoutType(getInstanceNameAttr());
  if (auto attr = getInnerSymAttr()) {
    p << " sym ";
    attr.print(p);
  }
  p << ' ';
  p.printAttributeWithoutType(getModuleNameAttr());
  printOptionalParameterList(p, *this, getParameters());
  printInputPortList(p, *this, getInputs(), getInputs().getTypes(),
                     getArgNames());
  p << " -> ";
  printOutputPortList(p, *this, getResultTypes(), getResultNames());

  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{"instanceName",
                       InnerSymbolTable::getInnerSymbolAttrName(), "moduleName",
                       "argNames", "resultNames", "parameters"});
}

//===----------------------------------------------------------------------===//
// InstanceChoiceOp
//===----------------------------------------------------------------------===//

std::optional<size_t> InstanceChoiceOp::getTargetResultIndex() {
  // Inner symbols on instance operations target the op not any result.
  return std::nullopt;
}

LogicalResult
InstanceChoiceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  for (Attribute name : getModuleNamesAttr()) {
    if (failed(instance_like_impl::verifyInstanceOfHWModule(
            *this, cast<FlatSymbolRefAttr>(name), getInputs(), getResultTypes(),
            getArgNames(), getResultNames(), getParameters(), symbolTable))) {
      return failure();
    }
  }
  return success();
}

LogicalResult InstanceChoiceOp::verify() {
  auto module = (*this)->getParentOfType<HWModuleOp>();
  if (!module)
    return success();

  auto moduleParameters = module->getAttrOfType<ArrayAttr>("parameters");
  instance_like_impl::EmitErrorFn emitError =
      [&](const std::function<bool(InFlightDiagnostic &)> &fn) {
        auto diag = emitOpError();
        if (fn(diag))
          diag.attachNote(module->getLoc()) << "module declared here";
      };
  return instance_like_impl::verifyParameterStructure(
      getParameters(), moduleParameters, emitError);
}

ParseResult InstanceChoiceOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  StringAttr optionNameAttr;
  StringAttr instanceNameAttr;
  InnerSymAttr innerSym;
  SmallVector<Attribute> moduleNames;
  SmallVector<Attribute> caseNames;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputsOperands;
  SmallVector<Type, 1> inputsTypes, allResultTypes;
  ArrayAttr argNames, resultNames, parameters;
  auto noneType = parser.getBuilder().getType<NoneType>();

  if (parser.parseAttribute(instanceNameAttr, noneType, "instanceName",
                            result.attributes))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("sym"))) {
    // Parsing an optional symbol name doesn't fail, so no need to check the
    // result.
    if (parser.parseCustomAttributeWithFallback(innerSym))
      return failure();
    result.addAttribute(InnerSymbolTable::getInnerSymbolAttrName(), innerSym);
  }

  if (parser.parseKeyword("option") ||
      parser.parseAttribute(optionNameAttr, noneType, "optionName",
                            result.attributes))
    return failure();

  FlatSymbolRefAttr defaultModuleName;
  if (parser.parseAttribute(defaultModuleName))
    return failure();
  moduleNames.push_back(defaultModuleName);

  while (succeeded(parser.parseOptionalKeyword("or"))) {
    FlatSymbolRefAttr moduleName;
    StringAttr targetName;
    if (parser.parseAttribute(moduleName) ||
        parser.parseOptionalKeyword("if") || parser.parseAttribute(targetName))
      return failure();
    moduleNames.push_back(moduleName);
    caseNames.push_back(targetName);
  }

  llvm::SMLoc parametersLoc, inputsOperandsLoc;
  if (parser.getCurrentLocation(&parametersLoc) ||
      parseOptionalParameterList(parser, parameters) ||
      parseInputPortList(parser, inputsOperands, inputsTypes, argNames) ||
      parser.resolveOperands(inputsOperands, inputsTypes, inputsOperandsLoc,
                             result.operands) ||
      parser.parseArrow() ||
      parseOutputPortList(parser, allResultTypes, resultNames) ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  result.addAttribute("moduleNames",
                      ArrayAttr::get(parser.getContext(), moduleNames));
  result.addAttribute("caseNames",
                      ArrayAttr::get(parser.getContext(), caseNames));
  result.addAttribute("argNames", argNames);
  result.addAttribute("resultNames", resultNames);
  result.addAttribute("parameters", parameters);
  result.addTypes(allResultTypes);
  return success();
}

void InstanceChoiceOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printAttributeWithoutType(getInstanceNameAttr());
  if (auto attr = getInnerSymAttr()) {
    p << " sym ";
    attr.print(p);
  }
  p << " option " << getOptionNameAttr() << ' ';

  auto moduleNames = getModuleNamesAttr();
  auto caseNames = getCaseNamesAttr();
  assert(moduleNames.size() == caseNames.size() + 1);

  p.printAttributeWithoutType(moduleNames[0]);
  for (size_t i = 0, n = caseNames.size(); i < n; ++i) {
    p << " or ";
    p.printAttributeWithoutType(moduleNames[i + 1]);
    p << " if ";
    p.printAttributeWithoutType(caseNames[i]);
  }

  printOptionalParameterList(p, *this, getParameters());
  printInputPortList(p, *this, getInputs(), getInputs().getTypes(),
                     getArgNames());
  p << " -> ";
  printOutputPortList(p, *this, getResultTypes(), getResultNames());

  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{"instanceName",
                       InnerSymbolTable::getInnerSymbolAttrName(),
                       "moduleNames", "caseNames", "argNames", "resultNames",
                       "parameters", "optionName"});
}

ArrayAttr InstanceChoiceOp::getReferencedModuleNamesAttr() {
  SmallVector<Attribute> moduleNames;
  for (Attribute attr : getModuleNamesAttr()) {
    moduleNames.push_back(cast<FlatSymbolRefAttr>(attr).getAttr());
  }
  return ArrayAttr::get(getContext(), moduleNames);
}

//===----------------------------------------------------------------------===//
// HWOutputOp
//===----------------------------------------------------------------------===//

/// Verify that the num of operands and types fit the declared results.
LogicalResult OutputOp::verify() {
  // Check that the we (hw.output) have the same number of operands as our
  // region has results.
  ModuleType modType;
  if (auto mod = dyn_cast<HWModuleOp>((*this)->getParentOp()))
    modType = mod.getHWModuleType();
  else {
    emitOpError("must have a module parent");
    return failure();
  }
  auto modResults = modType.getOutputTypes();
  OperandRange outputValues = getOperands();
  if (modResults.size() != outputValues.size()) {
    emitOpError("must have same number of operands as region results.");
    return failure();
  }

  // Check that the types of our operands and the region's results match.
  for (size_t i = 0, e = modResults.size(); i < e; ++i) {
    if (modResults[i] != outputValues[i].getType()) {
      emitOpError("output types must match module. In "
                  "operand ")
          << i << ", expected " << modResults[i] << ", but got "
          << outputValues[i].getType() << ".";
      return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

static ParseResult parseSliceTypes(OpAsmParser &p, Type &srcType,
                                   Type &idxType) {
  Type type;
  if (p.parseType(type))
    return p.emitError(p.getCurrentLocation(), "Expected type");
  auto arrType = type_dyn_cast<ArrayType>(type);
  if (!arrType)
    return p.emitError(p.getCurrentLocation(), "Expected !hw.array type");
  srcType = type;
  unsigned idxWidth = llvm::Log2_64_Ceil(arrType.getNumElements());
  idxType = IntegerType::get(p.getBuilder().getContext(), idxWidth);
  return success();
}

static void printSliceTypes(OpAsmPrinter &p, Operation *, Type srcType,
                            Type idxType) {
  p.printType(srcType);
}

ParseResult ArrayCreateOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 16> operands;
  Type elemType;

  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(elemType))
    return failure();

  if (operands.size() == 0)
    return parser.emitError(inputOperandsLoc,
                            "Cannot construct an array of length 0");
  result.addTypes({ArrayType::get(elemType, operands.size())});

  for (auto operand : operands)
    if (parser.resolveOperand(operand, elemType, result.operands))
      return failure();
  return success();
}

void ArrayCreateOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printOperands(getInputs());
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getInputs()[0].getType();
}

void ArrayCreateOp::build(OpBuilder &b, OperationState &state,
                          ValueRange values) {
  assert(values.size() > 0 && "Cannot build array of zero elements");
  Type elemType = values[0].getType();
  assert(llvm::all_of(
             values,
             [elemType](Value v) -> bool { return v.getType() == elemType; }) &&
         "All values must have same type.");
  build(b, state, ArrayType::get(elemType, values.size()), values);
}

LogicalResult ArrayCreateOp::verify() {
  unsigned returnSize = cast<ArrayType>(getType()).getNumElements();
  if (getInputs().size() != returnSize)
    return failure();
  return success();
}

OpFoldResult ArrayCreateOp::fold(FoldAdaptor adaptor) {
  if (llvm::any_of(adaptor.getInputs(), [](Attribute attr) { return !attr; }))
    return {};
  return ArrayAttr::get(getContext(), adaptor.getInputs());
}

// Check whether an integer value is an offset from a base.
bool hw::isOffset(Value base, Value index, uint64_t offset) {
  if (auto constBase = base.getDefiningOp<hw::ConstantOp>()) {
    if (auto constIndex = index.getDefiningOp<hw::ConstantOp>()) {
      // If both values are a constant, check if index == base + offset.
      // To account for overflow, the addition is performed with an extra bit
      // and the offset is asserted to fit in the bit width of the base.
      auto baseValue = constBase.getValue();
      auto indexValue = constIndex.getValue();

      unsigned bits = baseValue.getBitWidth();
      assert(bits == indexValue.getBitWidth() && "mismatched widths");

      if (bits < 64 && offset >= (1ull << bits))
        return false;

      APInt baseExt = baseValue.zextOrTrunc(bits + 1);
      APInt indexExt = indexValue.zextOrTrunc(bits + 1);
      return baseExt + offset == indexExt;
    }
  }
  return false;
}

// Canonicalize a create of consecutive elements to a slice.
static LogicalResult foldCreateToSlice(ArrayCreateOp op,
                                       PatternRewriter &rewriter) {
  // Do not canonicalize create of get into a slice.
  auto arrayTy = hw::type_cast<ArrayType>(op.getType());
  if (arrayTy.getNumElements() <= 1)
    return failure();
  auto elemTy = arrayTy.getElementType();

  // Check if create arguments are consecutive elements of the same array.
  // Attempt to break a create of gets into a sequence of consecutive intervals.
  struct Chunk {
    Value input;
    Value index;
    size_t size;
  };
  SmallVector<Chunk> chunks;
  for (Value value : llvm::reverse(op.getInputs())) {
    auto get = value.getDefiningOp<ArrayGetOp>();
    if (!get)
      return failure();

    Value input = get.getInput();
    Value index = get.getIndex();
    if (!chunks.empty()) {
      auto &c = *chunks.rbegin();
      if (c.input == get.getInput() && isOffset(c.index, index, c.size)) {
        c.size++;
        continue;
      }
    }

    chunks.push_back(Chunk{input, index, 1});
  }

  // If there is a single slice, eliminate the create.
  if (chunks.size() == 1) {
    auto &chunk = chunks[0];
    rewriter.replaceOp(op, rewriter.createOrFold<ArraySliceOp>(
                               op.getLoc(), arrayTy, chunk.input, chunk.index));
    return success();
  }

  // If the number of chunks is significantly less than the number of
  // elements, replace the create with a concat of the identified slices.
  if (chunks.size() * 2 < arrayTy.getNumElements()) {
    SmallVector<Value> slices;
    for (auto &chunk : llvm::reverse(chunks)) {
      auto sliceTy = ArrayType::get(elemTy, chunk.size);
      slices.push_back(rewriter.createOrFold<ArraySliceOp>(
          op.getLoc(), sliceTy, chunk.input, chunk.index));
    }
    rewriter.replaceOpWithNewOp<ArrayConcatOp>(op, arrayTy, slices);
    return success();
  }

  return failure();
}

LogicalResult ArrayCreateOp::canonicalize(ArrayCreateOp op,
                                          PatternRewriter &rewriter) {
  if (succeeded(foldCreateToSlice(op, rewriter)))
    return success();
  return failure();
}

Value ArrayCreateOp::getUniformElement() {
  if (!getInputs().empty() && llvm::all_equal(getInputs()))
    return getInputs()[0];
  return {};
}

static std::optional<uint64_t> getUIntFromValue(Value value) {
  auto idxOp = dyn_cast_or_null<ConstantOp>(value.getDefiningOp());
  if (!idxOp)
    return std::nullopt;
  APInt idxAttr = idxOp.getValue();
  if (idxAttr.getBitWidth() > 64)
    return std::nullopt;
  return idxAttr.getLimitedValue();
}

LogicalResult ArraySliceOp::verify() {
  unsigned inputSize =
      type_cast<ArrayType>(getInput().getType()).getNumElements();
  if (llvm::Log2_64_Ceil(inputSize) !=
      getLowIndex().getType().getIntOrFloatBitWidth())
    return emitOpError(
        "ArraySlice: index width must match clog2 of array size");
  return success();
}

OpFoldResult ArraySliceOp::fold(FoldAdaptor adaptor) {
  // If we are slicing the entire input, then return it.
  if (getType() == getInput().getType())
    return getInput();
  return {};
}

LogicalResult ArraySliceOp::canonicalize(ArraySliceOp op,
                                         PatternRewriter &rewriter) {
  auto sliceTy = hw::type_cast<ArrayType>(op.getType());
  auto elemTy = sliceTy.getElementType();
  uint64_t sliceSize = sliceTy.getNumElements();
  if (sliceSize == 0)
    return failure();

  if (sliceSize == 1) {
    // slice(a, n) -> create(a[n])
    auto get = rewriter.create<ArrayGetOp>(op.getLoc(), op.getInput(),
                                           op.getLowIndex());
    rewriter.replaceOpWithNewOp<ArrayCreateOp>(op, op.getType(),
                                               get.getResult());
    return success();
  }

  auto offsetOpt = getUIntFromValue(op.getLowIndex());
  if (!offsetOpt)
    return failure();

  auto inputOp = op.getInput().getDefiningOp();
  if (auto inputSlice = dyn_cast_or_null<ArraySliceOp>(inputOp)) {
    // slice(slice(a, n), m) -> slice(a, n + m)
    if (inputSlice == op)
      return failure();

    auto inputIndex = inputSlice.getLowIndex();
    auto inputOffsetOpt = getUIntFromValue(inputIndex);
    if (!inputOffsetOpt)
      return failure();

    uint64_t offset = *offsetOpt + *inputOffsetOpt;
    auto lowIndex =
        rewriter.create<ConstantOp>(op.getLoc(), inputIndex.getType(), offset);
    rewriter.replaceOpWithNewOp<ArraySliceOp>(op, op.getType(),
                                              inputSlice.getInput(), lowIndex);
    return success();
  }

  if (auto inputCreate = dyn_cast_or_null<ArrayCreateOp>(inputOp)) {
    // slice(create(a0, a1, ..., an), m) -> create(am, ...)
    auto inputs = inputCreate.getInputs();

    uint64_t begin = inputs.size() - *offsetOpt - sliceSize;
    rewriter.replaceOpWithNewOp<ArrayCreateOp>(op, op.getType(),
                                               inputs.slice(begin, sliceSize));
    return success();
  }

  if (auto inputConcat = dyn_cast_or_null<ArrayConcatOp>(inputOp)) {
    // slice(concat(a1, a2, ...)) -> concat(a2, slice(a3, ..), ...)
    SmallVector<Value> chunks;
    uint64_t sliceStart = *offsetOpt;
    for (auto input : llvm::reverse(inputConcat.getInputs())) {
      // Check whether the input intersects with the slice.
      uint64_t inputSize =
          hw::type_cast<ArrayType>(input.getType()).getNumElements();
      if (inputSize == 0 || inputSize <= sliceStart) {
        sliceStart -= inputSize;
        continue;
      }

      // Find the indices to slice from this input by intersection.
      uint64_t cutEnd = std::min(inputSize, sliceStart + sliceSize);
      uint64_t cutSize = cutEnd - sliceStart;
      assert(cutSize != 0 && "slice cannot be empty");

      if (cutSize == inputSize) {
        // The whole input fits in the slice, add it.
        assert(sliceStart == 0 && "invalid cut size");
        chunks.push_back(input);
      } else {
        // Slice the required bits from the input.
        unsigned width = inputSize == 1 ? 1 : llvm::Log2_64_Ceil(inputSize);
        auto lowIndex = rewriter.create<ConstantOp>(
            op.getLoc(), rewriter.getIntegerType(width), sliceStart);
        chunks.push_back(rewriter.create<ArraySliceOp>(
            op.getLoc(), hw::ArrayType::get(elemTy, cutSize), input, lowIndex));
      }

      sliceStart = 0;
      sliceSize -= cutSize;
      if (sliceSize == 0)
        break;
    }

    assert(chunks.size() > 0 && "missing sliced items");
    if (chunks.size() == 1)
      rewriter.replaceOp(op, chunks[0]);
    else
      rewriter.replaceOpWithNewOp<ArrayConcatOp>(
          op, llvm::to_vector(llvm::reverse(chunks)));
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// ArrayConcatOp
//===----------------------------------------------------------------------===//

static ParseResult parseArrayConcatTypes(OpAsmParser &p,
                                         SmallVectorImpl<Type> &inputTypes,
                                         Type &resultType) {
  Type elemType;
  uint64_t resultSize = 0;

  auto parseElement = [&]() -> ParseResult {
    Type ty;
    if (p.parseType(ty))
      return failure();
    auto arrTy = type_dyn_cast<ArrayType>(ty);
    if (!arrTy)
      return p.emitError(p.getCurrentLocation(), "Expected !hw.array type");
    if (elemType && elemType != arrTy.getElementType())
      return p.emitError(p.getCurrentLocation(), "Expected array element type ")
             << elemType;

    elemType = arrTy.getElementType();
    inputTypes.push_back(ty);
    resultSize += arrTy.getNumElements();
    return success();
  };

  if (p.parseCommaSeparatedList(parseElement))
    return failure();

  resultType = ArrayType::get(elemType, resultSize);
  return success();
}

static void printArrayConcatTypes(OpAsmPrinter &p, Operation *,
                                  TypeRange inputTypes, Type resultType) {
  llvm::interleaveComma(inputTypes, p, [&p](Type t) { p << t; });
}

void ArrayConcatOp::build(OpBuilder &b, OperationState &state,
                          ValueRange values) {
  assert(!values.empty() && "Cannot build array of zero elements");
  ArrayType arrayTy = cast<ArrayType>(values[0].getType());
  Type elemTy = arrayTy.getElementType();
  assert(llvm::all_of(values,
                      [elemTy](Value v) -> bool {
                        return isa<ArrayType>(v.getType()) &&
                               cast<ArrayType>(v.getType()).getElementType() ==
                                   elemTy;
                      }) &&
         "All values must be of ArrayType with the same element type.");

  uint64_t resultSize = 0;
  for (Value val : values)
    resultSize += cast<ArrayType>(val.getType()).getNumElements();
  build(b, state, ArrayType::get(elemTy, resultSize), values);
}

OpFoldResult ArrayConcatOp::fold(FoldAdaptor adaptor) {
  auto inputs = adaptor.getInputs();
  SmallVector<Attribute> array;
  for (size_t i = 0, e = getNumOperands(); i < e; ++i) {
    if (!inputs[i])
      return {};
    llvm::copy(cast<ArrayAttr>(inputs[i]), std::back_inserter(array));
  }
  return ArrayAttr::get(getContext(), array);
}

// Flatten a concatenation of array creates into a single create.
static bool flattenConcatOp(ArrayConcatOp op, PatternRewriter &rewriter) {
  for (auto input : op.getInputs())
    if (!input.getDefiningOp<ArrayCreateOp>())
      return false;

  SmallVector<Value> items;
  for (auto input : op.getInputs()) {
    auto create = cast<ArrayCreateOp>(input.getDefiningOp());
    for (auto item : create.getInputs())
      items.push_back(item);
  }

  rewriter.replaceOpWithNewOp<ArrayCreateOp>(op, items);
  return true;
}

// Merge consecutive slice expressions in a concatenation.
static bool mergeConcatSlices(ArrayConcatOp op, PatternRewriter &rewriter) {
  struct Slice {
    Value input;
    Value index;
    size_t size;
    Value op;
    SmallVector<Location> locs;
  };

  SmallVector<Value> items;
  std::optional<Slice> last;
  bool changed = false;

  auto concatenate = [&] {
    // If there is only one op in the slice, place it to the items list.
    if (!last)
      return;
    if (last->op) {
      items.push_back(last->op);
      last.reset();
      return;
    }

    // Otherwise, create a new slice of with the given size and place it.
    // In this case, the concat op is replaced, using the new argument.
    changed = true;
    auto loc = FusedLoc::get(op.getContext(), last->locs);
    auto origTy = hw::type_cast<ArrayType>(last->input.getType());
    auto arrayTy = ArrayType::get(origTy.getElementType(), last->size);
    items.push_back(rewriter.createOrFold<ArraySliceOp>(
        loc, arrayTy, last->input, last->index));

    last.reset();
  };

  auto append = [&](Value op, Value input, Value index, size_t size) {
    // If this slice is an extension of the previous one, extend the size
    // saved.  In this case, a new slice of is created and the concatenation
    // operator is rewritten.  Otherwise, flush the last slice.
    if (last) {
      if (last->input == input && isOffset(last->index, index, last->size)) {
        last->size += size;
        last->op = {};
        last->locs.push_back(op.getLoc());
        return;
      }
      concatenate();
    }
    last.emplace(Slice{input, index, size, op, {op.getLoc()}});
  };

  for (auto item : llvm::reverse(op.getInputs())) {
    if (auto slice = item.getDefiningOp<ArraySliceOp>()) {
      auto size = hw::type_cast<ArrayType>(slice.getType()).getNumElements();
      append(item, slice.getInput(), slice.getLowIndex(), size);
      continue;
    }

    if (auto create = item.getDefiningOp<ArrayCreateOp>()) {
      if (create.getInputs().size() == 1) {
        if (auto get = create.getInputs()[0].getDefiningOp<ArrayGetOp>()) {
          append(item, get.getInput(), get.getIndex(), 1);
          continue;
        }
      }
    }

    concatenate();
    items.push_back(item);
  }
  concatenate();

  if (!changed)
    return false;

  if (items.size() == 1) {
    rewriter.replaceOp(op, items[0]);
  } else {
    std::reverse(items.begin(), items.end());
    rewriter.replaceOpWithNewOp<ArrayConcatOp>(op, items);
  }
  return true;
}

LogicalResult ArrayConcatOp::canonicalize(ArrayConcatOp op,
                                          PatternRewriter &rewriter) {
  // concat(create(a1, ...), create(a3, ...), ...) -> create(a1, ..., a3, ...)
  if (flattenConcatOp(op, rewriter))
    return success();

  // concat(slice(a, n, m), slice(a, n + m, p)) -> concat(slice(a, n, m + p))
  if (mergeConcatSlices(op, rewriter))
    return success();

  return failure();
}

//===----------------------------------------------------------------------===//
// EnumConstantOp
//===----------------------------------------------------------------------===//

ParseResult EnumConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse a Type instead of an EnumType since the type might be a type alias.
  // The validity of the canonical type is checked during construction of the
  // EnumFieldAttr.
  Type type;
  StringRef field;

  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  if (parser.parseKeyword(&field) || parser.parseColonType(type))
    return failure();

  auto fieldAttr = EnumFieldAttr::get(
      loc, StringAttr::get(parser.getContext(), field), type);

  if (!fieldAttr)
    return failure();

  result.addAttribute("field", fieldAttr);
  result.addTypes(type);

  return success();
}

void EnumConstantOp::print(OpAsmPrinter &p) {
  p << " " << getField().getField().getValue() << " : "
    << getField().getType().getValue();
}

void EnumConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getField().getField().str());
}

void EnumConstantOp::build(OpBuilder &builder, OperationState &odsState,
                           EnumFieldAttr field) {
  return build(builder, odsState, field.getType().getValue(), field);
}

OpFoldResult EnumConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");
  return getFieldAttr();
}

LogicalResult EnumConstantOp::verify() {
  auto fieldAttr = getFieldAttr();
  auto fieldType = fieldAttr.getType().getValue();
  // This check ensures that we are using the exact same type, without looking
  // through type aliases.
  if (fieldType != getType())
    emitOpError("return type ")
        << getType() << " does not match attribute type " << fieldAttr;
  return success();
}

//===----------------------------------------------------------------------===//
// EnumCmpOp
//===----------------------------------------------------------------------===//

LogicalResult EnumCmpOp::verify() {
  // Compare the canonical types.
  auto lhsType = type_cast<EnumType>(getLhs().getType());
  auto rhsType = type_cast<EnumType>(getRhs().getType());
  if (rhsType != lhsType)
    emitOpError("types do not match");
  return success();
}

//===----------------------------------------------------------------------===//
// StructCreateOp
//===----------------------------------------------------------------------===//

ParseResult StructCreateOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  Type declOrAliasType;

  if (parser.parseLParen() || parser.parseOperandList(operands) ||
      parser.parseRParen() || parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(declOrAliasType))
    return failure();

  auto declType = type_dyn_cast<StructType>(declOrAliasType);
  if (!declType)
    return parser.emitError(parser.getNameLoc(),
                            "expected !hw.struct type or alias");

  llvm::SmallVector<Type, 4> structInnerTypes;
  declType.getInnerTypes(structInnerTypes);
  result.addTypes(declOrAliasType);

  if (parser.resolveOperands(operands, structInnerTypes, inputOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

void StructCreateOp::print(OpAsmPrinter &printer) {
  printer << " (";
  printer.printOperands(getInput());
  printer << ")";
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getType();
}

LogicalResult StructCreateOp::verify() {
  auto elements = hw::type_cast<StructType>(getType()).getElements();

  if (elements.size() != getInput().size())
    return emitOpError("structure field count mismatch");

  for (const auto &[field, value] : llvm::zip(elements, getInput()))
    if (field.type != value.getType())
      return emitOpError("structure field `")
             << field.name << "` type does not match";

  return success();
}

OpFoldResult StructCreateOp::fold(FoldAdaptor adaptor) {
  // struct_create(struct_explode(x)) => x
  if (!getInput().empty())
    if (auto explodeOp = getInput()[0].getDefiningOp<StructExplodeOp>();
        explodeOp && getInput() == explodeOp.getResults() &&
        getResult().getType() == explodeOp.getInput().getType())
      return explodeOp.getInput();

  auto inputs = adaptor.getInput();
  if (llvm::any_of(inputs, [](Attribute attr) { return !attr; }))
    return {};
  return ArrayAttr::get(getContext(), inputs);
}

//===----------------------------------------------------------------------===//
// StructExplodeOp
//===----------------------------------------------------------------------===//

ParseResult StructExplodeOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  OpAsmParser::UnresolvedOperand operand;
  Type declType;

  if (parser.parseOperand(operand) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(declType))
    return failure();
  auto structType = type_dyn_cast<StructType>(declType);
  if (!structType)
    return parser.emitError(parser.getNameLoc(),
                            "invalid kind of type specified");

  llvm::SmallVector<Type, 4> structInnerTypes;
  structType.getInnerTypes(structInnerTypes);
  result.addTypes(structInnerTypes);

  if (parser.resolveOperand(operand, declType, result.operands))
    return failure();
  return success();
}

void StructExplodeOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printOperand(getInput());
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getInput().getType();
}

LogicalResult StructExplodeOp::fold(FoldAdaptor adaptor,
                                    SmallVectorImpl<OpFoldResult> &results) {
  auto input = adaptor.getInput();
  if (!input)
    return failure();
  llvm::copy(cast<ArrayAttr>(input), std::back_inserter(results));
  return success();
}

LogicalResult StructExplodeOp::canonicalize(StructExplodeOp op,
                                            PatternRewriter &rewriter) {
  auto *inputOp = op.getInput().getDefiningOp();
  auto elements = type_cast<StructType>(op.getInput().getType()).getElements();
  auto result = failure();
  auto opResults = op.getResults();
  for (uint32_t index = 0; index < elements.size(); index++) {
    if (auto foldResult = foldStructExtract(inputOp, index)) {
      rewriter.replaceAllUsesWith(opResults[index], foldResult);
      result = success();
    }
  }
  return result;
}

void StructExplodeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  auto structType = type_cast<StructType>(getInput().getType());
  for (auto [res, field] : llvm::zip(getResults(), structType.getElements()))
    setNameFn(res, field.name.str());
}

void StructExplodeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                            Value input) {
  StructType inputType = dyn_cast<StructType>(input.getType());
  assert(inputType);
  SmallVector<Type, 16> fieldTypes;
  for (auto field : inputType.getElements())
    fieldTypes.push_back(field.type);
  build(odsBuilder, odsState, fieldTypes, input);
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

/// Ensure an aggregate op's field index is within the bounds of
/// the aggregate type and the accessed field is of 'elementType'.
template <typename AggregateOp, typename AggregateType>
static LogicalResult verifyAggregateFieldIndexAndType(AggregateOp &op,
                                                      AggregateType aggType,
                                                      Type elementType) {
  auto index = op.getFieldIndex();
  if (index >= aggType.getElements().size())
    return op.emitOpError() << "field index " << index
                            << " exceeds element count of aggregate type";

  if (getCanonicalType(elementType) !=
      getCanonicalType(aggType.getElements()[index].type))
    return op.emitOpError()
           << "type " << aggType.getElements()[index].type
           << " of accessed field in aggregate at index " << index
           << " does not match expected type " << elementType;

  return success();
}

LogicalResult StructExtractOp::verify() {
  return verifyAggregateFieldIndexAndType<StructExtractOp, StructType>(
      *this, getInput().getType(), getType());
}

/// Use the same parser for both struct_extract and union_extract since the
/// syntax is identical.
template <typename AggregateType>
static ParseResult parseExtractOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand operand;
  StringAttr fieldName;
  Type declType;

  if (parser.parseOperand(operand) || parser.parseLSquare() ||
      parser.parseAttribute(fieldName) || parser.parseRSquare() ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(declType))
    return failure();
  auto aggType = type_dyn_cast<AggregateType>(declType);
  if (!aggType)
    return parser.emitError(parser.getNameLoc(),
                            "invalid kind of type specified");

  auto fieldIndex = aggType.getFieldIndex(fieldName);
  if (!fieldIndex) {
    parser.emitError(parser.getNameLoc(), "field name '" +
                                              fieldName.getValue() +
                                              "' not found in aggregate type");
    return failure();
  }

  auto indexAttr =
      IntegerAttr::get(IntegerType::get(parser.getContext(), 32), *fieldIndex);
  result.addAttribute("fieldIndex", indexAttr);
  Type resultType = aggType.getElements()[*fieldIndex].type;
  result.addTypes(resultType);

  if (parser.resolveOperand(operand, declType, result.operands))
    return failure();
  return success();
}

/// Use the same printer for both struct_extract and union_extract since the
/// syntax is identical.
template <typename AggType>
static void printExtractOp(OpAsmPrinter &printer, AggType op) {
  printer << " ";
  printer.printOperand(op.getInput());
  printer << "[\"" << op.getFieldName() << "\"]";
  printer.printOptionalAttrDict(op->getAttrs(), {"fieldIndex"});
  printer << " : " << op.getInput().getType();
}

ParseResult StructExtractOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  return parseExtractOp<StructType>(parser, result);
}

void StructExtractOp::print(OpAsmPrinter &printer) {
  printExtractOp(printer, *this);
}

void StructExtractOp::build(OpBuilder &builder, OperationState &odsState,
                            Value input, StructType::FieldInfo field) {
  auto fieldIndex =
      type_cast<StructType>(input.getType()).getFieldIndex(field.name);
  assert(fieldIndex.has_value() && "field name not found in aggregate type");
  build(builder, odsState, field.type, input, *fieldIndex);
}

void StructExtractOp::build(OpBuilder &builder, OperationState &odsState,
                            Value input, StringAttr fieldName) {
  auto structType = type_cast<StructType>(input.getType());
  auto fieldIndex = structType.getFieldIndex(fieldName);
  assert(fieldIndex.has_value() && "field name not found in aggregate type");
  auto resultType = structType.getElements()[*fieldIndex].type;
  build(builder, odsState, resultType, input, *fieldIndex);
}

OpFoldResult StructExtractOp::fold(FoldAdaptor adaptor) {
  if (auto constOperand = adaptor.getInput()) {
    // Fold extract from aggregate constant
    auto operandAttr = llvm::cast<ArrayAttr>(constOperand);
    return operandAttr.getValue()[getFieldIndex()];
  }

  if (auto foldResult =
          foldStructExtract(getInput().getDefiningOp(), getFieldIndex()))
    return foldResult;
  return {};
}

LogicalResult StructExtractOp::canonicalize(StructExtractOp op,
                                            PatternRewriter &rewriter) {
  auto inputOp = op.getInput().getDefiningOp();

  // b = extract(inject(x["a"], v0)["b"]) => extract(x, "b")
  if (auto structInject = dyn_cast_or_null<StructInjectOp>(inputOp)) {
    if (structInject.getFieldIndex() != op.getFieldIndex()) {
      rewriter.replaceOpWithNewOp<StructExtractOp>(
          op, op.getType(), structInject.getInput(), op.getFieldIndexAttr());
      return success();
    }
  }

  return failure();
}

void StructExtractOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getFieldName());
}

//===----------------------------------------------------------------------===//
// StructInjectOp
//===----------------------------------------------------------------------===//

void StructInjectOp::build(OpBuilder &builder, OperationState &odsState,
                           Value input, StringAttr fieldName, Value newValue) {
  auto structType = type_cast<StructType>(input.getType());
  auto fieldIndex = structType.getFieldIndex(fieldName);
  assert(fieldIndex.has_value() && "field name not found in aggregate type");
  build(builder, odsState, input, *fieldIndex, newValue);
}

LogicalResult StructInjectOp::verify() {
  return verifyAggregateFieldIndexAndType<StructInjectOp, StructType>(
      *this, getInput().getType(), getNewValue().getType());
}

ParseResult StructInjectOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
  OpAsmParser::UnresolvedOperand operand, val;
  StringAttr fieldName;
  Type declType;

  if (parser.parseOperand(operand) || parser.parseLSquare() ||
      parser.parseAttribute(fieldName) || parser.parseRSquare() ||
      parser.parseComma() || parser.parseOperand(val) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(declType))
    return failure();
  auto structType = type_dyn_cast<StructType>(declType);
  if (!structType)
    return parser.emitError(inputOperandsLoc, "invalid kind of type specified");

  auto fieldIndex = structType.getFieldIndex(fieldName);
  if (!fieldIndex) {
    parser.emitError(parser.getNameLoc(), "field name '" +
                                              fieldName.getValue() +
                                              "' not found in aggregate type");
    return failure();
  }

  auto indexAttr =
      IntegerAttr::get(IntegerType::get(parser.getContext(), 32), *fieldIndex);
  result.addAttribute("fieldIndex", indexAttr);
  result.addTypes(declType);

  Type resultType = structType.getElements()[*fieldIndex].type;
  if (parser.resolveOperands({operand, val}, {declType, resultType},
                             inputOperandsLoc, result.operands))
    return failure();
  return success();
}

void StructInjectOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printOperand(getInput());
  printer << "[\"" << getFieldName() << "\"], ";
  printer.printOperand(getNewValue());
  printer.printOptionalAttrDict((*this)->getAttrs(), {"fieldIndex"});
  printer << " : " << getInput().getType();
}

OpFoldResult StructInjectOp::fold(FoldAdaptor adaptor) {
  auto input = adaptor.getInput();
  auto newValue = adaptor.getNewValue();
  if (!input || !newValue)
    return {};
  SmallVector<Attribute> array;
  llvm::copy(cast<ArrayAttr>(input), std::back_inserter(array));
  array[getFieldIndex()] = newValue;
  return ArrayAttr::get(getContext(), array);
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
    inject = dyn_cast_or_null<StructInjectOp>(input.getDefiningOp());
  } while (inject);
  assert(input && "missing input to inject chain");

  auto ty = hw::type_cast<StructType>(op.getType());
  auto elements = ty.getElements();

  // If the inject chain sets all fields, canonicalize to create.
  if (fields.size() == elements.size()) {
    SmallVector<Value> createFields;
    for (const auto &field : elements) {
      auto it = fields.find(field.name);
      assert(it != fields.end() && "missing field");
      createFields.push_back(it->second);
    }
    rewriter.replaceOpWithNewOp<StructCreateOp>(op, ty, createFields);
    return success();
  }

  // Nothing to canonicalize, only the original inject in the chain.
  if (injects.size() == fields.size())
    return failure();

  // Eliminate overwrites. The hash map contains the last write to each field.
  for (uint32_t fieldIndex = 0; fieldIndex < elements.size(); fieldIndex++) {
    auto it = fields.find(elements[fieldIndex].name);
    if (it == fields.end())
      continue;
    input = rewriter.create<StructInjectOp>(op.getLoc(), ty, input, fieldIndex,
                                            it->second);
  }

  rewriter.replaceOp(op, input);
  return success();
}

//===----------------------------------------------------------------------===//
// UnionCreateOp
//===----------------------------------------------------------------------===//

LogicalResult UnionCreateOp::verify() {
  return verifyAggregateFieldIndexAndType<UnionCreateOp, UnionType>(
      *this, getType(), getInput().getType());
}

void UnionCreateOp::build(OpBuilder &builder, OperationState &odsState,
                          Type unionType, StringAttr fieldName, Value input) {
  auto fieldIndex = type_cast<UnionType>(unionType).getFieldIndex(fieldName);
  assert(fieldIndex.has_value() && "field name not found in aggregate type");
  build(builder, odsState, unionType, *fieldIndex, input);
}

ParseResult UnionCreateOp::parse(OpAsmParser &parser, OperationState &result) {
  Type declOrAliasType;
  StringAttr fieldName;
  OpAsmParser::UnresolvedOperand input;
  llvm::SMLoc fieldLoc = parser.getCurrentLocation();

  if (parser.parseAttribute(fieldName) || parser.parseComma() ||
      parser.parseOperand(input) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(declOrAliasType))
    return failure();

  auto declType = type_dyn_cast<UnionType>(declOrAliasType);
  if (!declType)
    return parser.emitError(parser.getNameLoc(),
                            "expected !hw.union type or alias");

  auto fieldIndex = declType.getFieldIndex(fieldName);
  if (!fieldIndex) {
    parser.emitError(fieldLoc, "cannot find union field '")
        << fieldName.getValue() << '\'';
    return failure();
  }

  auto indexAttr =
      IntegerAttr::get(IntegerType::get(parser.getContext(), 32), *fieldIndex);
  result.addAttribute("fieldIndex", indexAttr);
  Type inputType = declType.getElements()[*fieldIndex].type;

  if (parser.resolveOperand(input, inputType, result.operands))
    return failure();
  result.addTypes({declOrAliasType});
  return success();
}

void UnionCreateOp::print(OpAsmPrinter &printer) {
  printer << " \"" << getFieldName() << "\", ";
  printer.printOperand(getInput());
  printer.printOptionalAttrDict((*this)->getAttrs(), {"fieldIndex"});
  printer << " : " << getType();
}

//===----------------------------------------------------------------------===//
// UnionExtractOp
//===----------------------------------------------------------------------===//

ParseResult UnionExtractOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseExtractOp<UnionType>(parser, result);
}

void UnionExtractOp::print(OpAsmPrinter &printer) {
  printExtractOp(printer, *this);
}

LogicalResult UnionExtractOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  UnionExtractOpAdaptor op(operands, attrs,
                           properties.as<UnionExtractOp::Properties *>());
  auto unionElements =
      hw::type_cast<UnionType>(op.getInput().getType()).getElements();
  auto fieldIndex = op.getFieldIndex();
  if (fieldIndex >= unionElements.size()) {
    if (loc)
      mlir::emitError(*loc, "field index " + Twine(fieldIndex) +
                                " exceeds element count of aggregate type");
    return failure();
  }
  results.push_back(unionElements[fieldIndex].type);
  return success();
}

void UnionExtractOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                           Value input, StringAttr fieldName) {
  auto unionType = type_cast<UnionType>(input.getType());
  auto fieldIndex = unionType.getFieldIndex(fieldName);
  assert(fieldIndex.has_value() && "field name not found in aggregate type");
  auto resultType = unionType.getElements()[*fieldIndex].type;
  build(odsBuilder, odsState, resultType, input, *fieldIndex);
}

//===----------------------------------------------------------------------===//
// ArrayGetOp
//===----------------------------------------------------------------------===//

// An array_get of an array_create with a constant index can just be the
// array_create operand at the constant index. If the array_create has a
// single uniform value for each element, just return that value regardless of
// the index. If the array is constructed from a constant by a bitcast
// operation, we can fold into a constant.
OpFoldResult ArrayGetOp::fold(FoldAdaptor adaptor) {
  auto inputCst = dyn_cast_or_null<ArrayAttr>(adaptor.getInput());
  auto indexCst = dyn_cast_or_null<IntegerAttr>(adaptor.getIndex());

  if (inputCst) {
    // Constant array index.
    if (indexCst) {
      auto indexVal = indexCst.getValue();
      if (indexVal.getBitWidth() < 64) {
        auto index = indexVal.getZExtValue();
        return inputCst[inputCst.size() - 1 - index];
      }
    }
    // If all elements of the array are the same, we can return any element of
    // array.
    if (!inputCst.empty() && llvm::all_equal(inputCst))
      return inputCst[0];
  }

  // array_get(bitcast(c), i) -> c[i*w+w-1:i*w]
  if (auto bitcast = getInput().getDefiningOp<hw::BitcastOp>()) {
    auto intTy = dyn_cast<IntegerType>(getType());
    if (!intTy)
      return {};
    auto bitcastInputOp = bitcast.getInput().getDefiningOp<hw::ConstantOp>();
    if (!bitcastInputOp)
      return {};
    if (!indexCst)
      return {};
    auto bitcastInputCst = bitcastInputOp.getValue();
    // Calculate the index. Make sure to zero-extend the index value before
    // multiplying the element width.
    auto startIdx = indexCst.getValue().zext(bitcastInputCst.getBitWidth()) *
                    getType().getIntOrFloatBitWidth();
    // Extract [startIdx + width - 1: startIdx].
    return IntegerAttr::get(intTy, bitcastInputCst.lshr(startIdx).trunc(
                                       intTy.getIntOrFloatBitWidth()));
  }

  auto inputCreate = getInput().getDefiningOp<ArrayCreateOp>();
  if (!inputCreate)
    return {};

  if (auto uniformValue = inputCreate.getUniformElement())
    return uniformValue;

  if (!indexCst || indexCst.getValue().getBitWidth() > 64)
    return {};

  uint64_t index = indexCst.getValue().getLimitedValue();
  auto createInputs = inputCreate.getInputs();
  if (index >= createInputs.size())
    return {};
  return createInputs[createInputs.size() - index - 1];
}

LogicalResult ArrayGetOp::canonicalize(ArrayGetOp op,
                                       PatternRewriter &rewriter) {
  auto idxOpt = getUIntFromValue(op.getIndex());
  if (!idxOpt)
    return failure();

  auto *inputOp = op.getInput().getDefiningOp();
  if (auto inputSlice = dyn_cast_or_null<ArraySliceOp>(inputOp)) {
    // get(slice(a, n), m) -> get(a, n + m)
    auto offsetOp = inputSlice.getLowIndex();
    auto offsetOpt = getUIntFromValue(offsetOp);
    if (!offsetOpt)
      return failure();

    uint64_t offset = *offsetOpt + *idxOpt;
    auto newOffset =
        rewriter.create<ConstantOp>(op.getLoc(), offsetOp.getType(), offset);
    rewriter.replaceOpWithNewOp<ArrayGetOp>(op, inputSlice.getInput(),
                                            newOffset);
    return success();
  }

  if (auto inputConcat = dyn_cast_or_null<ArrayConcatOp>(inputOp)) {
    // get(concat(a0, a1, ...), m) -> get(an, m - s0 - s1 - ...)
    uint64_t elemIndex = *idxOpt;
    for (auto input : llvm::reverse(inputConcat.getInputs())) {
      size_t size = hw::type_cast<ArrayType>(input.getType()).getNumElements();
      if (elemIndex >= size) {
        elemIndex -= size;
        continue;
      }

      unsigned indexWidth = size == 1 ? 1 : llvm::Log2_64_Ceil(size);
      auto newIdxOp = rewriter.create<ConstantOp>(
          op.getLoc(), rewriter.getIntegerType(indexWidth), elemIndex);

      rewriter.replaceOpWithNewOp<ArrayGetOp>(op, input, newIdxOp);
      return success();
    }
    return failure();
  }

  // array_get const, (array_get sel, (array_create a, b, c, d)) -->
  //   array_get sel, (array_create (array_get const a), (array_get const b),
  //   (array_get const, c), (array_get const, d))
  if (auto innerGet = dyn_cast_or_null<hw::ArrayGetOp>(inputOp)) {
    if (!innerGet.getIndex().getDefiningOp<hw::ConstantOp>()) {
      if (auto create =
              innerGet.getInput().getDefiningOp<hw::ArrayCreateOp>()) {

        SmallVector<Value> newValues;
        for (auto operand : create.getOperands())
          newValues.push_back(rewriter.createOrFold<hw::ArrayGetOp>(
              op.getLoc(), operand, op.getIndex()));

        rewriter.replaceOpWithNewOp<hw::ArrayGetOp>(
            op,
            rewriter.createOrFold<hw::ArrayCreateOp>(op.getLoc(), newValues),
            innerGet.getIndex());
        return success();
      }
    }
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// TypedeclOp
//===----------------------------------------------------------------------===//

StringRef TypedeclOp::getPreferredName() {
  return getVerilogName().value_or(getName());
}

Type TypedeclOp::getAliasType() {
  auto parentScope = cast<hw::TypeScopeOp>(getOperation()->getParentOp());
  return hw::TypeAliasType::get(
      SymbolRefAttr::get(parentScope.getSymNameAttr(),
                         {FlatSymbolRefAttr::get(*this)}),
      getType());
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

OpFoldResult BitcastOp::fold(FoldAdaptor) {
  // Identity.
  // bitcast(%a) : A -> A ==> %a
  if (getOperand().getType() == getType())
    return getOperand();

  return {};
}

LogicalResult BitcastOp::canonicalize(BitcastOp op, PatternRewriter &rewriter) {
  // Composition.
  // %b = bitcast(%a) : A -> B
  //      bitcast(%b) : B -> C
  // ===> bitcast(%a) : A -> C
  auto inputBitcast =
      dyn_cast_or_null<BitcastOp>(op.getInput().getDefiningOp());
  if (!inputBitcast)
    return failure();
  auto bitcast = rewriter.createOrFold<BitcastOp>(op.getLoc(), op.getType(),
                                                  inputBitcast.getInput());
  rewriter.replaceOp(op, bitcast);
  return success();
}

LogicalResult BitcastOp::verify() {
  if (getBitWidth(getInput().getType()) != getBitWidth(getResult().getType()))
    return this->emitOpError("Bitwidth of input must match result");
  return success();
}

//===----------------------------------------------------------------------===//
// HierPathOp helpers.
//===----------------------------------------------------------------------===//

bool HierPathOp::dropModule(StringAttr moduleToDrop) {
  SmallVector<Attribute, 4> newPath;
  bool updateMade = false;
  for (auto nameRef : getNamepath()) {
    // nameRef is either an InnerRefAttr or a FlatSymbolRefAttr.
    if (auto ref = dyn_cast<hw::InnerRefAttr>(nameRef)) {
      if (ref.getModule() == moduleToDrop)
        updateMade = true;
      else
        newPath.push_back(ref);
    } else {
      if (cast<FlatSymbolRefAttr>(nameRef).getAttr() == moduleToDrop)
        updateMade = true;
      else
        newPath.push_back(nameRef);
    }
  }
  if (updateMade)
    setNamepathAttr(ArrayAttr::get(getContext(), newPath));
  return updateMade;
}

bool HierPathOp::inlineModule(StringAttr moduleToDrop) {
  SmallVector<Attribute, 4> newPath;
  bool updateMade = false;
  StringRef inlinedInstanceName = "";
  for (auto nameRef : getNamepath()) {
    // nameRef is either an InnerRefAttr or a FlatSymbolRefAttr.
    if (auto ref = dyn_cast<hw::InnerRefAttr>(nameRef)) {
      if (ref.getModule() == moduleToDrop) {
        inlinedInstanceName = ref.getName().getValue();
        updateMade = true;
      } else if (!inlinedInstanceName.empty()) {
        newPath.push_back(hw::InnerRefAttr::get(
            ref.getModule(),
            StringAttr::get(getContext(), inlinedInstanceName + "_" +
                                              ref.getName().getValue())));
        inlinedInstanceName = "";
      } else
        newPath.push_back(ref);
    } else {
      if (cast<FlatSymbolRefAttr>(nameRef).getAttr() == moduleToDrop)
        updateMade = true;
      else
        newPath.push_back(nameRef);
    }
  }
  if (updateMade)
    setNamepathAttr(ArrayAttr::get(getContext(), newPath));
  return updateMade;
}

bool HierPathOp::updateModule(StringAttr oldMod, StringAttr newMod) {
  SmallVector<Attribute, 4> newPath;
  bool updateMade = false;
  for (auto nameRef : getNamepath()) {
    // nameRef is either an InnerRefAttr or a FlatSymbolRefAttr.
    if (auto ref = dyn_cast<hw::InnerRefAttr>(nameRef)) {
      if (ref.getModule() == oldMod) {
        newPath.push_back(hw::InnerRefAttr::get(newMod, ref.getName()));
        updateMade = true;
      } else
        newPath.push_back(ref);
    } else {
      if (cast<FlatSymbolRefAttr>(nameRef).getAttr() == oldMod) {
        newPath.push_back(FlatSymbolRefAttr::get(newMod));
        updateMade = true;
      } else
        newPath.push_back(nameRef);
    }
  }
  if (updateMade)
    setNamepathAttr(ArrayAttr::get(getContext(), newPath));
  return updateMade;
}

bool HierPathOp::updateModuleAndInnerRef(
    StringAttr oldMod, StringAttr newMod,
    const llvm::DenseMap<StringAttr, StringAttr> &innerSymRenameMap) {
  auto fromRef = FlatSymbolRefAttr::get(oldMod);
  if (oldMod == newMod)
    return false;

  auto namepathNew = getNamepath().getValue().vec();
  bool updateMade = false;
  // Break from the loop if the module is found, since it can occur only once.
  for (auto &element : namepathNew) {
    if (auto innerRef = dyn_cast<hw::InnerRefAttr>(element)) {
      if (innerRef.getModule() != oldMod)
        continue;
      auto symName = innerRef.getName();
      // Since the module got updated, the old innerRef symbol inside oldMod
      // should also be updated to the new symbol inside the newMod.
      auto to = innerSymRenameMap.find(symName);
      if (to != innerSymRenameMap.end())
        symName = to->second;
      updateMade = true;
      element = hw::InnerRefAttr::get(newMod, symName);
      break;
    }
    if (element != fromRef)
      continue;

    updateMade = true;
    element = FlatSymbolRefAttr::get(newMod);
    break;
  }
  if (updateMade)
    setNamepathAttr(ArrayAttr::get(getContext(), namepathNew));
  return updateMade;
}

bool HierPathOp::truncateAtModule(StringAttr atMod, bool includeMod) {
  SmallVector<Attribute, 4> newPath;
  bool updateMade = false;
  for (auto nameRef : getNamepath()) {
    // nameRef is either an InnerRefAttr or a FlatSymbolRefAttr.
    if (auto ref = dyn_cast<hw::InnerRefAttr>(nameRef)) {
      if (ref.getModule() == atMod) {
        updateMade = true;
        if (includeMod)
          newPath.push_back(ref);
      } else
        newPath.push_back(ref);
    } else {
      if (cast<FlatSymbolRefAttr>(nameRef).getAttr() == atMod && !includeMod)
        updateMade = true;
      else
        newPath.push_back(nameRef);
    }
    if (updateMade)
      break;
  }
  if (updateMade)
    setNamepathAttr(ArrayAttr::get(getContext(), newPath));
  return updateMade;
}

/// Return just the module part of the namepath at a specific index.
StringAttr HierPathOp::modPart(unsigned i) {
  return TypeSwitch<Attribute, StringAttr>(getNamepath()[i])
      .Case<FlatSymbolRefAttr>([](auto a) { return a.getAttr(); })
      .Case<hw::InnerRefAttr>([](auto a) { return a.getModule(); });
}

/// Return the root module.
StringAttr HierPathOp::root() {
  assert(!getNamepath().empty());
  return modPart(0);
}

/// Return true if the NLA has the module in its path.
bool HierPathOp::hasModule(StringAttr modName) {
  for (auto nameRef : getNamepath()) {
    // nameRef is either an InnerRefAttr or a FlatSymbolRefAttr.
    if (auto ref = dyn_cast<hw::InnerRefAttr>(nameRef)) {
      if (ref.getModule() == modName)
        return true;
    } else {
      if (cast<FlatSymbolRefAttr>(nameRef).getAttr() == modName)
        return true;
    }
  }
  return false;
}

/// Return true if the NLA has the InnerSym .
bool HierPathOp::hasInnerSym(StringAttr modName, StringAttr symName) const {
  for (auto nameRef : const_cast<HierPathOp *>(this)->getNamepath())
    if (auto ref = dyn_cast<hw::InnerRefAttr>(nameRef))
      if (ref.getName() == symName && ref.getModule() == modName)
        return true;

  return false;
}

/// Return just the reference part of the namepath at a specific index.  This
/// will return an empty attribute if this is the leaf and the leaf is a module.
StringAttr HierPathOp::refPart(unsigned i) {
  return TypeSwitch<Attribute, StringAttr>(getNamepath()[i])
      .Case<FlatSymbolRefAttr>([](auto a) { return StringAttr({}); })
      .Case<hw::InnerRefAttr>([](auto a) { return a.getName(); });
}

/// Return the leaf reference.  This returns an empty attribute if the leaf
/// reference is a module.
StringAttr HierPathOp::ref() {
  assert(!getNamepath().empty());
  return refPart(getNamepath().size() - 1);
}

/// Return the leaf module.
StringAttr HierPathOp::leafMod() {
  assert(!getNamepath().empty());
  return modPart(getNamepath().size() - 1);
}

/// Returns true if this NLA targets an instance of a module (as opposed to
/// an instance's port or something inside an instance).
bool HierPathOp::isModule() { return !ref(); }

/// Returns true if this NLA targets something inside a module (as opposed
/// to a module or an instance of a module);
bool HierPathOp::isComponent() { return (bool)ref(); }

// Verify the HierPathOp.
// 1. Iterate over the namepath.
// 2. The namepath should be a valid instance path, specified either on a
// module or a declaration inside a module.
// 3. Each element in the namepath is an InnerRefAttr except possibly the
// last element.
// 4. Make sure that the InnerRefAttr is legal, by verifying the module name
// and the corresponding inner_sym on the instance.
// 5. Make sure that the instance path is legal, by verifying the sequence of
// instance and the expected module occurs as the next element in the path.
// 6. The last element of the namepath, can be an InnerRefAttr on either a
// module port or a declaration inside the module.
// 7. The last element of the namepath can also be a module symbol.
LogicalResult HierPathOp::verifyInnerRefs(hw::InnerRefNamespace &ns) {
  ArrayAttr expectedModuleNames = {};
  auto checkExpectedModule = [&](Attribute name) -> LogicalResult {
    if (!expectedModuleNames)
      return success();
    if (llvm::any_of(expectedModuleNames,
                     [name](Attribute attr) { return attr == name; }))
      return success();
    auto diag = emitOpError() << "instance path is incorrect. Expected ";
    size_t n = expectedModuleNames.size();
    if (n != 1) {
      diag << "one of ";
    }
    for (size_t i = 0; i < n; ++i) {
      if (i != 0)
        diag << ((i + 1 == n) ? " or " : ", ");
      diag << cast<StringAttr>(expectedModuleNames[i]);
    }
    diag << ". Instead found: " << name;
    return diag;
  };

  if (!getNamepath() || getNamepath().empty())
    return emitOpError() << "the instance path cannot be empty";
  for (unsigned i = 0, s = getNamepath().size() - 1; i < s; ++i) {
    hw::InnerRefAttr innerRef = dyn_cast<hw::InnerRefAttr>(getNamepath()[i]);
    if (!innerRef)
      return emitOpError()
             << "the instance path can only contain inner sym reference"
             << ", only the leaf can refer to a module symbol";

    if (failed(checkExpectedModule(innerRef.getModule())))
      return failure();

    auto instOp = ns.lookupOp<igraph::InstanceOpInterface>(innerRef);
    if (!instOp)
      return emitOpError() << " module: " << innerRef.getModule()
                           << " does not contain any instance with symbol: "
                           << innerRef.getName();
    expectedModuleNames = instOp.getReferencedModuleNamesAttr();
  }

  // The instance path has been verified. Now verify the last element.
  auto leafRef = getNamepath()[getNamepath().size() - 1];
  if (auto innerRef = dyn_cast<hw::InnerRefAttr>(leafRef)) {
    if (!ns.lookup(innerRef)) {
      return emitOpError() << " operation with symbol: " << innerRef
                           << " was not found ";
    }
    if (failed(checkExpectedModule(innerRef.getModule())))
      return failure();
  } else if (failed(checkExpectedModule(
                 cast<FlatSymbolRefAttr>(leafRef).getAttr()))) {
    return failure();
  }
  return success();
}

void HierPathOp::print(OpAsmPrinter &p) {
  p << " ";

  // Print visibility if present.
  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibility =
          getOperation()->getAttrOfType<StringAttr>(visibilityAttrName))
    p << visibility.getValue() << ' ';

  p.printSymbolName(getSymName());
  p << " [";
  llvm::interleaveComma(getNamepath().getValue(), p, [&](Attribute attr) {
    if (auto ref = dyn_cast<hw::InnerRefAttr>(attr)) {
      p.printSymbolName(ref.getModule().getValue());
      p << "::";
      p.printSymbolName(ref.getName().getValue());
    } else {
      p.printSymbolName(cast<FlatSymbolRefAttr>(attr).getValue());
    }
  });
  p << "]";
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      {SymbolTable::getSymbolAttrName(), "namepath", visibilityAttrName});
}

ParseResult HierPathOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the visibility attribute.
  (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  // Parse the symbol name.
  StringAttr symName;
  if (parser.parseSymbolName(symName, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the namepath.
  SmallVector<Attribute> namepath;
  if (parser.parseCommaSeparatedList(
          OpAsmParser::Delimiter::Square, [&]() -> ParseResult {
            auto loc = parser.getCurrentLocation();
            SymbolRefAttr ref;
            if (parser.parseAttribute(ref))
              return failure();

            // "A" is a Ref, "A::b" is a InnerRef, "A::B::c" is an error.
            auto pathLength = ref.getNestedReferences().size();
            if (pathLength == 0)
              namepath.push_back(
                  FlatSymbolRefAttr::get(ref.getRootReference()));
            else if (pathLength == 1)
              namepath.push_back(hw::InnerRefAttr::get(ref.getRootReference(),
                                                       ref.getLeafReference()));
            else
              return parser.emitError(loc,
                                      "only one nested reference is allowed");
            return success();
          }))
    return failure();
  result.addAttribute("namepath",
                      ArrayAttr::get(parser.getContext(), namepath));

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// TriggeredOp
//===----------------------------------------------------------------------===//

void TriggeredOp::build(OpBuilder &builder, OperationState &odsState,
                        EventControlAttr event, Value trigger,
                        ValueRange inputs) {
  odsState.addOperands(trigger);
  odsState.addOperands(inputs);
  odsState.addAttribute(getEventAttrName(odsState.name), event);
  auto *r = odsState.addRegion();
  Block *b = new Block();
  r->push_back(b);

  llvm::SmallVector<Location> argLocs;
  llvm::transform(inputs, std::back_inserter(argLocs),
                  [&](Value v) { return v.getLoc(); });
  b->addArguments(inputs.getTypes(), argLocs);
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/HW/HW.cpp.inc"
