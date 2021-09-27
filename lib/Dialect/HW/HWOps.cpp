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
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWVisitors.h"
#include "circt/Dialect/HW/ModuleImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace circt;
using namespace hw;

/// Return true if the specified operation is a combinational logic op.
bool hw::isCombinational(Operation *op) {
  struct IsCombClassifier : public TypeOpVisitor<IsCombClassifier, bool> {
    bool visitInvalidTypeOp(Operation *op) { return false; }
    bool visitUnhandledTypeOp(Operation *op) { return true; }
  };

  return (op->getDialect() && op->getDialect()->getNamespace() == "comb") ||
         IsCombClassifier().dispatchTypeOpVisitor(op);
}

enum class Delimiter {
  None,
  Paren,               // () enclosed list
  OptionalLessGreater, // <> enclosed list or absent
};

/// Check parameter specified by `value` to see if it is valid within the scope
/// of the specified module `module`.  If not, emit an error at the location of
/// `usingOp` and return failure, otherwise return success.  If `usingOp` is
/// null, then no diagnostic is generated.
///
/// If `disallowParamRefs` is true, then parameter references are not allowed.
LogicalResult hw::checkParameterInContext(Attribute value, Operation *module,
                                          Operation *usingOp,
                                          bool disallowParamRefs) {
  // Literals are always ok.  Their types are already known to match
  // expectations.
  if (value.isa<IntegerAttr>() || value.isa<FloatAttr>() ||
      value.isa<StringAttr>() || value.isa<ParamVerbatimAttr>())
    return success();

  // Check both arms of an expression.
  if (auto binop = value.dyn_cast<ParamBinaryAttr>()) {
    if (failed(checkParameterInContext(binop.getLhs(), module, usingOp,
                                       disallowParamRefs)) ||
        failed(checkParameterInContext(binop.getRhs(), module, usingOp,
                                       disallowParamRefs)))
      return failure();
    return success();
  }

  // Parameter references need more analysis to make sure they are valid within
  // this module.
  if (auto parameterRef = value.dyn_cast<ParamDeclRefAttr>()) {
    auto nameAttr = parameterRef.getName();

    // Don't allow references to parameters from the default values of a
    // parameter list.
    if (disallowParamRefs) {
      if (usingOp)
        usingOp->emitOpError("parameter ")
            << nameAttr << " cannot be used as a default value for a parameter";
      return failure();
    }

    // Find the corresponding attribute in the module.
    for (auto param : module->getAttrOfType<ArrayAttr>("parameters")) {
      auto paramAttr = param.cast<ParamDeclAttr>();
      if (paramAttr.getName() != nameAttr)
        continue;

      // If the types match then the reference is ok.
      if (paramAttr.getType().getValue() == parameterRef.getType())
        return success();

      if (usingOp) {
        auto diag = usingOp->emitOpError("parameter ")
                    << nameAttr << " used with type " << parameterRef.getType()
                    << "; should have type " << paramAttr.getType().getValue();
        diag.attachNote(module->getLoc()) << "module declared here";
      }
      return failure();
    }

    if (usingOp) {
      auto diag = usingOp->emitOpError("use of unknown parameter ") << nameAttr;
      diag.attachNote(module->getLoc()) << "module declared here";
    }
    return failure();
  }

  if (usingOp)
    usingOp->emitOpError("invalid parameter value ") << value;
  return failure();
}

/// Return true if the specified attribute tree is made up of nodes that are
/// valid in a parameter expression.
bool hw::isValidParameterExpression(Attribute attr, Operation *module) {
  return succeeded(checkParameterInContext(attr, module, nullptr, false));
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

static void printConstantOp(OpAsmPrinter &p, ConstantOp &op) {
  p << " ";
  p.printAttribute(op.valueAttr());
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});
}

static ParseResult parseConstantOp(OpAsmParser &parser,
                                   OperationState &result) {
  IntegerAttr valueAttr;

  if (parser.parseAttribute(valueAttr, "value", result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  result.addTypes(valueAttr.getType());
  return success();
}

static LogicalResult verifyConstantOp(ConstantOp constant) {
  // If the result type has a bitwidth, then the attribute must match its width.
  if (constant.value().getBitWidth() != constant.getType().getWidth())
    return constant.emitError(
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
  auto numBits = type.cast<IntegerType>().getWidth();
  build(builder, result, APInt(numBits, (uint64_t)value, /*isSigned=*/true));
}

void ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  auto intTy = getType();
  auto intCst = getValue();

  // Sugar i1 constants with 'true' and 'false'.
  if (intTy.getWidth() == 1)
    return setNameFn(getResult(), intCst.isZero() ? "false" : "true");

  // Otherwise, build a complex name with the value and type.
  SmallVector<char, 32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << 'c' << intCst << '_' << intTy;
  setNameFn(getResult(), specialName.str());
}

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> constants) {
  assert(constants.empty() && "constant has no operands");
  return valueAttr();
}

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

static LogicalResult verifyParamValueOp(ParamValueOp op) {
  // Check that the attribute expression is valid in this module.
  return checkParameterInContext(op.value(),
                                 op->getParentOfType<hw::HWModuleOp>(), op);
}

OpFoldResult ParamValueOp::fold(ArrayRef<Attribute> constants) {
  assert(constants.empty() && "hw.param.value has no operands");
  return valueAttr();
}

//===----------------------------------------------------------------------===//
// HWModuleOp
//===----------------------------------------------------------------------===/

/// Return true if this is an hw.module, external module, generated module etc.
bool hw::isAnyModule(Operation *module) {
  return isa<HWModuleOp>(module) || isa<HWModuleExternOp>(module) ||
         isa<HWModuleGeneratedOp>(module);
}

/// Return the signature for a module as a function type from the module itself
/// or from an hw::InstanceOp.
FunctionType hw::getModuleType(Operation *moduleOrInstance) {
  if (auto instance = dyn_cast<InstanceOp>(moduleOrInstance)) {
    SmallVector<Type> inputs(instance->getOperandTypes());
    SmallVector<Type> results(instance->getResultTypes());
    return FunctionType::get(instance->getContext(), inputs, results);
  }

  assert(isAnyModule(moduleOrInstance) &&
         "must be called on instance or module");
  auto typeAttr =
      moduleOrInstance->getAttrOfType<TypeAttr>(HWModuleOp::getTypeAttrName());
  return typeAttr.getValue().cast<FunctionType>();
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

/// Return the port name for the specified argument or result.
StringAttr hw::getModuleArgumentNameAttr(Operation *module, size_t argNo) {
  auto argNames = module->getAttrOfType<ArrayAttr>("argNames");
  // Tolerate malformed IR here to enable debug printing etc.
  if (argNames && argNo < argNames.size())
    return argNames[argNo].cast<StringAttr>();
  return StringAttr();
}

StringAttr hw::getModuleResultNameAttr(Operation *module, size_t resultNo) {
  auto resultNames = module->getAttrOfType<ArrayAttr>("resultNames");
  // Tolerate malformed IR here to enable debug printing etc.
  if (resultNames && resultNo < resultNames.size())
    return resultNames[resultNo].cast<StringAttr>();
  return StringAttr();
}

void hw::setModuleArgumentNames(Operation *module, ArrayRef<Attribute> names) {
  assert(isAnyModule(module) && "Must be called on a module");
  assert(getModuleType(module).getNumInputs() == names.size() &&
         "incorrect number of arguments names specified");
  module->setAttr("argNames", ArrayAttr::get(module->getContext(), names));
}

void hw::setModuleResultNames(Operation *module, ArrayRef<Attribute> names) {
  assert(isAnyModule(module) && "Must be called on a module");
  assert(getModuleType(module).getNumResults() == names.size() &&
         "incorrect number of arguments names specified");
  module->setAttr("resultNames", ArrayAttr::get(module->getContext(), names));
}

// Flag for parsing different module types
enum ExternModKind { PlainMod, ExternMod, GenMod };

static void buildModule(OpBuilder &builder, OperationState &result,
                        StringAttr name, const ModulePortInfo &ports,
                        ArrayAttr parameters,
                        ArrayRef<NamedAttribute> attributes) {
  using namespace mlir::function_like_impl;

  // Add an attribute for the name.
  result.addAttribute(SymbolTable::getSymbolAttrName(), name);

  SmallVector<Attribute> argNames, resultNames;

  SmallVector<Type, 4> argTypes;
  for (auto elt : ports.inputs) {
    if (elt.direction == PortDirection::INOUT && !elt.type.isa<hw::InOutType>())
      elt.type = hw::InOutType::get(elt.type);
    argTypes.push_back(elt.type);
    argNames.push_back(elt.name);
  }

  SmallVector<Type, 4> resultTypes;
  for (auto elt : ports.outputs) {
    resultTypes.push_back(elt.type);
    resultNames.push_back(elt.name);
  }

  // Allow clients to pass in null for the parameters list.
  if (!parameters)
    parameters = builder.getArrayAttr({});

  // Record the argument and result types as an attribute.
  auto type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  result.addAttribute("argNames", builder.getArrayAttr(argNames));
  result.addAttribute("resultNames", builder.getArrayAttr(resultNames));
  result.addAttribute("parameters", parameters);
  result.addAttributes(attributes);
  result.addRegion();
}

void HWModuleOp::build(OpBuilder &builder, OperationState &result,
                       StringAttr name, const ModulePortInfo &ports,
                       ArrayAttr parameters,
                       ArrayRef<NamedAttribute> attributes) {
  buildModule(builder, result, name, ports, parameters, attributes);

  // Create a region and a block for the body.
  auto *bodyRegion = result.regions[0].get();
  Block *body = new Block();
  bodyRegion->push_back(body);

  // Add arguments to the body block.
  for (auto elt : ports.inputs)
    body->addArgument(elt.type);

  HWModuleOp::ensureTerminator(*bodyRegion, builder, result.location);
}

void HWModuleOp::build(OpBuilder &builder, OperationState &result,
                       StringAttr name, ArrayRef<PortInfo> ports,
                       ArrayAttr parameters,
                       ArrayRef<NamedAttribute> attributes) {
  build(builder, result, name, ModulePortInfo(ports), parameters, attributes);
}

/// Return the name to use for the Verilog module that we're referencing
/// here.  This is typically the symbol, but can be overridden with the
/// verilogName attribute.
StringRef HWModuleExternOp::getVerilogModuleName() {
  if (auto vname = verilogName())
    return vname.getValue();
  return getName();
}

/// Return the name to use for the Verilog module that we're referencing
/// here.  This is typically the symbol, but can be overridden with the
/// verilogName attribute.
StringAttr HWModuleExternOp::getVerilogModuleNameAttr() {
  if (auto vName = verilogNameAttr())
    return vName;

  return (*this)->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
}

void HWModuleExternOp::build(OpBuilder &builder, OperationState &result,
                             StringAttr name, const ModulePortInfo &ports,
                             StringRef verilogName, ArrayAttr parameters,
                             ArrayRef<NamedAttribute> attributes) {
  buildModule(builder, result, name, ports, parameters, attributes);

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

void HWModuleGeneratedOp::build(OpBuilder &builder, OperationState &result,
                                FlatSymbolRefAttr genKind, StringAttr name,
                                const ModulePortInfo &ports,
                                StringRef verilogName, ArrayAttr parameters,
                                ArrayRef<NamedAttribute> attributes) {
  buildModule(builder, result, name, ports, parameters, attributes);
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

/// Return an encapsulated set of information about input and output ports of
/// the specified module or instance.  The input ports always come before the
/// output ports in the list.
ModulePortInfo hw::getModulePortInfo(Operation *op) {
  assert((isa<InstanceOp>(op) || isAnyModule(op)) &&
         "Can only get module ports from an instance or module");

  SmallVector<PortInfo> inputs, outputs;
  auto argTypes = getModuleType(op).getInputs();

  auto argNames = op->getAttrOfType<ArrayAttr>("argNames");
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    bool isInOut = false;
    auto type = argTypes[i];

    if (auto inout = type.dyn_cast<InOutType>()) {
      isInOut = true;
      type = inout.getElementType();
    }

    auto direction = isInOut ? PortDirection::INOUT : PortDirection::INPUT;
    inputs.push_back({argNames[i].cast<StringAttr>(), direction, type, i});
  }

  auto resultNames = op->getAttrOfType<ArrayAttr>("resultNames");
  auto resultTypes = getModuleType(op).getResults();
  for (unsigned i = 0, e = resultTypes.size(); i < e; ++i) {
    outputs.push_back({resultNames[i].cast<StringAttr>(), PortDirection::OUTPUT,
                       resultTypes[i], i});
  }
  return ModulePortInfo(inputs, outputs);
}

/// Return an encapsulated set of information about input and output ports of
/// the specified module or instance.  The input ports always come before the
/// output ports in the list.
SmallVector<PortInfo> hw::getAllModulePortInfos(Operation *op) {
  assert((isa<InstanceOp>(op) || isAnyModule(op)) &&
         "Can only get module ports from an instance or module");

  SmallVector<PortInfo> results;
  auto argTypes = getModuleType(op).getInputs();
  auto argNames = op->getAttrOfType<ArrayAttr>("argNames");
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    bool isInOut = false;
    auto type = argTypes[i];

    if (auto inout = type.dyn_cast<InOutType>()) {
      isInOut = true;
      type = inout.getElementType();
    }

    auto direction = isInOut ? PortDirection::INOUT : PortDirection::INPUT;
    results.push_back({argNames[i].cast<StringAttr>(), direction, type, i});
  }

  auto resultNames = op->getAttrOfType<ArrayAttr>("resultNames");
  auto resultTypes = getModuleType(op).getResults();
  for (unsigned i = 0, e = resultTypes.size(); i < e; ++i) {
    results.push_back({resultNames[i].cast<StringAttr>(), PortDirection::OUTPUT,
                       resultTypes[i], i});
  }
  return results;
}

static bool hasAttribute(StringRef name, ArrayRef<NamedAttribute> attrs) {
  for (auto &argAttr : attrs)
    if (argAttr.first == name)
      return true;
  return false;
}

/// Parse an parameter list if present.
/// module-parameter-list ::= `<` parameter-decl (`,` parameter-decl)* `>`
/// parameter-decl ::= identifier `:` type
/// parameter-decl ::= identifier `:` type `=` attribute
///
static ParseResult parseOptionalParameters(OpAsmParser &parser,
                                           SmallVector<Attribute> &parameters) {

  return parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::OptionalLessGreater, [&]() {
        StringAttr name;
        Type type;
        Attribute value;

        if (!(name = module_like_impl::parsePortName(parser)) ||
            parser.parseColonType(type))
          return failure();

        // Parse the default value if present.
        if (succeeded(parser.parseOptionalEqual())) {
          if (parser.parseAttribute(value, type))
            return failure();
        }

        auto &builder = parser.getBuilder();
        parameters.push_back(ParamDeclAttr::get(builder.getContext(), name,
                                                TypeAttr::get(type), value));
        return success();
      });
}

static ParseResult parseHWModuleOp(OpAsmParser &parser, OperationState &result,
                                   ExternModKind modKind = PlainMod) {
  using namespace mlir::function_like_impl;

  auto loc = parser.getCurrentLocation();

  SmallVector<OpAsmParser::OperandType, 4> entryArgs;
  SmallVector<NamedAttrList, 4> argAttrs;
  SmallVector<NamedAttrList, 4> resultAttrs;
  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 4> resultTypes;
  SmallVector<Attribute> parameters;
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  FlatSymbolRefAttr kindAttr;
  if (modKind == GenMod) {
    if (parser.parseComma() ||
        parser.parseAttribute(kindAttr, "generatorKind", result.attributes)) {
      return failure();
    }
  }

  // Parse the function signature.
  bool isVariadic = false;
  SmallVector<Attribute> resultNames;
  if (parseOptionalParameters(parser, parameters) ||
      module_like_impl::parseModuleFunctionSignature(
          parser, entryArgs, argTypes, argAttrs, isVariadic, resultTypes,
          resultAttrs, resultNames) ||
      // If function attributes are present, parse them.
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Record the argument and result types as an attribute.  This is necessary
  // for external modules.
  auto type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

  auto *context = result.getContext();

  if (hasAttribute("resultNames", result.attributes) ||
      hasAttribute("parameters", result.attributes)) {
    parser.emitError(
        loc, "explicit `resultNames` / `parameters` attributes not allowed");
    return failure();
  }

  // Use the argument and result names if not already specified.
  SmallVector<Attribute> argNames;
  if (!entryArgs.empty()) {
    for (auto &arg : entryArgs)
      argNames.push_back(module_like_impl::getPortNameAttr(context, arg.name));
  } else if (!argTypes.empty()) {
    // The parser returns empty names in a special way.
    argNames.assign(argTypes.size(), StringAttr::get(context, ""));
  }

  // An explicit `argNames` attribute overrides the MLIR names.  This is how
  // we represent port names that aren't valid MLIR identifiers.  Result and
  // parameter names are printed quoted when they aren't valid identifiers, so
  // they don't need this affordance.
  if (!hasAttribute("argNames", result.attributes))
    result.addAttribute("argNames", ArrayAttr::get(context, argNames));
  result.addAttribute("resultNames", ArrayAttr::get(context, resultNames));
  result.addAttribute("parameters", ArrayAttr::get(context, parameters));

  assert(argAttrs.size() == argTypes.size());
  assert(resultAttrs.size() == resultTypes.size());

  // Add the attributes to the function arguments.
  addArgAndResultAttrs(builder, result, argAttrs, resultAttrs);

  // Parse the optional function body.
  auto *body = result.addRegion();
  if (modKind == PlainMod) {
    if (parser.parseRegion(*body, entryArgs,
                           entryArgs.empty() ? ArrayRef<Type>() : argTypes))
      return failure();

    HWModuleOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  }
  return success();
}

static ParseResult parseHWModuleExternOp(OpAsmParser &parser,
                                         OperationState &result) {
  return parseHWModuleOp(parser, result, ExternMod);
}

static ParseResult parseHWModuleGeneratedOp(OpAsmParser &parser,
                                            OperationState &result) {
  return parseHWModuleOp(parser, result, GenMod);
}

FunctionType getHWModuleOpType(Operation *op) {
  auto typeAttr = op->getAttrOfType<TypeAttr>(HWModuleOp::getTypeAttrName());
  return typeAttr.getValue().cast<FunctionType>();
}

/// Print a parameter list for a module or instance.
static void printParameterList(ArrayAttr parameters, OpAsmPrinter &p) {
  if (parameters.empty())
    return;

  p << '<';
  llvm::interleaveComma(parameters, p, [&](Attribute param) {
    auto paramAttr = param.cast<ParamDeclAttr>();
    p << paramAttr.getName().getValue() << ": " << paramAttr.getType();
    if (auto value = paramAttr.getValue()) {
      p << " = ";
      p.printAttributeWithoutType(value);
    }
  });
  p << '>';
}

static void printModuleOp(OpAsmPrinter &p, Operation *op,
                          ExternModKind modKind) {
  using namespace mlir::function_like_impl;

  FunctionType fnType = getHWModuleOpType(op);
  auto argTypes = fnType.getInputs();
  auto resultTypes = fnType.getResults();

  // Print the operation and the function name.
  p << ' ';
  p.printSymbolName(SymbolTable::getSymbolName(op).getValue());
  if (modKind == GenMod) {
    p << ", ";
    p.printSymbolName(cast<HWModuleGeneratedOp>(op).generatorKind());
  }

  // Print the parameter list if present.
  printParameterList(op->getAttrOfType<ArrayAttr>("parameters"), p);

  bool needArgNamesAttr = false;
  module_like_impl::printModuleSignature(p, op, argTypes, /*isVariadic=*/false,
                                         resultTypes, needArgNamesAttr);

  SmallVector<StringRef, 3> omittedAttrs;
  if (modKind == GenMod)
    omittedAttrs.push_back("generatorKind");
  if (!needArgNamesAttr)
    omittedAttrs.push_back("argNames");
  omittedAttrs.push_back("resultNames");
  omittedAttrs.push_back("parameters");

  printFunctionAttributes(p, op, argTypes.size(), resultTypes.size(),
                          omittedAttrs);
}

static void printHWModuleExternOp(OpAsmPrinter &p, HWModuleExternOp op) {
  printModuleOp(p, op, ExternMod);
}
static void printHWModuleGeneratedOp(OpAsmPrinter &p, HWModuleGeneratedOp op) {
  printModuleOp(p, op, GenMod);
}

static void printHWModuleOp(OpAsmPrinter &p, HWModuleOp op) {
  printModuleOp(p, op, PlainMod);

  // Print the body if this is not an external function.
  Region &body = op.getBody();
  if (!body.empty())
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
}

static LogicalResult verifyModuleCommon(Operation *module) {
  assert(isAnyModule(module) &&
         "verifier hook should only be called on modules");

  auto moduleType = getModuleType(module);
  auto argNames = module->getAttrOfType<ArrayAttr>("argNames");
  auto resultNames = module->getAttrOfType<ArrayAttr>("resultNames");
  if (argNames.size() != moduleType.getNumInputs())
    return module->emitOpError("incorrect number of argument names");
  if (resultNames.size() != moduleType.getNumResults())
    return module->emitOpError("incorrect number of result names");

  // Check parameter default values are sensible.
  for (auto param : module->getAttrOfType<ArrayAttr>("parameters")) {
    auto paramAttr = param.cast<ParamDeclAttr>();

    // Default values are allowed to be missing.
    auto value = paramAttr.getValue();
    if (!value)
      continue;

    if (value.getType() != paramAttr.getType().getValue())
      return module->emitOpError("parameter ")
             << paramAttr << " should have type "
             << paramAttr.getType().getValue() << "; has type "
             << value.getType();

    // Verify that this is a valid parameter value, disallowing parameter
    // references.  We could allow parameters to refer to each other in the
    // future with lexical ordering if there is a need.
    if (failed(checkParameterInContext(value, module, module,
                                       /*disallowParamRefs=*/true)))
      return failure();
  }
  return success();
}

static LogicalResult verifyHWModuleOp(HWModuleOp op) {
  return verifyModuleCommon(op);
}

static LogicalResult verifyHWModuleExternOp(HWModuleExternOp op) {
  return verifyModuleCommon(op);
}

/// Lookup the generator for the symbol.  This returns null on
/// invalid IR.
Operation *HWModuleGeneratedOp::getGeneratorKindOp() {
  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  return topLevelModuleOp.lookupSymbol(generatorKind());
}

static LogicalResult verifyHWModuleGeneratedOp(HWModuleGeneratedOp op) {
  if (failed(verifyModuleCommon(op)))
    return failure();

  auto referencedKind = op.getGeneratorKindOp();
  if (referencedKind == nullptr)
    return op.emitError("Cannot find generator definition '")
           << op.generatorKind() << "'";

  if (!isa<HWGeneratorSchemaOp>(referencedKind))
    return op.emitError("Symbol resolved to '")
           << referencedKind->getName()
           << "' which is not a HWGeneratorSchemaOp";

  auto referencedKindOp = dyn_cast<HWGeneratorSchemaOp>(referencedKind);
  auto paramRef = referencedKindOp.requiredAttrs();
  auto dict = op->getAttrDictionary();
  for (auto str : paramRef) {
    auto strAttr = str.dyn_cast<StringAttr>();
    if (!strAttr)
      return op.emitError("Unknown attribute type, expected a string");
    if (!dict.get(strAttr.getValue()))
      return op.emitError("Missing attribute '") << strAttr.getValue() << "'";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

/// Create a instance that refers to a known module.
void InstanceOp::build(OpBuilder &builder, OperationState &result,
                       Operation *module, StringAttr name,
                       ArrayRef<Value> inputs, ArrayAttr parameters,
                       StringAttr sym_name) {
  assert(isAnyModule(module) && "Can only reference a module");

  if (!parameters)
    parameters = builder.getArrayAttr({});

  FunctionType modType = getModuleType(module);
  build(builder, result, modType.getResults(), name,
        FlatSymbolRefAttr::get(SymbolTable::getSymbolName(module)), inputs,
        module->getAttrOfType<ArrayAttr>("argNames"),
        module->getAttrOfType<ArrayAttr>("resultNames"), parameters, sym_name);
}

/// Lookup the module or extmodule for the symbol.  This returns null on
/// invalid IR.
Operation *InstanceOp::getReferencedModule(const SymbolCache *cache) {
  if (cache)
    if (auto *result = cache->getDefinition(moduleNameAttr()))
      return result;

  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  return topLevelModuleOp.lookupSymbol(moduleName());
}

LogicalResult InstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto *module = symbolTable.lookupNearestSymbolFrom(*this, moduleNameAttr());
  if (module == nullptr)
    return emitError("Cannot find module definition '") << moduleName() << "'";

  // It must be some sort of module.
  if (!isAnyModule(module))
    return emitError("symbol reference '")
           << moduleName() << "' isn't a module";

  // Check that input and result types are consistent with the referenced
  // module.
  // Emit an error message on the instance, with a note indicating which module
  // is being referenced.
  auto emitError =
      [&](std::function<void(InFlightDiagnostic & diag)> fn) -> LogicalResult {
    auto diag = emitOpError();
    fn(diag);
    diag.attachNote(module->getLoc()) << "module declared here";
    return failure();
  };

  // Make sure our port and result names match.
  ArrayAttr argNames = argNamesAttr();
  ArrayAttr modArgNames = module->getAttrOfType<ArrayAttr>("argNames");

  // Check operand types first.
  auto numOperands = getOperation()->getNumOperands();
  auto expectedOperandTypes = getModuleType(module).getInputs();

  if (expectedOperandTypes.size() != numOperands)
    return emitError([&](auto &diag) {
      diag << "has a wrong number of operands; expected "
           << expectedOperandTypes.size() << " but got " << numOperands;
    });

  if (argNames.size() != numOperands)
    return emitError([&](auto &diag) {
      diag << "has a wrong number of input port names; expected " << numOperands
           << " but got " << argNames.size();
    });

  for (size_t i = 0; i != numOperands; ++i) {
    auto expectedType = expectedOperandTypes[i];
    auto operandType = getOperand(i).getType();
    if (operandType != expectedType)
      return emitError([&](auto &diag) {
        diag << "operand type #" << i << " must be " << expectedType
             << ", but got " << operandType;
      });

    if (argNames[i] != modArgNames[i])
      return emitError([&](auto &diag) {
        diag << "input label #" << i << " must be " << modArgNames[i]
             << ", but got " << argNames[i];
      });
  }

  // Check result types and labels.
  auto numResults = getOperation()->getNumResults();
  auto expectedResultTypes = getModuleType(module).getResults();
  ArrayAttr resultNames = resultNamesAttr();
  ArrayAttr modResultNames = module->getAttrOfType<ArrayAttr>("resultNames");

  if (expectedResultTypes.size() != numResults)
    return emitError([&](auto &diag) {
      diag << "has a wrong number of results; expected "
           << expectedResultTypes.size() << " but got " << numResults;
    });
  if (resultNames.size() != numResults)
    return emitError([&](auto &diag) {
      diag << "has a wrong number of results port labels; expected "
           << numResults << " but got " << resultNames.size();
    });

  for (size_t i = 0; i != numResults; ++i) {
    auto expectedType = expectedResultTypes[i];
    auto resultType = getResult(i).getType();
    if (resultType != expectedType)
      return emitError([&](auto &diag) {
        diag << "result type #" << i << " must be " << expectedType
             << ", but got " << resultType;
      });

    if (resultNames[i] != modResultNames[i])
      return emitError([&](auto &diag) {
        diag << "input label #" << i << " must be " << modResultNames[i]
             << ", but got " << resultNames[i];
      });
  }

  // Check parameters match up.
  ArrayAttr parameters = this->parameters();
  ArrayAttr modParameters = module->getAttrOfType<ArrayAttr>("parameters");
  auto numParameters = parameters.size();
  if (numParameters != modParameters.size())
    return emitError([&](auto &diag) {
      diag << "expected " << modParameters.size() << " parameters but had "
           << numParameters;
    });

  for (size_t i = 0; i != numParameters; ++i) {
    auto param = parameters[i].cast<ParamDeclAttr>();
    auto modParam = modParameters[i].cast<ParamDeclAttr>();

    auto paramName = param.getName();
    if (paramName != modParam.getName())
      return emitError([&](auto &diag) {
        diag << "parameter #" << i << " should have name " << modParam.getName()
             << " but has name " << paramName;
      });

    if (param.getType() != modParam.getType())
      return emitError([&](auto &diag) {
        diag << "parameter " << paramName << " should have type "
             << modParam.getType() << " but has type " << param.getType();
      });

    // All instance parameters must have a value.  Specify the same value as
    // a module's default value if you want the default.
    if (!param.getValue())
      return emitOpError("parameter ") << paramName << " must have a value";
  }

  return success();
}

LogicalResult InstanceOp::verifyCustom() {
  // Check that all the parameter values specified to the instance are
  // structurally valid.
  for (auto param : parameters()) {
    auto paramAttr = param.cast<ParamDeclAttr>();
    auto value = paramAttr.getValue();
    assert(value && "SymbolUses verifier should already check this exists");

    if (value.getType() != paramAttr.getType().getValue())
      return emitOpError("parameter ") << paramAttr << " should have type "
                                       << paramAttr.getType().getValue()
                                       << "; has type " << value.getType();

    if (failed(checkParameterInContext(
            value, (*this)->getParentOfType<HWModuleOp>(), *this)))
      return failure();
  }
  return success();
}

static ParseResult parseInstanceOp(OpAsmParser &parser,
                                   OperationState &result) {
  StringAttr instanceNameAttr;
  StringAttr sym_nameAttr;
  FlatSymbolRefAttr moduleNameAttr;
  SmallVector<OpAsmParser::OperandType, 4> inputsOperands;
  SmallVector<Type> inputsTypes;
  SmallVector<Type> allResultTypes;
  SmallVector<Attribute> argNames, resultNames, parameters;
  auto noneType = parser.getBuilder().getType<NoneType>();

  if (parser.parseAttribute(instanceNameAttr, noneType, "instanceName",
                            result.attributes))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("sym"))) {
    // Parsing an optional symbol name doesn't fail, so no need to check the
    // result.
    (void)parser.parseOptionalSymbolName(sym_nameAttr, "sym_name",
                                         result.attributes);
  }

  auto parseInputPort = [&]() -> ParseResult {
    argNames.push_back(module_like_impl::parsePortName(parser));
    if (!argNames.back())
      return failure();
    inputsOperands.push_back({});
    inputsTypes.push_back({});
    return failure(parser.parseColon() ||
                   parser.parseOperand(inputsOperands.back()) ||
                   parser.parseColon() || parser.parseType(inputsTypes.back()));
  };

  auto parseResultPort = [&]() -> ParseResult {
    resultNames.push_back(module_like_impl::parsePortName(parser));
    if (!resultNames.back())
      return failure();
    allResultTypes.push_back({});
    return parser.parseColonType(allResultTypes.back());
  };

  llvm::SMLoc parametersLoc, inputsOperandsLoc;
  if (parser.parseAttribute(moduleNameAttr, noneType, "moduleName",
                            result.attributes) ||
      parser.getCurrentLocation(&parametersLoc) ||
      parseOptionalParameters(parser, parameters) ||
      parser.getCurrentLocation(&inputsOperandsLoc) ||
      parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                     parseInputPort) ||
      parser.resolveOperands(inputsOperands, inputsTypes, inputsOperandsLoc,
                             result.operands) ||
      parser.parseArrow() ||
      parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                     parseResultPort) ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  result.addAttribute("argNames", parser.getBuilder().getArrayAttr(argNames));
  result.addAttribute("resultNames",
                      parser.getBuilder().getArrayAttr(resultNames));
  result.addAttribute("parameters",
                      parser.getBuilder().getArrayAttr(parameters));
  result.addTypes(allResultTypes);
  return success();
}

static void printInstanceOp(OpAsmPrinter &p, InstanceOp op) {
  ModulePortInfo portInfo = getModulePortInfo(op);
  size_t nextInputPort = 0, nextOutputPort = 0;

  auto printPortName = [&](size_t &nextPort, SmallVector<PortInfo> &portList) {
    // Allow printing mangled instances.
    if (nextPort >= portList.size()) {
      p << "<corrupt port>: ";
      return;
    }

    module_like_impl::printPortName(portList[nextPort++].name, p.getStream());
    p << ": ";
  };

  p << ' ';
  p.printAttributeWithoutType(op.instanceNameAttr());
  if (auto attr = op.sym_nameAttr()) {
    p << " sym ";
    p.printSymbolName(attr.getValue());
  }
  p << ' ';
  p.printAttributeWithoutType(op.moduleNameAttr());
  printParameterList(op.parameters(), p);
  p << '(';
  llvm::interleaveComma(op.inputs(), p, [&](Value op) {
    printPortName(nextInputPort, portInfo.inputs);
    p << op << ": " << op.getType();
  });
  p << ") -> (";
  llvm::interleaveComma(op.getResults(), p, [&](Value res) {
    printPortName(nextOutputPort, portInfo.outputs);
    p << res.getType();
  });
  p << ')';
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{
                              "instanceName", "sym_name", "moduleName",
                              "argNames", "resultNames", "parameters"});
}

/// Return the name of the specified input port or null if it cannot be
/// determined.
StringAttr InstanceOp::getArgumentName(size_t idx) {
  auto names = argNames();
  // Tolerate malformed IR here to enable debug printing etc.
  if (names && idx < names.size())
    return names[idx].cast<StringAttr>();
  return StringAttr();
}

/// Return the name of the specified result or null if it cannot be
/// determined.
StringAttr InstanceOp::getResultName(size_t idx) {
  auto names = resultNames();
  // Tolerate malformed IR here to enable debug printing etc.
  if (names && idx < names.size())
    return names[idx].cast<StringAttr>();
  return StringAttr();
}

/// Change the name of the specified input port.
void InstanceOp::setArgumentName(size_t i, StringAttr name) {
  auto names = argNames();
  SmallVector<Attribute> newNames(names.begin(), names.end());
  if (newNames[i] == name)
    return;
  newNames[i] = name;
  setArgumentNames(ArrayAttr::get(getContext(), names));
}

/// Change the name of the specified output port.
void InstanceOp::setResultName(size_t i, StringAttr name) {
  auto names = resultNames();
  SmallVector<Attribute> newNames(names.begin(), names.end());
  if (newNames[i] == name)
    return;
  newNames[i] = name;
  setResultNames(ArrayAttr::get(getContext(), names));
}

/// Suggest a name for each result value based on the saved result names
/// attribute.
void InstanceOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // Provide default names for instance results.
  std::string name = instanceName().str() + ".";
  size_t baseNameLen = name.size();

  for (size_t i = 0, e = getNumResults(); i != e; ++i) {
    auto resName = getResultName(i);
    name.resize(baseNameLen);
    if (resName && !resName.getValue().empty())
      name += resName.getValue().str();
    else
      name += std::to_string(i);
    setNameFn(getResult(i), name);
  }
}

//===----------------------------------------------------------------------===//
// HWOutputOp
//===----------------------------------------------------------------------===//

/// Verify that the num of operands and types fit the declared results.
static LogicalResult verifyOutputOp(OutputOp *op) {
  // Check that the we (hw.output) have the same number of operands as our
  // region has results.
  auto opParent = (*op)->getParentOp();
  FunctionType modType = getModuleType(opParent);
  ArrayRef<Type> modResults = modType.getResults();
  OperandRange outputValues = op->getOperands();
  if (modResults.size() != outputValues.size()) {
    op->emitOpError("must have same number of operands as region results.");
    return failure();
  }

  // Check that the types of our operands and the region's results match.
  for (size_t i = 0, e = modResults.size(); i < e; ++i) {
    if (modResults[i] != outputValues[i].getType()) {
      op->emitOpError("output types must match module. In "
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
  unsigned idxWidth = llvm::Log2_64_Ceil(arrType.getSize());
  idxType = IntegerType::get(p.getBuilder().getContext(), idxWidth);
  return success();
}

static void printSliceTypes(OpAsmPrinter &p, Operation *, Type srcType,
                            Type idxType) {
  p.printType(srcType);
}

static ParseResult parseArrayCreateOp(OpAsmParser &parser,
                                      OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
  llvm::SmallVector<OpAsmParser::OperandType, 16> operands;
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

static void printArrayCreateOp(OpAsmPrinter &p, ArrayCreateOp op) {
  p << " ";
  p.printOperands(op.inputs());
  p << " : " << op.inputs()[0].getType();
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
    resultSize += arrTy.getSize();
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
  ArrayType arrayTy = values[0].getType().cast<ArrayType>();
  Type elemTy = arrayTy.getElementType();
  assert(llvm::all_of(values,
                      [elemTy](Value v) -> bool {
                        return v.getType().isa<ArrayType>() &&
                               v.getType().cast<ArrayType>().getElementType() ==
                                   elemTy;
                      }) &&
         "All values must be of ArrayType with the same element type.");

  uint64_t resultSize = 0;
  for (Value val : values)
    resultSize += val.getType().cast<ArrayType>().getSize();
  build(b, state, ArrayType::get(elemTy, resultSize), values);
}

//===----------------------------------------------------------------------===//
// StructCreateOp
//===----------------------------------------------------------------------===//

static ParseResult parseStructCreateOp(OpAsmParser &parser,
                                       OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
  llvm::SmallVector<OpAsmParser::OperandType, 4> operands;
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

static void printStructCreateOp(OpAsmPrinter &printer, hw::StructCreateOp op) {
  printer << " (";
  printer.printOperands(op.input());
  printer << ")";
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : " << op.getType();
}

//===----------------------------------------------------------------------===//
// StructExplodeOp
//===----------------------------------------------------------------------===//

static ParseResult parseStructExplodeOp(OpAsmParser &parser,
                                        OperationState &result) {
  OpAsmParser::OperandType operand;
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

static void printStructExplodeOp(OpAsmPrinter &printer,
                                 hw::StructExplodeOp op) {
  printer << " ";
  printer.printOperand(op.input());
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : " << op.input().getType();
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

/// Use the same parser for both struct_extract and union_extract since the
/// syntax is identical.
template <typename AggregateType>
static ParseResult parseExtractOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType operand;
  StringAttr fieldName;
  Type declType;

  if (parser.parseOperand(operand) || parser.parseLSquare() ||
      parser.parseAttribute(fieldName, "field", result.attributes) ||
      parser.parseRSquare() ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(declType))
    return failure();
  auto aggType = type_dyn_cast<AggregateType>(declType);
  if (!aggType)
    return parser.emitError(parser.getNameLoc(),
                            "invalid kind of type specified");

  Type resultType = aggType.getFieldType(fieldName.getValue());
  if (!resultType) {
    parser.emitError(parser.getNameLoc(), "invalid field name specified");
    return failure();
  }
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
  printer.printOperand(op.input());
  printer << "[\"" << op.field() << "\"]";
  printer.printOptionalAttrDict(op->getAttrs(), {"field"});
  printer << " : " << op.input().getType();
}

static ParseResult parseStructExtractOp(OpAsmParser &parser,
                                        OperationState &result) {
  return parseExtractOp<StructType>(parser, result);
}

static void printStructExtractOp(OpAsmPrinter &printer,
                                 hw::StructExtractOp op) {
  printExtractOp(printer, op);
}

void StructExtractOp::build(OpBuilder &builder, OperationState &odsState,
                            Value input, StructType::FieldInfo field) {
  build(builder, odsState, field.type, input, field.name);
}

//===----------------------------------------------------------------------===//
// StructInjectOp
//===----------------------------------------------------------------------===//

static ParseResult parseStructInjectOp(OpAsmParser &parser,
                                       OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
  OpAsmParser::OperandType operand, val;
  StringAttr fieldName;
  Type declType;

  if (parser.parseOperand(operand) || parser.parseLSquare() ||
      parser.parseAttribute(fieldName, "field", result.attributes) ||
      parser.parseRSquare() || parser.parseComma() ||
      parser.parseOperand(val) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(declType))
    return failure();
  auto structType = type_dyn_cast<StructType>(declType);
  if (!structType)
    return parser.emitError(inputOperandsLoc, "invalid kind of type specified");

  Type resultType = structType.getFieldType(fieldName.getValue());
  if (!resultType) {
    parser.emitError(inputOperandsLoc, "invalid field name specified");
    return failure();
  }
  result.addTypes(declType);

  if (parser.resolveOperands({operand, val}, {declType, resultType},
                             inputOperandsLoc, result.operands))
    return failure();
  return success();
}

static void printStructInjectOp(OpAsmPrinter &printer, hw::StructInjectOp op) {
  printer << " ";
  printer.printOperand(op.input());
  printer << "[\"" << op.field() << "\"], ";
  printer.printOperand(op.newValue());
  printer.printOptionalAttrDict(op->getAttrs(), {"field"});
  printer << " : " << op.input().getType();
}

//===----------------------------------------------------------------------===//
// UnionCreateOp
//===----------------------------------------------------------------------===//

static ParseResult parseUnionCreateOp(OpAsmParser &parser,
                                      OperationState &result) {
  Type declOrAliasType;
  StringAttr field;
  OpAsmParser::OperandType input;
  llvm::SMLoc fieldLoc = parser.getCurrentLocation();

  if (parser.parseAttribute(field, "field", result.attributes) ||
      parser.parseComma() || parser.parseOperand(input) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(declOrAliasType))
    return failure();

  auto declType = type_dyn_cast<UnionType>(declOrAliasType);
  if (!declType)
    return parser.emitError(parser.getNameLoc(),
                            "expected !hw.union type or alias");

  Type inputType = declType.getFieldType(field.getValue());
  if (!inputType) {
    parser.emitError(fieldLoc, "cannot find union field '")
        << field.getValue() << '\'';
    return failure();
  }

  if (parser.resolveOperand(input, inputType, result.operands))
    return failure();
  result.addTypes({declOrAliasType});
  return success();
}

static void printUnionCreateOp(OpAsmPrinter &printer, hw::UnionCreateOp op) {
  printer << " \"" << op.field() << "\", ";
  printer.printOperand(op.input());
  printer.printOptionalAttrDict(op->getAttrs(), {"field"});
  printer << " : " << op.getType();
}

//===----------------------------------------------------------------------===//
// UnionExtractOp
//===----------------------------------------------------------------------===//

static ParseResult parseUnionExtractOp(OpAsmParser &parser,
                                       OperationState &result) {
  return parseExtractOp<UnionType>(parser, result);
}

static void printUnionExtractOp(OpAsmPrinter &printer, hw::UnionExtractOp op) {
  printExtractOp(printer, op);
}

//===----------------------------------------------------------------------===//
// ArrayGetOp
//===----------------------------------------------------------------------===//

void ArrayGetOp::build(OpBuilder &builder, OperationState &result, Value input,
                       Value index) {
  auto resultType = type_cast<ArrayType>(input.getType()).getElementType();
  build(builder, result, resultType, input, index);
}

//===----------------------------------------------------------------------===//
// TypedeclOp
//===----------------------------------------------------------------------===//

StringRef TypedeclOp::getPreferredName() {
  return verilogName().getValueOr(getName());
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/HW/HW.cpp.inc"
