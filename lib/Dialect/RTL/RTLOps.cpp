//===- RTLOps.cpp - Implement the RTL operations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the RTL ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTL/RTLOps.h"
#include "circt/Dialect/RTL/RTLTypes.h"
#include "circt/Dialect/RTL/RTLVisitors.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace circt;
using namespace rtl;

//===----------------------------------------------------------------------===//
// RTLModuleOp
//===----------------------------------------------------------------------===/

static void buildModule(OpBuilder &builder, OperationState &result,
                        StringAttr name, ArrayRef<ModulePortInfo> ports) {
  using namespace mlir::impl;

  // Add an attribute for the name.
  result.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), name);

  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 4> resultTypes;
  for (auto elt : ports) {
    if (elt.isOutput())
      resultTypes.push_back(elt.type);
    else
      argTypes.push_back(elt.type);
  }

  // Record the argument and result types as an attribute.
  auto type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

  // Record the names of the arguments if present.
  SmallString<8> attrNameBuf;
  SmallString<8> attrDirBuf;
  for (const ModulePortInfo &port : ports) {
    SmallVector<NamedAttribute, 2> argAttrs;
    if (!port.name.getValue().empty())
      argAttrs.push_back(
          NamedAttribute(builder.getIdentifier("rtl.name"), port.name));

    StringRef attrName = port.isOutput()
                             ? getResultAttrName(port.argNum, attrNameBuf)
                             : getArgAttrName(port.argNum, attrNameBuf);
    result.addAttribute(attrName, builder.getDictionaryAttr(argAttrs));
  }
  result.addRegion();
}

void RTLModuleOp::build(OpBuilder &builder, OperationState &result,
                        StringAttr name, ArrayRef<ModulePortInfo> ports) {
  buildModule(builder, result, name, ports);

  // Create a region and a block for the body.
  auto *bodyRegion = result.regions[0].get();
  Block *body = new Block();
  bodyRegion->push_back(body);

  // Add arguments to the body block.
  for (auto elt : ports)
    if (!elt.isOutput())
      body->addArgument(elt.type);

  RTLModuleOp::ensureTerminator(*bodyRegion, builder, result.location);
}

/// Return the name to use for the Verilog module that we're referencing
/// here.  This is typically the symbol, but can be overridden with the
/// verilogName attribute.
StringRef RTLExternModuleOp::getVerilogModuleName() {
  if (auto vname = verilogName())
    return vname.getValue();
  return getName();
}

void RTLExternModuleOp::build(OpBuilder &builder, OperationState &result,
                              StringAttr name, ArrayRef<ModulePortInfo> ports,
                              StringRef verilogName) {
  buildModule(builder, result, name, ports);

  if (!verilogName.empty())
    result.addAttribute("verilogName", builder.getStringAttr(verilogName));
}

FunctionType rtl::getModuleType(Operation *op) {
  auto typeAttr = op->getAttrOfType<TypeAttr>(RTLModuleOp::getTypeAttrName());
  return typeAttr.getValue().cast<FunctionType>();
}

StringAttr rtl::getRTLNameAttr(ArrayRef<NamedAttribute> attrs) {
  for (auto &argAttr : attrs) {
    if (argAttr.first != "rtl.name")
      continue;
    return argAttr.second.dyn_cast<StringAttr>();
  }
  return StringAttr();
}

void rtl::getModulePortInfo(Operation *op,
                            SmallVectorImpl<ModulePortInfo> &results) {
  auto argTypes = getModuleType(op).getInputs();

  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    auto argAttrs = ::mlir::impl::getArgAttrs(op, i);
    bool isInOut = false;
    auto type = argTypes[i];

    if (auto inout = type.dyn_cast<InOutType>()) {
      isInOut = true;
      type = inout.getElementType();
    }

    results.push_back({getRTLNameAttr(argAttrs),
                       isInOut ? PortDirection::INOUT : PortDirection::INPUT,
                       type, i});
  }

  auto resultTypes = getModuleType(op).getResults();
  for (unsigned i = 0, e = resultTypes.size(); i < e; ++i) {
    auto argAttrs = ::mlir::impl::getResultAttrs(op, i);
    results.push_back(
        {getRTLNameAttr(argAttrs), PortDirection::OUTPUT, resultTypes[i], i});
  }
}

/// Parse a function result list.
///
///   function-result-list ::= function-result-list-parens
///   function-result-list-parens ::= `(` `)`
///                                 | `(` function-result-list-no-parens `)`
///   function-result-list-no-parens ::= function-result (`,` function-result)*
///   function-result ::= (percent-identifier `:`) type attribute-dict?
///
static ParseResult
parseFunctionResultList(OpAsmParser &parser, SmallVectorImpl<Type> &resultTypes,
                        SmallVectorImpl<NamedAttrList> &resultAttrs) {
  if (parser.parseLParen())
    return failure();

  // Special case for an empty set of parens.
  if (succeeded(parser.parseOptionalRParen()))
    return success();

  // Parse individual function results.
  do {
    resultTypes.emplace_back();
    resultAttrs.emplace_back();

    OpAsmParser::OperandType operandName;
    auto namePresent = parser.parseOptionalOperand(operandName);
    StringRef implicitName;
    if (namePresent.hasValue()) {
      if (namePresent.getValue() || parser.parseColon())
        return failure();

      // If the name was specified, then we will use it.
      implicitName = operandName.name.drop_front();
    }

    if (parser.parseType(resultTypes.back()) ||
        parser.parseOptionalAttrDict(resultAttrs.back()))
      return failure();

    // If we have an implicit name and no explicit rtl.name attribute, then use
    // the implicit name as the rtl.name attribute.
    if (!implicitName.empty() && !getRTLNameAttr(resultAttrs.back())) {
      auto nameAttr = parser.getBuilder().getStringAttr(implicitName);
      resultAttrs.back().append("rtl.name", nameAttr);
    }
  } while (succeeded(parser.parseOptionalComma()));
  return parser.parseRParen();
}

/// This is a variant of mlor::parseFunctionSignature that allows names on
/// result arguments.
static ParseResult parseModuleFunctionSignature(
    OpAsmParser &parser, bool allowVariadic,
    SmallVectorImpl<OpAsmParser::OperandType> &argNames,
    SmallVectorImpl<Type> &argTypes, SmallVectorImpl<NamedAttrList> &argAttrs,
    bool &isVariadic, SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<NamedAttrList> &resultAttrs) {
  bool allowArgAttrs = true;
  if (impl::parseFunctionArgumentList(parser, allowArgAttrs, allowVariadic,
                                      argNames, argTypes, argAttrs, isVariadic))
    return failure();
  if (succeeded(parser.parseOptionalArrow()))
    return parseFunctionResultList(parser, resultTypes, resultAttrs);
  return success();
}

static ParseResult parseRTLModuleOp(OpAsmParser &parser, OperationState &result,
                                    bool isExtModule = false) {
  using namespace mlir::impl;

  SmallVector<OpAsmParser::OperandType, 4> entryArgs;
  SmallVector<NamedAttrList, 4> argAttrs;
  SmallVector<NamedAttrList, 4> resultAttrs;
  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 4> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  bool isVariadic = false;

  if (parseModuleFunctionSignature(parser, /*allowVariadic=*/false, entryArgs,
                                   argTypes, argAttrs, isVariadic, resultTypes,
                                   resultAttrs))
    return failure();

  // Record the argument and result types as an attribute.  This is necessary
  // for external modules.
  auto type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

  // If function attributes are present, parse them.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  assert(argAttrs.size() == argTypes.size());
  assert(resultAttrs.size() == resultTypes.size());

  auto *context = result.getContext();

  // Postprocess each of the arguments.  If there was no 'rtl.name'
  // attribute, and if the argument name was non-numeric, then add the
  // rtl.name attribute with the textual name from the IR.  The name in the
  // text file is a load-bearing part of the IR, but we don't want the
  // verbosity in dumps of including it explicitly in the attribute
  // dictionary.
  for (size_t i = 0, e = argAttrs.size(); i != e; ++i) {
    auto &attrs = argAttrs[i];

    // If an explicit name attribute was present, don't add the implicit one.
    bool hasNameAttr = false;
    for (auto &elt : attrs)
      if (elt.first.str() == "rtl.name")
        hasNameAttr = true;
    if (hasNameAttr || entryArgs.empty())
      continue;

    auto &arg = entryArgs[i];

    // The name of an argument is of the form "%42" or "%id", and since
    // parsing succeeded, we know it always has one character.
    assert(arg.name.size() > 1 && arg.name[0] == '%' && "Unknown MLIR name");
    if (isdigit(arg.name[1]))
      continue;

    auto nameAttr = StringAttr::get(arg.name.drop_front(), context);
    attrs.push_back({Identifier::get("rtl.name", context), nameAttr});
  }

  // Add the attributes to the function arguments.
  addArgAndResultAttrs(builder, result, argAttrs, resultAttrs);

  // Parse the optional function body.
  auto *body = result.addRegion();
  if (!isExtModule) {
    if (parser.parseRegion(*body, entryArgs,
                           entryArgs.empty() ? ArrayRef<Type>() : argTypes))
      return failure();

    RTLModuleOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  }
  return success();
}

static ParseResult parseRTLExternModuleOp(OpAsmParser &parser,
                                          OperationState &result) {
  return parseRTLModuleOp(parser, result, /*isExtModule:*/ true);
}

FunctionType getRTLModuleOpType(Operation *op) {
  auto typeAttr = op->getAttrOfType<TypeAttr>(RTLModuleOp::getTypeAttrName());
  return typeAttr.getValue().cast<FunctionType>();
}

static void printModuleSignature(OpAsmPrinter &p, Operation *op,
                                 ArrayRef<Type> argTypes, bool isVariadic,
                                 ArrayRef<Type> resultTypes) {
  Region &body = op->getRegion(0);
  bool isExternal = body.empty();

  p << '(';
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    if (i > 0)
      p << ", ";

    Value argumentValue;
    if (!isExternal) {
      argumentValue = body.front().getArgument(i);
      p.printOperand(argumentValue);
      p << ": ";
    }

    p.printType(argTypes[i]);

    auto argAttrs = ::mlir::impl::getArgAttrs(op, i);

    // If the argument has the rtl.name attribute, and if it was used by
    // the printer exactly (not name mangled with a suffix etc) then we can
    // omit the rtl.name attribute from the argument attribute dictionary.
    ArrayRef<StringRef> elidedAttrs;
    StringRef tmp;
    if (argumentValue) {
      if (auto nameAttr = getRTLNameAttr(argAttrs)) {

        // Check to make sure the asmprinter is printing it correctly.
        SmallString<32> resultNameStr;
        llvm::raw_svector_ostream tmpStream(resultNameStr);
        p.printOperand(argumentValue, tmpStream);

        // If the name is the same as we would otherwise use, then we're good!
        if (tmpStream.str().drop_front() == nameAttr.getValue()) {
          tmp = "rtl.name";
          elidedAttrs = tmp;
        }
      }
    }
    p.printOptionalAttrDict(argAttrs, elidedAttrs);
  }

  if (isVariadic) {
    if (!argTypes.empty())
      p << ", ";
    p << "...";
  }

  p << ')';

  // We print result types specially since we support named arguments.
  if (!resultTypes.empty()) {
    auto &os = p.getStream();
    os << " -> (";
    for (size_t i = 0, e = resultTypes.size(); i < e; ++i) {
      if (i != 0)
        os << ", ";
      auto resultAttrs = ::mlir::impl::getResultAttrs(op, i);
      StringAttr name = getRTLNameAttr(resultAttrs);
      if (name)
        os << '%' << name.getValue() << ": ";

      p.printType(resultTypes[i]);
      p.printOptionalAttrDict(resultAttrs, {"rtl.name"});
    }
    os << ')';
  }
}

static void printRTLModuleOp(OpAsmPrinter &p, Operation *op) {
  using namespace mlir::impl;

  FunctionType fnType = getRTLModuleOpType(op);
  auto argTypes = fnType.getInputs();
  auto resultTypes = fnType.getResults();

  // Print the operation and the function name.
  auto funcName =
      op->getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName())
          .getValue();
  p << op->getName() << ' ';
  p.printSymbolName(funcName);

  printModuleSignature(p, op, argTypes, /*isVariadic=*/false, resultTypes);
  printFunctionAttributes(p, op, argTypes.size(), resultTypes.size());
}

static void print(OpAsmPrinter &p, RTLExternModuleOp op) {
  printRTLModuleOp(p, op);
}

static void print(OpAsmPrinter &p, RTLModuleOp op) {
  printRTLModuleOp(p, op);

  // Print the body if this is not an external function.
  Region &body = op.getBody();
  if (!body.empty())
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
}

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

/// Lookup the module or extmodule for the symbol.  This returns null on
/// invalid IR.
Operation *InstanceOp::getReferencedModule() {
  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  if (!topLevelModuleOp)
    return nullptr;

  return topLevelModuleOp.lookupSymbol(moduleName());
}

static LogicalResult verifyInstanceOp(InstanceOp op) {
  // Check that this instance is inside a module.
  auto module = dyn_cast<RTLModuleOp>(op->getParentOp());
  if (!module) {
    op.emitOpError("should be embedded in an 'rtl.module'");
    return failure();
  }

  auto referencedModule = op.getReferencedModule();
  if (referencedModule == nullptr)
    return op.emitError("Cannot find module definition '")
           << op.moduleName() << "'";

  if (!isa<RTLModuleOp>(referencedModule) &&
      !isa<RTLExternModuleOp>(referencedModule))
    return op.emitError("Symbol resolved to '")
           << referencedModule->getName()
           << "' which is not a RTL[Ext]ModuleOp";

  if (auto paramDictOpt = op.parameters()) {
    DictionaryAttr paramDict = paramDictOpt.getValue();
    auto checkParmValue = [&](NamedAttribute elt) -> bool {
      auto value = elt.second;
      if (value.isa<IntegerAttr>() || value.isa<StringAttr>() ||
          value.isa<FloatAttr>())
        return true;
      op.emitError() << "has unknown extmodule parameter value '" << elt.first
                     << "' = " << value;
      return false;
    };

    if (!llvm::all_of(paramDict, checkParmValue))
      return failure();
  }

  return success();
}

StringAttr InstanceOp::getResultName(size_t idx) {
  auto *module = getReferencedModule();
  if (!module)
    return {};

  SmallVector<ModulePortInfo, 4> results;
  getModulePortInfo(module, results);

  for (auto &port : results) {
    if (!port.isOutput())
      continue;
    if (idx == 0)
      return port.name;
    --idx;
  }

  return StringAttr();
}

/// Suggest a name for each result value based on the saved result names
/// attribute.
void InstanceOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // Provide default names for instance results.
  std::string name;
  for (size_t i = 0, e = getNumResults(); i != e; ++i) {
    auto resultName = getResultName(i);
    name = instanceName().str() + ".";
    if (resultName)
      name += resultName.getValue().str();
    else
      name += std::to_string(i);
    setNameFn(getResult(i), name);
  }
}

//===----------------------------------------------------------------------===//
// RTLOutputOp
//===----------------------------------------------------------------------===//

/// Verify that the num of operands and types fit the declared results.
static LogicalResult verifyOutputOp(OutputOp *op) {
  OperandRange outputValues = op->getOperands();
  auto opParent = (*op)->getParentOp();

  // Check that we are in the correct region. OutputOp should be directly
  // contained by an RTLModuleOp region. We'll loosen this restriction if
  // there's a compelling use case.
  if (!isa<RTLModuleOp>(opParent)) {
    op->emitOpError("operation expected to be in a RTLModuleOp.");
    return failure();
  }

  // Check that the we (rtl.output) have the same number of operands as our
  // region has results.
  FunctionType modType = getModuleType(opParent);
  ArrayRef<Type> modResults = modType.getResults();
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
// RTL combinational ops
//===----------------------------------------------------------------------===//

/// Return true if the specified operation is a combinatorial logic op.
bool rtl::isCombinatorial(Operation *op) {
  struct IsCombClassifier
      : public CombinatorialVisitor<IsCombClassifier, bool> {
    bool visitInvalidComb(Operation *op) { return false; }
    bool visitUnhandledComb(Operation *op) { return true; }
  };

  return IsCombClassifier().dispatchCombinatorialVisitor(op);
}

static Attribute getIntAttr(const APInt &value, MLIRContext *context) {
  return IntegerAttr::get(IntegerType::get(context, value.getBitWidth()),
                          value);
}

namespace {
struct ConstantIntMatcher {
  APInt &value;
  ConstantIntMatcher(APInt &value) : value(value) {}
  bool match(Operation *op) {
    if (auto cst = dyn_cast<ConstantOp>(op)) {
      value = cst.value();
      return true;
    }
    return false;
  }
};
} // end anonymous namespace

static inline ConstantIntMatcher m_RConstant(APInt &value) {
  return ConstantIntMatcher(value);
}

//===----------------------------------------------------------------------===//
// WireOp
//===----------------------------------------------------------------------===//

void WireOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   Type elementType, StringAttr name) {
  if (name)
    odsState.addAttribute("name", name);

  odsState.addTypes(InOutType::get(elementType));
}

static void printWireOp(OpAsmPrinter &p, Operation *op) {
  p << op->getName();
  // Note that we only need to print the "name" attribute if the asmprinter
  // result name disagrees with it.  This can happen in strange cases, e.g.
  // when there are conflicts.
  bool namesDisagree = false;

  SmallString<32> resultNameStr;
  llvm::raw_svector_ostream tmpStream(resultNameStr);
  p.printOperand(op->getResult(0), tmpStream);
  auto expectedName = op->getAttrOfType<StringAttr>("name");
  if (!expectedName ||
      tmpStream.str().drop_front() != expectedName.getValue()) {
    namesDisagree = true;
  }

  if (namesDisagree)
    p.printOptionalAttrDict(op->getAttrs());
  else
    p.printOptionalAttrDict(op->getAttrs(), {"name"});

  p << " : " << op->getResult(0).getType();
}

static ParseResult parseWireOp(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typeLoc;
  Type resultType;

  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.getCurrentLocation(&typeLoc) || parser.parseType(resultType))
    return failure();

  result.addTypes(resultType);

  // If the attribute dictionary contains no 'name' attribute, infer it from
  // the SSA name (if specified).
  bool hadName = llvm::any_of(result.attributes, [](NamedAttribute attr) {
    return attr.first == "name";
  });

  // If there was no name specified, check to see if there was a useful name
  // specified in the asm file.
  if (hadName)
    return success();

  auto resultName = parser.getResultName(0);
  if (!resultName.first.empty() && !isdigit(resultName.first[0])) {
    StringRef name = resultName.first;
    auto *context = result.getContext();
    auto nameAttr = parser.getBuilder().getStringAttr(name);
    result.attributes.push_back({Identifier::get("name", context), nameAttr});
  }

  return success();
}

/// Suggest a name for each result value based on the saved result names
/// attribute.
void WireOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // If the wire has an optional 'name' attribute, use it.
  if (auto nameAttr = (*this)->getAttrOfType<StringAttr>("name"))
    setNameFn(getResult(), nameAttr.getValue());
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyConstantOp(ConstantOp constant) {
  // If the result type has a bitwidth, then the attribute must match its width.
  auto intType = constant.getType().cast<IntegerType>();
  if (constant.value().getBitWidth() != intType.getWidth())
    return constant.emitError(
        "firrtl.constant attribute bitwidth doesn't match return type");

  return success();
}

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> constants) {
  assert(constants.empty() && "constant has no operands");
  return valueAttr();
}

/// Build a ConstantOp from an APInt, infering the result type from the
/// width of the APInt.
void ConstantOp::build(OpBuilder &builder, OperationState &result,
                       const APInt &value) {

  auto type = IntegerType::get(builder.getContext(), value.getBitWidth());
  auto attr = builder.getIntegerAttr(type, value);
  return build(builder, result, type, attr);
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

/// Flattens a single input in `op` if `hasOneUse` is true and it can be defined
/// as an Op. Returns true if successful, and false otherwise.
///
/// Example: op(1, 2, op(3, 4), 5) -> op(1, 2, 3, 4, 5)  // returns true
///
template <typename Op>
static bool tryFlatteningOperands(Op op, PatternRewriter &rewriter) {
  auto inputs = op.inputs();

  for (size_t i = 0, size = inputs.size(); i != size; ++i) {
    // Don't duplicate logic.
    if (!inputs[i].hasOneUse())
      continue;
    auto flattenOp = inputs[i].template getDefiningOp<Op>();
    if (!flattenOp)
      continue;
    auto flattenOpInputs = flattenOp.inputs();

    SmallVector<Value, 4> newOperands;
    newOperands.reserve(size + flattenOpInputs.size());

    auto flattenOpIndex = inputs.begin() + i;
    newOperands.append(inputs.begin(), flattenOpIndex);
    newOperands.append(flattenOpInputs.begin(), flattenOpInputs.end());
    newOperands.append(flattenOpIndex + 1, inputs.end());

    rewriter.replaceOpWithNewOp<Op>(op, op.getType(), newOperands);
    return true;
  }
  return false;
}

void ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  auto intTy = getType().cast<IntegerType>();
  auto intCst = getValue();

  // Sugar i1 constants with 'true' and 'false'.
  if (intTy.getWidth() == 1)
    return setNameFn(getResult(), intCst.isNullValue() ? "false" : "true");

  // Otherwise, build a complex name with the value and type.
  SmallVector<char, 32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << 'c' << intCst << '_' << intTy;
  setNameFn(getResult(), specialName.str());
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

// Verify SExtOp and ZExtOp.
static LogicalResult verifyExtOp(Operation *op) {
  // The source must be smaller than the dest type.  Both are already known to
  // be signless integers.
  auto srcType = op->getOperand(0).getType().cast<IntegerType>();
  auto dstType = op->getResult(0).getType().cast<IntegerType>();
  if (srcType.getWidth() >= dstType.getWidth()) {
    op->emitOpError("extension must increase bitwidth of operand");
    return failure();
  }

  return success();
}

OpFoldResult ZExtOp::fold(ArrayRef<Attribute> constants) {
  // Constant fold.
  if (auto input = constants[0].dyn_cast_or_null<IntegerAttr>()) {
    auto destWidth = getType().cast<IntegerType>().getWidth();
    return getIntAttr(input.getValue().zext(destWidth), getContext());
  }

  return {};
}

OpFoldResult SExtOp::fold(ArrayRef<Attribute> constants) {
  // Constant fold.
  if (auto input = constants[0].dyn_cast_or_null<IntegerAttr>()) {
    auto destWidth = getType().cast<IntegerType>().getWidth();
    return getIntAttr(input.getValue().sext(destWidth), getContext());
  }

  return {};
}

OpFoldResult AndROp::fold(ArrayRef<Attribute> constants) {
  // Constant fold.
  if (auto input = constants[0].dyn_cast_or_null<IntegerAttr>())
    return getIntAttr(APInt(1, input.getValue().isAllOnesValue()),
                      getContext());

  return {};
}

OpFoldResult OrROp::fold(ArrayRef<Attribute> constants) {
  // Constant fold.
  if (auto input = constants[0].dyn_cast_or_null<IntegerAttr>())
    return getIntAttr(APInt(1, input.getValue() != 0), getContext());

  return {};
}

OpFoldResult XorROp::fold(ArrayRef<Attribute> constants) {
  // Constant fold.
  if (auto input = constants[0].dyn_cast_or_null<IntegerAttr>())
    return getIntAttr(APInt(1, input.getValue().countPopulation() & 1),
                      getContext());

  return {};
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

static LogicalResult verifyExtractOp(ExtractOp op) {
  unsigned srcWidth = op.input().getType().cast<IntegerType>().getWidth();
  unsigned dstWidth = op.getType().cast<IntegerType>().getWidth();
  if (op.lowBit() >= srcWidth || srcWidth - op.lowBit() < dstWidth)
    return op.emitOpError("from bit too large for input"), failure();

  return success();
}

OpFoldResult ExtractOp::fold(ArrayRef<Attribute> constants) {
  // If we are extracting the entire input, then return it.
  if (input().getType() == getType())
    return input();

  // Constant fold.
  if (auto input = constants[0].dyn_cast_or_null<IntegerAttr>()) {
    unsigned dstWidth = getType().cast<IntegerType>().getWidth();
    return getIntAttr(input.getValue().lshr(lowBit()).trunc(dstWidth),
                      getContext());
  }
  return {};
}

static ParseResult parseSliceTypes(OpAsmParser &p, Type &srcType,
                                   Type &idxType) {
  ArrayType arrType;
  if (p.parseType(arrType))
    return failure();
  srcType = arrType;
  unsigned idxWidth = llvm::Log2_64_Ceil(arrType.getSize());
  idxType = IntegerType::get(p.getBuilder().getContext(), idxWidth);
  return success();
}

static void printSliceTypes(OpAsmPrinter &p, Operation *, Type srcType,
                            Type idxType) {
  p.printType(srcType);
}

//===----------------------------------------------------------------------===//
// Variadic operations
//===----------------------------------------------------------------------===//

// Reduce all operands to a single value by applying the `calculate` function.
// This will fail if any of the operands are not constant.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT =
              function_ref<ElementValueT(ElementValueT, ElementValueT)>>
static Attribute constFoldVariadicOp(ArrayRef<Attribute> operands,
                                     const CalculationT &calculate) {
  if (!operands.size())
    return {};

  if (!operands[0])
    return {};

  ElementValueT accum = operands[0].cast<AttrElementT>().getValue();
  for (auto i = operands.begin() + 1, end = operands.end(); i != end; ++i) {
    auto attr = *i;
    if (!attr)
      return {};

    auto typedAttr = attr.cast<AttrElementT>();
    calculate(accum, typedAttr.getValue());
  }
  return AttrElementT::get(operands[0].getType(), accum);
}

static LogicalResult verifyUTVariadicRTLOp(Operation *op) {
  if (op->getOperands().empty())
    return op->emitOpError("requires 1 or more args");

  return success();
}

OpFoldResult AndOp::fold(ArrayRef<Attribute> constants) {
  auto width = getType().cast<IntegerType>().getWidth();
  APInt value(/*numBits=*/width, -1, /*isSigned=*/false);

  // and(x, 0, 1) -> 0 -- annulment
  for (auto operand : constants) {
    if (!operand)
      continue;
    value &= operand.cast<IntegerAttr>().getValue();
    if (value.isNullValue())
      return getIntAttr(value, getContext());
  }

  // and(x, x, x) -> x -- noop
  if (llvm::all_of(inputs(), [&](auto in) { return in == inputs()[0]; }))
    return inputs()[0];

  // Constant fold
  return constFoldVariadicOp<IntegerAttr>(
      constants, [](APInt &a, const APInt &b) { a &= b; });
}

void AndOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  struct Folder final : public OpRewritePattern<AndOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(AndOp op,
                                  PatternRewriter &rewriter) const override {
      auto inputs = op.inputs();
      auto size = inputs.size();
      assert(size > 1 && "expected 2 or more operands");

      APInt value, value2;

      // and(..., '1) -> and(...) -- identity
      if (matchPattern(inputs.back(), m_RConstant(value)) &&
          value.isAllOnesValue()) {
        rewriter.replaceOpWithNewOp<AndOp>(op, op.getType(),
                                           inputs.drop_back());
        return success();
      }

      // TODO: remove all duplicate arguments
      // and(..., x, x) -> and(..., x) -- idempotent
      if (inputs[size - 1] == inputs[size - 2]) {
        rewriter.replaceOpWithNewOp<AndOp>(op, op.getType(),
                                           inputs.drop_back());
        return success();
      }

      // TODO: Combine all constants in one shot
      // and(..., c1, c2) -> and(..., c3) where c3 = c1 & c2 -- constant folding
      if (matchPattern(inputs[size - 1], m_RConstant(value)) &&
          matchPattern(inputs[size - 2], m_RConstant(value2))) {
        auto cst = rewriter.create<ConstantOp>(op.getLoc(), value & value2);
        SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
        newOperands.push_back(cst);
        rewriter.replaceOpWithNewOp<AndOp>(op, op.getType(), newOperands);
        return success();
      }
      // and(x, and(...)) -> and(x, ...) -- flatten
      if (tryFlatteningOperands(op, rewriter))
        return success();

      /// TODO: and(..., x, not(x)) -> and(..., 0) -- complement
      return failure();
    }
  };
  results.insert<Folder>(context);
}

OpFoldResult OrOp::fold(ArrayRef<Attribute> constants) {
  auto width = getType().cast<IntegerType>().getWidth();
  APInt value(/*numBits=*/width, 0, /*isSigned=*/false);

  // or(x, 1, 0) -> 1
  for (auto operand : constants) {
    if (!operand)
      continue;
    value |= operand.cast<IntegerAttr>().getValue();
    if (value.isAllOnesValue())
      return getIntAttr(value, getContext());
  }

  // or(x, x, x) -> x
  if (llvm::all_of(inputs(), [&](auto in) { return in == inputs()[0]; }))
    return inputs()[0];

  // Constant fold
  return constFoldVariadicOp<IntegerAttr>(
      constants, [](APInt &a, const APInt &b) { a |= b; });
}

void OrOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                       MLIRContext *context) {
  struct Folder final : public OpRewritePattern<OrOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(OrOp op,
                                  PatternRewriter &rewriter) const override {
      auto inputs = op.inputs();
      auto size = inputs.size();
      assert(size > 1 && "expected 2 or more operands");

      APInt value, value2;

      // or(..., 0) -> or(...) -- identity
      if (matchPattern(inputs.back(), m_RConstant(value)) &&
          value.isNullValue()) {

        rewriter.replaceOpWithNewOp<OrOp>(op, op.getType(), inputs.drop_back());
        return success();
      }

      // or(..., x, x) -> or(..., x) -- idempotent
      if (inputs[size - 1] == inputs[size - 2]) {
        rewriter.replaceOpWithNewOp<OrOp>(op, op.getType(), inputs.drop_back());
        return success();
      }

      // or(..., c1, c2) -> or(..., c3) where c3 = c1 | c2 -- constant folding
      if (matchPattern(inputs[size - 1], m_RConstant(value)) &&
          matchPattern(inputs[size - 2], m_RConstant(value2))) {
        auto cst = rewriter.create<ConstantOp>(op.getLoc(), value | value2);
        SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
        newOperands.push_back(cst);
        rewriter.replaceOpWithNewOp<OrOp>(op, op.getType(), newOperands);
        return success();
      }

      // or(x, or(...)) -> or(x, ...) -- flatten
      if (tryFlatteningOperands(op, rewriter))
        return success();

      /// TODO: or(..., x, not(x)) -> or(..., '1) -- complement
      return failure();
    }
  };
  results.insert<Folder>(context);
}

OpFoldResult XorOp::fold(ArrayRef<Attribute> constants) {
  auto size = inputs().size();

  // xor(x) -> x -- noop
  if (size == 1u)
    return inputs()[0];

  // xor(x, x) -> 0 -- idempotent
  if (size == 2u && inputs()[0] == inputs()[1])
    return IntegerAttr::get(getType(), 0);

  // Constant fold
  return constFoldVariadicOp<IntegerAttr>(
      constants, [](APInt &a, const APInt &b) { a ^= b; });
}

void XorOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  struct Folder final : public OpRewritePattern<XorOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(XorOp op,
                                  PatternRewriter &rewriter) const override {
      auto inputs = op.inputs();
      auto size = inputs.size();
      assert(size > 1 && "expected 2 or more operands");

      APInt value, value2;

      // xor(..., 0) -> xor(...) -- identity
      if (matchPattern(inputs.back(), m_RConstant(value)) &&
          value.isNullValue()) {

        rewriter.replaceOpWithNewOp<XorOp>(op, op.getType(),
                                           inputs.drop_back());
        return success();
      }

      if (inputs[size - 1] == inputs[size - 2]) {
        assert(size > 2 &&
               "expected idempotent case for 2 elements handled already.");
        // xor(..., x, x) -> xor (...) -- idempotent
        rewriter.replaceOpWithNewOp<XorOp>(op, op.getType(),
                                           inputs.drop_back(/*n=*/2));
        return success();
      }

      // xor(..., c1, c2) -> xor(..., c3) where c3 = c1 ^ c2 -- constant folding
      if (matchPattern(inputs[size - 1], m_RConstant(value)) &&
          matchPattern(inputs[size - 2], m_RConstant(value2))) {
        auto cst = rewriter.create<ConstantOp>(op.getLoc(), value ^ value2);
        SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
        newOperands.push_back(cst);
        rewriter.replaceOpWithNewOp<XorOp>(op, op.getType(), newOperands);
        return success();
      }

      // xor(x, xor(...)) -> xor(x, ...) -- flatten
      if (tryFlatteningOperands(op, rewriter))
        return success();

      /// TODO: xor(..., '1) -> not(xor(...))
      /// TODO: xor(..., x, not(x)) -> xor(..., '1)
      return failure();
    }
  };
  results.insert<Folder>(context);
}

OpFoldResult MergeOp::fold(ArrayRef<Attribute> constants) {
  // rtl.merge(x, x, x) -> x.
  if (llvm::all_of(inputs(), [&](auto in) { return in == inputs()[0]; }))
    return inputs()[0];

  return {};
}

OpFoldResult AddOp::fold(ArrayRef<Attribute> constants) {
  auto size = inputs().size();

  // add(x) -> x -- noop
  if (size == 1u)
    return inputs()[0];

  // Constant fold
  return constFoldVariadicOp<IntegerAttr>(
      constants, [](APInt &a, const APInt &b) { a += b; });
}

void AddOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  struct Folder final : public OpRewritePattern<AddOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(AddOp op,
                                  PatternRewriter &rewriter) const override {
      auto inputs = op.inputs();
      auto size = inputs.size();
      assert(size > 1 && "expected 2 or more operands");

      APInt value, value2;

      // add(..., 0) -> add(...) -- identity
      if (matchPattern(inputs.back(), m_RConstant(value)) &&
          value.isNullValue()) {
        rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(),
                                           inputs.drop_back());
        return success();
      }

      // add(..., c1, c2) -> add(..., c3) where c3 = c1 + c2 -- constant folding
      if (matchPattern(inputs[size - 1], m_RConstant(value)) &&
          matchPattern(inputs[size - 2], m_RConstant(value2))) {
        auto cst = rewriter.create<ConstantOp>(op.getLoc(), value + value2);
        SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
        newOperands.push_back(cst);
        rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), newOperands);
        return success();
      }

      // add(..., x, x) -> add(..., shl(x, 1))
      if (inputs[size - 1] == inputs[size - 2]) {
        SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));

        auto one = rewriter.create<ConstantOp>(op.getLoc(), op.getType(), 1);
        auto shiftLeftOp =
            rewriter.create<rtl::ShlOp>(op.getLoc(), inputs.back(), one);

        newOperands.push_back(shiftLeftOp);
        rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), newOperands);
        return success();
      }

      auto shlOp = inputs[size - 1].getDefiningOp<rtl::ShlOp>();
      // add(..., x, shl(x, c)) -> add(..., mul(x, (1 << c) + 1))
      if (shlOp && shlOp.lhs() == inputs[size - 2] &&
          matchPattern(shlOp.rhs(), m_RConstant(value))) {

        APInt one(/*numBits=*/value.getBitWidth(), 1, /*isSigned=*/false);
        auto rhs =
            rewriter.create<ConstantOp>(op.getLoc(), (one << value) + one);

        std::array<Value, 2> factors = {shlOp.lhs(), rhs};
        auto mulOp = rewriter.create<rtl::MulOp>(op.getLoc(), factors);

        SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
        newOperands.push_back(mulOp);
        rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), newOperands);
        return success();
      }

      auto mulOp = inputs[size - 1].getDefiningOp<rtl::MulOp>();
      // add(..., x, mul(x, c)) -> add(..., mul(x, c + 1))
      if (mulOp && mulOp.inputs().size() == 2 &&
          mulOp.inputs()[0] == inputs[size - 2] &&
          matchPattern(mulOp.inputs()[1], m_RConstant(value))) {

        APInt one(/*numBits=*/value.getBitWidth(), 1, /*isSigned=*/false);
        auto rhs = rewriter.create<ConstantOp>(op.getLoc(), value + one);
        std::array<Value, 2> factors = {mulOp.inputs()[0], rhs};
        auto newMulOp = rewriter.create<rtl::MulOp>(op.getLoc(), factors);

        SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
        newOperands.push_back(newMulOp);
        rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), newOperands);
        return success();
      }

      // add(x, add(...)) -> add(x, ...) -- flatten
      if (tryFlatteningOperands(op, rewriter))
        return success();

      return failure();
    }
  };
  results.insert<Folder>(context);
}

OpFoldResult MulOp::fold(ArrayRef<Attribute> constants) {
  auto size = inputs().size();

  // mul(x) -> x -- noop
  if (size == 1u)
    return inputs()[0];

  auto width = getType().getWidth();
  APInt value(/*numBits=*/width, 1, /*isSigned=*/false);

  // mul(x, 0, 1) -> 0 -- annulment
  for (auto operand : constants) {
    if (!operand)
      continue;
    value *= operand.cast<IntegerAttr>().getValue();
    if (value.isNullValue())
      return getIntAttr(value, getContext());
  }

  // Constant fold
  return constFoldVariadicOp<IntegerAttr>(
      constants, [](APInt &a, const APInt &b) { a *= b; });
}

void MulOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  struct Folder final : public OpRewritePattern<MulOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(MulOp op,
                                  PatternRewriter &rewriter) const override {
      auto inputs = op.inputs();
      auto size = inputs.size();
      assert(size > 1 && "expected 2 or more operands");

      APInt value, value2;

      // mul(x, c) -> shl(x, log2(c)), where c is a power of two.
      if (size == 2 && matchPattern(inputs.back(), m_RConstant(value)) &&
          value.isPowerOf2()) {
        auto shift = rewriter.create<ConstantOp>(op.getLoc(), op.getType(),
                                                 value.exactLogBase2());
        auto shlOp = rewriter.create<rtl::ShlOp>(op.getLoc(), inputs[0], shift);

        rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(),
                                           ArrayRef<Value>(shlOp));
        return success();
      }

      // mul(..., 1) -> mul(...) -- identity
      if (matchPattern(inputs.back(), m_RConstant(value)) && (value == 1u)) {
        rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(),
                                           inputs.drop_back());
        return success();
      }

      // mul(..., c1, c2) -> mul(..., c3) where c3 = c1 * c2 -- constant folding
      if (matchPattern(inputs[size - 1], m_RConstant(value)) &&
          matchPattern(inputs[size - 2], m_RConstant(value2))) {
        auto cst = rewriter.create<ConstantOp>(op.getLoc(), value * value2);
        SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
        newOperands.push_back(cst);
        rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), newOperands);
        return success();
      }

      // mul(a, mul(...)) -> mul(a, ...) -- flatten
      if (tryFlatteningOperands(op, rewriter))
        return success();

      return failure();
    }
  };
  results.insert<Folder>(context);
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

void ConcatOp::build(OpBuilder &builder, OperationState &result,
                     ValueRange inputs) {
  unsigned resultWidth = 0;
  for (auto input : inputs) {
    resultWidth += input.getType().cast<IntegerType>().getWidth();
  }
  build(builder, result, builder.getIntegerType(resultWidth), inputs);
}

// Constant folding
OpFoldResult ConcatOp::fold(ArrayRef<Attribute> constants) {
  // If all the operands are constant, we can fold.
  for (auto attr : constants)
    if (!attr || !attr.isa<IntegerAttr>())
      return {};

  // If we got here, we can constant fold.
  unsigned resultWidth = getType().getIntOrFloatBitWidth();
  APInt result(resultWidth, 0);

  unsigned nextInsertion = resultWidth;
  // Insert each chunk into the result.
  for (auto attr : constants) {
    auto chunk = attr.cast<IntegerAttr>().getValue();
    nextInsertion -= chunk.getBitWidth();
    result |= chunk.zext(resultWidth) << nextInsertion;
  }

  return getIntAttr(result, getContext());
}

static LogicalResult tryCanonicalizeConcat(ConcatOp op,
                                           PatternRewriter &rewriter) {
  auto inputs = op.inputs();
  auto size = inputs.size();
  assert(size > 1 && "expected 2 or more operands");

  // This function is used when we flatten neighboring operands of a (variadic)
  // concat into a new vesion of the concat.  first/last indices are inclusive.
  auto flattenConcat = [&](size_t firstOpIndex, size_t lastOpIndex,
                           ValueRange replacements) -> LogicalResult {
    SmallVector<Value, 4> newOperands;
    newOperands.append(inputs.begin(), inputs.begin() + firstOpIndex);
    newOperands.append(replacements.begin(), replacements.end());
    newOperands.append(inputs.begin() + lastOpIndex + 1, inputs.end());
    if (newOperands.size() == 1)
      rewriter.replaceOp(op, newOperands[0]);
    else
      rewriter.replaceOpWithNewOp<ConcatOp>(op, op.getType(), newOperands);
    return success();
  };

  for (size_t i = 0; i != size; ++i) {
    // If an operand to the concat is itself a concat, then we can fold them
    // together.
    if (auto subConcat = inputs[i].getDefiningOp<ConcatOp>())
      return flattenConcat(i, i, subConcat->getOperands());

    // We can flatten a zext into the concat, since a zext is a 0 plus the input
    // value.
    if (auto zext = inputs[i].getDefiningOp<ZExtOp>()) {
      unsigned zeroWidth = zext.getType().getIntOrFloatBitWidth() -
                           zext.input().getType().getIntOrFloatBitWidth();
      Value replacement[2] = {
          rewriter.create<ConstantOp>(op.getLoc(), APInt(zeroWidth, 0)),
          zext.input()};

      return flattenConcat(i, i, replacement);
    }

    // We can flatten a sext into the concat, since a sext is an extraction of
    // a sign bit plus the input value itself.
    if (auto sext = inputs[i].getDefiningOp<SExtOp>()) {
      unsigned inputWidth = sext.input().getType().getIntOrFloatBitWidth();

      auto sign = rewriter.create<ExtractOp>(op.getLoc(), rewriter.getI1Type(),
                                             sext.input(), inputWidth - 1);
      SmallVector<Value> replacements;
      unsigned signWidth = sext.getType().getIntOrFloatBitWidth() - inputWidth;
      replacements.append(signWidth, sign);
      replacements.push_back(sext.input());
      return flattenConcat(i, i, replacements);
    }

    // Check for canonicalization due to neighboring operands.
    if (i != 0) {
      // Merge neighboring constants.
      if (auto cst = inputs[i].getDefiningOp<ConstantOp>()) {
        if (auto prevCst = inputs[i - 1].getDefiningOp<ConstantOp>()) {
          unsigned prevWidth = prevCst.getValue().getBitWidth();
          unsigned thisWidth = cst.getValue().getBitWidth();
          auto resultCst = cst.getValue().zext(prevWidth + thisWidth);
          resultCst |= prevCst.getValue().zext(prevWidth + thisWidth)
                       << thisWidth;
          Value replacement =
              rewriter.create<ConstantOp>(op.getLoc(), resultCst);
          return flattenConcat(i - 1, i, replacement);
        }
      }

      // Merge neighboring extracts of neighboring inputs, e.g.
      // {A[3], A[2]} -> A[3,2]
      if (auto extract = inputs[i].getDefiningOp<ExtractOp>()) {
        if (auto prevExtract = inputs[i - 1].getDefiningOp<ExtractOp>()) {
          if (extract.input() == prevExtract.input()) {
            auto thisWidth = extract.getType().getIntOrFloatBitWidth();
            if (prevExtract.lowBit() == extract.lowBit() + thisWidth) {
              auto prevWidth = prevExtract.getType().getIntOrFloatBitWidth();
              auto resType = rewriter.getIntegerType(thisWidth + prevWidth);
              Value replacement = rewriter.create<ExtractOp>(
                  op.getLoc(), resType, extract.input());
              return flattenConcat(i - 1, i, replacement);
            }
          }
        }
      }
    }
  }

  // Return true if this extract is an extract of the sign bit of its input.
  auto isSignBitExtract = [](ExtractOp op) -> bool {
    if (op.getType().getIntOrFloatBitWidth() != 1)
      return false;
    return op.lowBit() == op.input().getType().getIntOrFloatBitWidth() - 1;
  };

  // Folds concat back into a sext.
  if (auto extract = inputs[0].getDefiningOp<ExtractOp>()) {
    if (extract.input() == inputs.back() && isSignBitExtract(extract)) {
      // Check intermediate bits.
      bool allMatch = true;
      // The intermediate bits are allowed to be difference instances of extract
      // (because canonicalize doesn't do CSE automatically) so long as they are
      // getting the sign bit.
      for (size_t i = 1, e = inputs.size() - 1; i != e; ++i) {
        auto extractInner = inputs[i].getDefiningOp<ExtractOp>();
        if (!extractInner || extractInner.input() != inputs.back() ||
            !isSignBitExtract(extractInner)) {
          allMatch = false;
          break;
        }
      }
      if (allMatch) {
        rewriter.replaceOpWithNewOp<SExtOp>(op, op.getType(), inputs.back());
        return success();
      }
    }
  }

  return failure();
}

void ConcatOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  struct Folder final : public OpRewritePattern<ConcatOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(ConcatOp op,
                                  PatternRewriter &rewriter) const override {
      return tryCanonicalizeConcat(op, rewriter);
    }
  };
  results.insert<Folder>(context);
}

//===----------------------------------------------------------------------===//
// ReadInOutOp
//===----------------------------------------------------------------------===//

void ReadInOutOp::build(OpBuilder &builder, OperationState &result,
                        Value input) {
  auto resultType = input.getType().cast<InOutType>().getElementType();
  build(builder, result, resultType, input);
}

//===----------------------------------------------------------------------===//
// StructCreateOp
//===----------------------------------------------------------------------===//

static ParseResult parseStructCreateOp(OpAsmParser &parser,
                                       OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
  llvm::SmallVector<OpAsmParser::OperandType, 4> operands;
  StructType declType;

  if (parser.parseLParen() || parser.parseOperandList(operands) ||
      parser.parseRParen() || parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(declType))
    return failure();

  llvm::SmallVector<Type, 4> structInnerTypes;
  declType.getInnerTypes(structInnerTypes);
  result.addTypes(declType);

  if (parser.resolveOperands(operands, structInnerTypes, inputOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

static void print(OpAsmPrinter &printer, rtl::StructCreateOp op) {
  printer << op.getOperationName() << " (";
  printer.printOperands(op.input());
  printer << ")";
  printer.printOptionalAttrDict(op.getAttrs());
  printer << " : " << op.getType();
}

//===----------------------------------------------------------------------===//
// StructExplodeOp
//===----------------------------------------------------------------------===//

static ParseResult parseStructExplodeOp(OpAsmParser &parser,
                                        OperationState &result) {
  OpAsmParser::OperandType operand;
  StructType declType;

  if (parser.parseOperand(operand) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(declType))
    return failure();

  llvm::SmallVector<Type, 4> structInnerTypes;
  declType.getInnerTypes(structInnerTypes);
  result.addTypes(structInnerTypes);

  if (parser.resolveOperand(operand, declType, result.operands))
    return failure();
  return success();
}

static void print(OpAsmPrinter &printer, rtl::StructExplodeOp op) {
  printer << op.getOperationName() << " ";
  printer.printOperand(op.input());
  printer.printOptionalAttrDict(op.getAttrs());
  printer << " : " << op.input().getType();
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

static ParseResult parseStructExtractOp(OpAsmParser &parser,
                                        OperationState &result) {
  OpAsmParser::OperandType operand;
  StringAttr fieldName;
  StructType declType;

  if (parser.parseOperand(operand) || parser.parseLSquare() ||
      parser.parseAttribute(fieldName, "field", result.attributes) ||
      parser.parseRSquare() ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(declType))
    return failure();

  Type resultType = declType.getFieldType(fieldName.getValue());
  if (!resultType) {
    parser.emitError(parser.getNameLoc(), "invalid field name specified");
    return failure();
  }
  result.addTypes(resultType);

  if (parser.resolveOperand(operand, declType, result.operands))
    return failure();
  return success();
}

static void print(OpAsmPrinter &printer, rtl::StructExtractOp op) {
  printer << op.getOperationName() << " ";
  printer.printOperand(op.input());
  printer << "[\"" << op.field() << "\"]";
  printer.printOptionalAttrDict(op.getAttrs(), {"field"});
  printer << " : " << op.input().getType();
}

//===----------------------------------------------------------------------===//
// StructInjectOp
//===----------------------------------------------------------------------===//

static ParseResult parseStructInjectOp(OpAsmParser &parser,
                                       OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
  OpAsmParser::OperandType operand, val;
  StringAttr fieldName;
  StructType declType;

  if (parser.parseOperand(operand) || parser.parseLSquare() ||
      parser.parseAttribute(fieldName, "field", result.attributes) ||
      parser.parseRSquare() || parser.parseComma() ||
      parser.parseOperand(val) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(declType))
    return failure();

  Type resultType = declType.getFieldType(fieldName.getValue());
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

static void print(OpAsmPrinter &printer, rtl::StructInjectOp op) {
  printer << op.getOperationName() << " ";
  printer.printOperand(op.input());
  printer << "[\"" << op.field() << "\"], ";
  printer.printOperand(op.newValue());
  printer.printOptionalAttrDict(op.getAttrs(), {"field"});
  printer << " : " << op.input().getType();
}

//===----------------------------------------------------------------------===//
// ArrayIndexOp
//===----------------------------------------------------------------------===//

void ArrayIndexOp::build(OpBuilder &builder, OperationState &result,
                         Value input, Value index) {
  auto resultType = input.getType().cast<InOutType>().getElementType();
  resultType = getAnyRTLArrayElementType(resultType);
  assert(resultType && "input should have 'inout of an array' type");
  build(builder, result, InOutType::get(resultType), input, index);
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/RTL/RTL.cpp.inc"
