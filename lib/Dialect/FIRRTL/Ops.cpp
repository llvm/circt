//===- Ops.cpp - Implement the FIRRTL operations --------------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/Ops.h"
#include "circt/Dialect/FIRRTL/Visitors.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/StandardTypes.h"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// CircuitOp
//===----------------------------------------------------------------------===//

void CircuitOp::build(OpBuilder &builder, OperationState &result,
                      StringAttr name) {
  // Add an attribute for the name.
  result.addAttribute(builder.getIdentifier("name"), name);

  // Create a region and a block for the body.  The argument of the region is
  // the loop induction variable.
  Region *bodyRegion = result.addRegion();
  Block *body = new Block();
  bodyRegion->push_back(body);
  CircuitOp::ensureTerminator(*bodyRegion, builder, result.location);
}

static void print(OpAsmPrinter &p, CircuitOp op) {
  p << op.getOperationName() << " ";
  p.printAttribute(op.nameAttr());

  p.printOptionalAttrDictWithKeyword(op.getAttrs(), {"name"});

  p.printRegion(op.body(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

static ParseResult parseCircuitOp(OpAsmParser &parser, OperationState &result) {
  // Parse the module name.
  StringAttr nameAttr;
  if (parser.parseAttribute(nameAttr, "name", result.attributes))
    return failure();

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*regionArgs*/ {}, /*argTypes*/ {}))
    return failure();

  CircuitOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

static LogicalResult verifyCircuitOp(CircuitOp &circuit) {
  StringRef main = circuit.name();

  // Check that the circuit has a non-empty name.
  if (main.empty()) {
    circuit.emitOpError("must have a non-empty name");
    return failure();
  }

  // Check that a module matching the "main" module exists in the circuit.
  if (!circuit.lookupSymbol(main)) {
    circuit.emitOpError("must contain one module that matches main name '" +
                        main + "'");
    return failure();
  }

  return success();
}

Region &CircuitOp::getBodyRegion() { return getOperation()->getRegion(0); }
Block *CircuitOp::getBody() { return &getBodyRegion().front(); }

//===----------------------------------------------------------------------===//
// FExtModuleOp and FModuleOp
//===----------------------------------------------------------------------===//

FunctionType firrtl::getModuleType(Operation *op) {
  auto typeAttr = op->getAttrOfType<TypeAttr>(FModuleOp::getTypeAttrName());
  return typeAttr.getValue().cast<FunctionType>();
}

/// This function can extract information about ports from a module and an
/// extmodule.
void firrtl::getModulePortInfo(Operation *op,
                               SmallVectorImpl<ModulePortInfo> &results) {
  auto argTypes = getModuleType(op).getInputs();

  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    auto argAttrs = ::mlir::impl::getArgAttrs(op, i);
    auto type = argTypes[i].dyn_cast<FIRRTLType>();

    // Convert IntegerType ports to IntType ports transparently.
    if (!type) {
      auto intType = argTypes[i].cast<IntegerType>();
      type = IntType::get(op->getContext(), intType.isSigned(),
                          intType.getWidth());
    }

    results.push_back({getFIRRTLNameAttr(argAttrs), type});
  }
}

static void buildModule(OpBuilder &builder, OperationState &result,
                        StringAttr name,
                        ArrayRef<std::pair<StringAttr, FIRRTLType>> ports) {
  using namespace mlir::impl;

  // Add an attribute for the name.
  result.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), name);

  SmallVector<Type, 4> argTypes;
  for (auto elt : ports)
    argTypes.push_back(elt.second);

  // Record the argument and result types as an attribute.
  auto type = builder.getFunctionType(argTypes, /*resultTypes*/ {});
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

  // Record the names of the arguments if present.
  SmallString<8> attrNameBuf;
  for (size_t i = 0, e = ports.size(); i != e; ++i) {
    if (ports[i].first.getValue().empty())
      continue;

    auto argAttr =
        NamedAttribute(builder.getIdentifier("firrtl.name"), ports[i].first);

    result.addAttribute(getArgAttrName(i, attrNameBuf),
                        builder.getDictionaryAttr(argAttr));
  }

  result.addRegion();
}

void FModuleOp::build(OpBuilder &builder, OperationState &result,
                      StringAttr name,
                      ArrayRef<std::pair<StringAttr, FIRRTLType>> ports) {
  buildModule(builder, result, name, ports);

  // Create a region and a block for the body.
  auto *bodyRegion = result.regions[0].get();
  Block *body = new Block();
  bodyRegion->push_back(body);

  // Add arguments to the body block.
  for (auto elt : ports)
    body->addArgument(elt.second);

  FModuleOp::ensureTerminator(*bodyRegion, builder, result.location);
}

void FExtModuleOp::build(OpBuilder &builder, OperationState &result,
                         StringAttr name,
                         ArrayRef<std::pair<StringAttr, FIRRTLType>> ports,
                         StringRef defnameAttr) {
  buildModule(builder, result, name, ports);
  if (!defnameAttr.empty())
    result.addAttribute("defname", builder.getStringAttr(defnameAttr));
}

// TODO: This ia a clone of mlir::impl::printFunctionSignature, refactor it to
// allow this customization.
static void printFunctionSignature2(OpAsmPrinter &p, Operation *op,
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

    // If the argument has the firrtl.name attribute, and if it was used by
    // the printer exactly (not name mangled with a suffix etc) then we can
    // omit the firrtl.name attribute from the argument attribute dictionary.
    ArrayRef<StringRef> elidedAttrs;
    StringRef tmp;
    if (argumentValue) {
      if (auto nameAttr = getFIRRTLNameAttr(argAttrs)) {

        // Check to make sure the asmprinter is printing it correctly.
        SmallString<32> resultNameStr;
        llvm::raw_svector_ostream tmpStream(resultNameStr);
        p.printOperand(argumentValue, tmpStream);

        // If the name is the same as we would otherwise use, then we're good!
        if (tmpStream.str().drop_front() == nameAttr.getValue()) {
          tmp = "firrtl.name";
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
}

static void printModuleLikeOp(OpAsmPrinter &p, Operation *op) {
  using namespace mlir::impl;

  FunctionType fnType = getModuleType(op);
  auto argTypes = fnType.getInputs();
  auto resultTypes = fnType.getResults();

  // TODO: Should refactor mlir::impl::printFunctionLikeOp to allow these
  // customizations.  Need to not print the terminator.

  // Print the operation and the function name.
  auto funcName =
      op->getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName())
          .getValue();
  p << op->getName() << ' ';
  p.printSymbolName(funcName);

  printFunctionSignature2(p, op, argTypes, /*isVariadic*/ false, resultTypes);
  printFunctionAttributes(p, op, argTypes.size(), resultTypes.size());
}

static void print(OpAsmPrinter &p, FExtModuleOp op) {
  printModuleLikeOp(p, op);
}

static void print(OpAsmPrinter &p, FModuleOp op) {
  printModuleLikeOp(p, op);

  // Print the body if this is not an external function.
  Region &body = op.getBody();
  if (!body.empty())
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
}

static ParseResult parseFModuleOp(OpAsmParser &parser, OperationState &result,
                                  bool isExtModule = false) {
  using namespace mlir::impl;

  // TODO: Should refactor mlir::impl::parseFunctionLikeOp to allow these
  // customizations for implicit argument names.  Need to not print the
  // terminator.

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
  if (parseFunctionSignature(parser, /*allowVariadic*/ false, entryArgs,
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

  // Postprocess each of the arguments.  If there was no 'firrtl.name'
  // attribute, and if the argument name was non-numeric, then add the
  // firrtl.name attribute with the textual name from the IR.  The name in the
  // text file is a load-bearing part of the IR, but we don't want the
  // verbosity in dumps of including it explicitly in the attribute
  // dictionary.
  for (size_t i = 0, e = argAttrs.size(); i != e; ++i) {
    auto &attrs = argAttrs[i];

    // If an explicit name attribute was present, don't add the implicit one.
    bool hasNameAttr = false;
    for (auto &elt : attrs)
      if (elt.first.str() == "firrtl.name")
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
    attrs.push_back({Identifier::get("firrtl.name", context), nameAttr});
  }

  // Add the attributes to the function arguments.
  addArgAndResultAttrs(builder, result, argAttrs, resultAttrs);

  // Parse the optional function body.
  auto *body = result.addRegion();
  if (parser.parseOptionalRegion(
          *body, entryArgs, entryArgs.empty() ? ArrayRef<Type>() : argTypes))
    return failure();

  if (!isExtModule)
    FModuleOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

static ParseResult parseFExtModuleOp(OpAsmParser &parser,
                                     OperationState &result) {
  return parseFModuleOp(parser, result, /*isExtModule:*/ true);
}

static LogicalResult verifyFModuleOp(FModuleOp module) {
  // The parent op must be a circuit op.
  auto *parentOp = module.getParentOp();
  if (!parentOp || !isa<CircuitOp>(parentOp)) {
    module.emitOpError("should be embedded into a firrtl.circuit");
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

/// Return the type of a mem given a list of named ports and their kind.
/// This returns a null type if there are duplicate port names.
FIRRTLType
MemOp::getTypeForPortList(uint64_t depth, FIRRTLType dataType,
                          ArrayRef<std::pair<Identifier, PortKind>> portList) {
  assert(dataType.isPassiveType() && "mem can only have passive datatype");

  auto *context = dataType.getContext();

  SmallVector<std::pair<Identifier, PortKind>, 4> ports(portList.begin(),
                                                        portList.end());

  // Canonicalize the port names into alphabetic order and check for duplicates.
  llvm::array_pod_sort(
      ports.begin(), ports.end(),
      [](const std::pair<Identifier, MemOp::PortKind> *lhs,
         const std::pair<Identifier, MemOp::PortKind> *rhs) -> int {
        return lhs->first.strref().compare(rhs->first.strref());
      });

  // Reject duplicate ports.
  for (size_t i = 1, e = ports.size(); i < e; ++i)
    if (ports[i - 1].first == ports[i].first)
      return {};

  // Figure out the number of bits needed for the address, and thus the address
  // type to use.
  auto addressType = UIntType::get(context, llvm::Log2_64_Ceil(depth));

  auto getId = [&](StringRef name) -> Identifier {
    return Identifier::get(name, context);
  };

  // Okay, we've validated the data, construct the result type.
  SmallVector<BundleType::BundleElement, 4> memFields;
  SmallVector<BundleType::BundleElement, 5> portFields;
  // Common fields for all port types.
  portFields.push_back({getId("addr"), addressType});
  portFields.push_back({getId("en"), UIntType::get(context, 1)});
  portFields.push_back({getId("clk"), ClockType::get(context)});

  for (auto port : ports) {
    // Reuse the first three fields, but drop the rest.
    portFields.erase(portFields.begin() + 3, portFields.end());
    switch (port.second) {
    case PortKind::Read:
      portFields.push_back({getId("data"), FlipType::get(dataType)});
      break;

    case PortKind::Write:
      portFields.push_back({getId("data"), dataType});
      portFields.push_back({getId("mask"), dataType.getMaskType()});
      break;

    case PortKind::ReadWrite:
      portFields.push_back({getId("wmode"), UIntType::get(context, 1)});
      portFields.push_back({getId("rdata"), FlipType::get(dataType)});
      portFields.push_back({getId("wdata"), dataType});
      portFields.push_back({getId("wmask"), dataType.getMaskType()});
      break;
    }

    memFields.push_back({port.first, BundleType::get(portFields, context)});
  }

  return BundleType::get(memFields, context);
}

/// Return the kind of port this is given the port type from a 'mem' decl.
static Optional<MemOp::PortKind> getMemPortKindFromType(FIRRTLType type) {
  auto portType = type.dyn_cast<BundleType>();
  if (!portType)
    return None;

  switch (portType.getNumElements()) {
  default:
    return None;
  case 4:
    return MemOp::PortKind::Read;
  case 5:
    return MemOp::PortKind::Write;
  case 7:
    return MemOp::PortKind::ReadWrite;
  }
}

/// Return the name and kind of ports supported by this memory.
void MemOp::getPorts(
    SmallVectorImpl<std::pair<Identifier, MemOp::PortKind>> &result) {
  // The type of a mem must be a bundle.
  auto bundle = getType().cast<BundleType>();

  // Each entry in the bundle is a port.
  for (auto elt : bundle.getElements()) {
    // Each port is a bundle.
    auto kind = getMemPortKindFromType(elt.second);
    assert(kind.hasValue() && "unknown port type!");
    result.push_back({elt.first, kind.getValue()});
  }
}

/// Return the kind of the specified port or None if the name is invalid.
Optional<MemOp::PortKind> MemOp::getPortKind(StringRef portName) {
  // The type of a mem must be a bundle.
  auto eltType = getType().cast<BundleType>().getElementType(portName);
  if (!eltType)
    return None;
  return getMemPortKindFromType(eltType);
}

/// Return the data-type field of the memory, the type of each element.
FIRRTLType MemOp::getDataTypeOrNull() {
  // The outer level of a mem is a bundle, containing the input and output
  // ports.
  auto bundle = getType().cast<BundleType>();

  // Mems with no read/write ports are legal.
  if (bundle.getElements().empty())
    return {};

  auto firstPort = bundle.getElements()[0];
  auto firstPortType = firstPort.second.getPassiveType().cast<BundleType>();
  return firstPortType.getElementType("data");
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

void WhenOp::createElseRegion() {
  assert(!hasElseRegion() && "already has an else region");
  OpBuilder builder(&elseRegion());
  WhenOp::ensureTerminator(elseRegion(), builder, getLoc());
}

void WhenOp::build(OpBuilder &builder, OperationState &result, Value condition,
                   bool withElseRegion) {
  result.addOperands(condition);

  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();
  WhenOp::ensureTerminator(*thenRegion, builder, result.location);
  if (withElseRegion)
    WhenOp::ensureTerminator(*elseRegion, builder, result.location);
}

static void print(OpAsmPrinter &p, WhenOp op) {
  p << op.getOperationName() << " ";
  p.printOperand(op.condition());

  p.printRegion(op.thenRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);

  // Print the 'else' regions if it has any blocks.
  auto &elseRegion = op.elseRegion();
  if (!elseRegion.empty()) {
    p << " else";
    p.printRegion(elseRegion,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
  }

  // Print the attribute list.
  p.printOptionalAttrDictWithKeyword(op.getAttrs());
}

static ParseResult parseWhenOp(OpAsmParser &parser, OperationState &result) {
  // Parse the module name.
  OpAsmParser::OperandType conditionOperand;
  if (parser.parseOperand(conditionOperand) ||
      parser.resolveOperand(conditionOperand,
                            UIntType::get(result.getContext(), 1),
                            result.operands))
    return failure();

  // Create the regions for 'then' and 'else'.  The latter must be created even
  // if it remains empty for the validity of the operation.
  result.regions.reserve(2);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();

  // Parse the 'then' region.
  if (parser.parseRegion(*thenRegion, {}, {}))
    return failure();
  WhenOp::ensureTerminator(*thenRegion, parser.getBuilder(), result.location);

  // If we find an 'else' keyword then parse the 'else' region.
  if (!parser.parseOptionalKeyword("else")) {
    if (parser.parseRegion(*elseRegion, {}, {}))
      return failure();
    WhenOp::ensureTerminator(*elseRegion, parser.getBuilder(), result.location);
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

/// Return true if the specified operation is a firrtl expression.
bool firrtl::isExpression(Operation *op) {
  struct IsExprClassifier : public ExprVisitor<IsExprClassifier, bool> {
    bool visitInvalidExpr(Operation *op) { return false; }
    bool visitUnhandledExpr(Operation *op) { return true; }
  };

  return IsExprClassifier().dispatchExprVisitor(op);
}

static LogicalResult verifyConstantOp(ConstantOp constant) {
  // If the result type has a bitwidth, then the attribute must match its width.
  auto intType = constant.getType().cast<IntType>();
  auto width = intType.getWidthOrSentinel();
  if (width != -1 && (int)constant.value().getBitWidth() != width) {
    constant.emitError(
        "firrtl.constant attribute bitwidth doesn't match return type");
    return failure();
  }

  return success();
}

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return valueAttr();
}

/// Build a ConstantOp from an APInt and a FIRRTL type, handling the attribute
/// formation for the 'value' attribute.
void ConstantOp::build(OpBuilder &builder, OperationState &result, IntType type,
                       const APInt &value) {

  int32_t width = type.getWidthOrSentinel();
  assert((width == -1 || (int32_t)value.getBitWidth() == width) &&
         "incorrect attribute bitwidth for firrtl.constant");

  auto signedness =
      type.isSigned() ? IntegerType::Signed : IntegerType::Unsigned;
  Type attrType =
      IntegerType::get(value.getBitWidth(), signedness, type.getContext());
  auto attr = builder.getIntegerAttr(attrType, value);
  return build(builder, result, type, attr);
}

// Return the result of a subfield operation.
FIRRTLType SubfieldOp::getResultType(FIRRTLType inType, StringRef fieldName) {
  if (auto bundleType = inType.dyn_cast<BundleType>()) {
    for (auto &elt : bundleType.getElements()) {
      if (elt.first == fieldName)
        return elt.second;
    }
  }

  if (auto flipType = inType.dyn_cast<FlipType>())
    if (auto subType = getResultType(flipType.getElementType(), fieldName))
      return FlipType::get(subType);

  return {};
}

FIRRTLType SubindexOp::getResultType(FIRRTLType inType, unsigned fieldIdx) {
  if (auto vectorType = inType.dyn_cast<FVectorType>())
    if (fieldIdx < vectorType.getNumElements())
      return vectorType.getElementType();

  if (auto flipType = inType.dyn_cast<FlipType>())
    if (auto subType = getResultType(flipType.getElementType(), fieldIdx))
      return FlipType::get(subType);

  return {};
}

FIRRTLType SubaccessOp::getResultType(FIRRTLType inType, FIRRTLType indexType) {
  if (auto vectorType = inType.dyn_cast<FVectorType>())
    if (indexType.getPassiveType().isa<UIntType>())
      return vectorType.getElementType();

  if (auto flipType = inType.dyn_cast<FlipType>())
    if (auto subType = getResultType(flipType.getElementType(), indexType))
      return FlipType::get(subType);

  return {};
}

//===----------------------------------------------------------------------===//
// Binary Primitives
//===----------------------------------------------------------------------===//

/// If LHS and RHS are both UInt or SInt types, the return true and compute the
/// max width of them if known.  If unknown, return -1 in maxWidth.
static bool isSameIntegerType(FIRRTLType lhs, FIRRTLType rhs,
                              int32_t &maxWidth) {
  // Must have two integer types with the same signedness.
  auto lhsi = lhs.dyn_cast<IntType>();
  if (!lhsi || lhsi.getTypeID() != rhs.getTypeID())
    return false;

  auto lhsWidth = lhsi.getWidth();
  auto rhsWidth = rhs.cast<IntType>().getWidth();
  if (lhsWidth.hasValue() && rhsWidth.hasValue())
    maxWidth = std::max(lhsWidth.getValue(), rhsWidth.getValue());
  else
    maxWidth = -1;
  return true;
}

FIRRTLType firrtl::getAddSubResult(FIRRTLType lhs, FIRRTLType rhs) {
  int32_t width;
  if (isSameIntegerType(lhs, rhs, width)) {
    if (width != -1)
      ++width;
    return IntType::get(lhs.getContext(), lhs.isa<SIntType>(), width);
  }

  return {};
}

FIRRTLType firrtl::getMulResult(FIRRTLType lhs, FIRRTLType rhs) {
  if (lhs.getTypeID() != rhs.getTypeID())
    return {};

  int32_t width = -1;
  if (auto lu = lhs.dyn_cast<UIntType>()) {
    auto widthV = lu.getWidth();
    auto ru = rhs.cast<UIntType>();
    if (widthV.hasValue() && ru.getWidth().getValue())
      width = widthV.getValue() + ru.getWidth().getValue();
    return UIntType::get(lhs.getContext(), width);
  }

  if (auto ls = lhs.dyn_cast<SIntType>()) {
    auto widthV = ls.getWidth();
    auto rs = rhs.cast<SIntType>();
    if (widthV.hasValue() && rs.getWidth().hasValue())
      width = ls.getWidthOrSentinel() + rs.getWidthOrSentinel();
    return SIntType::get(lhs.getContext(), width);
  }
  return {};
}

FIRRTLType firrtl::getDivResult(FIRRTLType lhs, FIRRTLType rhs) {
  if (lhs.getTypeID() != rhs.getTypeID())
    return {};

  // For unsigned, the width is the width of the numerator on the LHS.
  if (auto lu = lhs.dyn_cast<UIntType>())
    return UIntType::get(lhs.getContext(), lu.getWidthOrSentinel());

  // For signed, the width is the width of the numerator on the LHS, plus 1.
  if (auto ls = lhs.dyn_cast<SIntType>()) {
    int32_t width = -1;
    if (ls.getWidth().hasValue())
      width = ls.getWidth().getValue() + 1;
    return SIntType::get(lhs.getContext(), width);
  }
  return {};
}

FIRRTLType firrtl::getRemResult(FIRRTLType lhs, FIRRTLType rhs) {
  if (lhs.getTypeID() != rhs.getTypeID())
    return {};

  int32_t width = -1;
  if (auto lu = lhs.dyn_cast<UIntType>()) {
    auto widthV = lu.getWidth();
    auto ru = rhs.cast<UIntType>();
    if (widthV.hasValue() && ru.getWidth().getValue())
      width = std::min(widthV.getValue(), ru.getWidth().getValue());
    return UIntType::get(lhs.getContext(), width);
  }

  if (auto ls = lhs.dyn_cast<SIntType>()) {
    auto widthV = ls.getWidth();
    auto rs = rhs.cast<SIntType>();
    if (widthV.hasValue() && rs.getWidth().hasValue())
      width = std::min(ls.getWidthOrSentinel(), rs.getWidthOrSentinel());
    return SIntType::get(lhs.getContext(), width);
  }

  return {};
}

FIRRTLType firrtl::getCompareResult(FIRRTLType lhs, FIRRTLType rhs) {
  if ((lhs.isa<UIntType>() && rhs.isa<UIntType>()) ||
      (lhs.isa<SIntType>() && rhs.isa<SIntType>()))
    return UIntType::get(lhs.getContext(), 1);
  return {};
}

FIRRTLType firrtl::getBitwiseBinaryResult(FIRRTLType lhs, FIRRTLType rhs) {
  int32_t width;
  if (isSameIntegerType(lhs, rhs, width))
    return UIntType::get(lhs.getContext(), width);
  return {};
}

FIRRTLType firrtl::getCatResult(FIRRTLType lhs, FIRRTLType rhs) {
  if (auto lu = lhs.dyn_cast<UIntType>())
    if (auto ru = rhs.dyn_cast<UIntType>()) {
      int32_t width = -1;
      if (lu.getWidth().hasValue() && ru.getWidth().hasValue())
        width = lu.getWidthOrSentinel() + ru.getWidthOrSentinel();
      return UIntType::get(lhs.getContext(), width);
    }
  if (auto ls = lhs.dyn_cast<SIntType>())
    if (auto rs = rhs.dyn_cast<SIntType>()) {
      int32_t width = -1;
      if (ls.getWidth().hasValue() && rs.getWidth().hasValue())
        width = ls.getWidthOrSentinel() + rs.getWidthOrSentinel();
      return UIntType::get(lhs.getContext(), width);
    }
  return {};
}

FIRRTLType firrtl::getDShlResult(FIRRTLType lhs, FIRRTLType rhs) {
  auto lhsi = lhs.dyn_cast<IntType>();
  auto rhsui = rhs.dyn_cast<UIntType>();
  if (!rhsui || !lhsi)
    return {};

  // If the left or right has unknown result type, then the operation does too.
  auto width = lhsi.getWidthOrSentinel();
  if (width == -1 || !rhsui.getWidth().hasValue())
    width = -1;
  else
    width = width + (1 << rhsui.getWidth().getValue()) - 1;
  return IntType::get(lhs.getContext(), lhsi.isSigned(), width);
}

FIRRTLType firrtl::getDShrResult(FIRRTLType lhs, FIRRTLType rhs) {
  if (!lhs.isa<IntType>() || !rhs.isa<UIntType>())
    return {};
  return lhs;
}

FIRRTLType firrtl::getValidIfResult(FIRRTLType lhs, FIRRTLType rhs) {
  if (!lhs.isa<UIntType>())
    return {};
  auto lhsWidth = lhs.cast<UIntType>().getWidthOrSentinel();
  if (lhsWidth != -1 && lhsWidth != 1)
    return {};
  return rhs;
}

//===----------------------------------------------------------------------===//
// Unary Primitives
//===----------------------------------------------------------------------===//

FIRRTLType firrtl::getAsAsyncResetResult(FIRRTLType input) {
  if (input.isa<UIntType>() || input.isa<SIntType>() || input.isa<ClockType>())
    return AsyncResetType::get(input.getContext());
  return {};
}

FIRRTLType firrtl::getAsClockResult(FIRRTLType input) {
  if (input.isa<UIntType>() || input.isa<SIntType>() || input.isa<ClockType>())
    return ClockType::get(input.getContext());
  return {};
}

FIRRTLType firrtl::getAsSIntResult(FIRRTLType input) {
  if (input.isa<ClockType>() || input.isa<ResetType>() ||
      input.isa<AsyncResetType>())
    return SIntType::get(input.getContext(), 1);
  if (input.isa<SIntType>())
    return input;
  if (auto ui = input.dyn_cast<UIntType>())
    return SIntType::get(input.getContext(), ui.getWidthOrSentinel());
  return {};
}

FIRRTLType firrtl::getAsUIntResult(FIRRTLType input) {
  if (input.isa<ClockType>() || input.isa<ResetType>() ||
      input.isa<AsyncResetType>())
    return UIntType::get(input.getContext(), 1);
  if (input.isa<UIntType>())
    return input;
  if (auto si = input.dyn_cast<SIntType>())
    return UIntType::get(input.getContext(), si.getWidthOrSentinel());
  return {};
}

FIRRTLType firrtl::getCvtResult(FIRRTLType input) {
  if (auto uiType = input.dyn_cast<UIntType>()) {
    auto width = uiType.getWidthOrSentinel();
    if (width != -1)
      ++width;
    return SIntType::get(input.getContext(), width);
  }

  if (input.isa<SIntType>())
    return input;

  return {};
}

FIRRTLType firrtl::getNegResult(FIRRTLType input) {
  auto inputi = input.dyn_cast<IntType>();
  if (!inputi)
    return {};
  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1)
    ++width;
  return SIntType::get(input.getContext(), width);
}

FIRRTLType firrtl::getNotResult(FIRRTLType input) {
  auto inputi = input.dyn_cast<IntType>();
  if (!inputi)
    return {};
  return UIntType::get(input.getContext(), inputi.getWidthOrSentinel());
}

FIRRTLType firrtl::getReductionResult(FIRRTLType input) {
  if (!input.isa<IntType>())
    return {};
  return UIntType::get(input.getContext(), 1);
}

FIRRTLType firrtl::getAsPassiveResult(FIRRTLType input) {
  return input.getPassiveType();
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

FIRRTLType BitsPrimOp::getResultType(FIRRTLType input, int32_t high,
                                     int32_t low) {
  auto inputi = input.dyn_cast<IntType>();

  // High must be >= low and both most be non-negative.
  if (!inputi || high < low || low < 0)
    return {};

  // If the input has staticly known width, check it.  Both and low must be
  // strictly less than width.
  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1 && high >= width)
    return {};

  return UIntType::get(input.getContext(), high - low + 1);
}

void BitsPrimOp::build(OpBuilder &builder, OperationState &result, Value input,
                       unsigned high, unsigned low) {
  auto type = getResultType(input.getType().cast<FIRRTLType>().getPassiveType(),
                            high, low);
  assert(type && "invalid inputs building BitsPrimOp!");
  build(builder, result, type, input, high, low);
}

FIRRTLType HeadPrimOp::getResultType(FIRRTLType input, int32_t amount) {
  auto inputi = input.dyn_cast<IntType>();
  if (amount <= 0 || !inputi)
    return {};

  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1 && amount > width)
    return {};

  width = std::max(width, amount);
  return UIntType::get(input.getContext(), amount);
}

FIRRTLType MuxPrimOp::getResultType(FIRRTLType sel, FIRRTLType high,
                                    FIRRTLType low) {
  // Sel needs to be a one bit uint or an unknown width uint.
  auto selui = sel.dyn_cast<UIntType>();
  if (!selui || selui.getWidthOrSentinel() > 1)
    return {};

  // FIXME: This should be defined in terms of a more general type equivalence
  // operator.  We actually need a 'meet' operator of some sort.
  if (high == low)
    return low;

  // The base types need to be equivalent.
  if (high.getTypeID() != low.getTypeID())
    return {};
  if (low.isa<ClockType>() || low.isa<ResetType>() || low.isa<AsyncResetType>())
    return low;

  // Two different UInt types can be compatible.  If either has unknown width,
  // then return it.  If both are known but different width, then return the
  // larger one.
  if (auto lowui = low.dyn_cast<UIntType>()) {
    if (!lowui.getWidth().hasValue())
      return lowui;
    auto highui = high.cast<UIntType>();
    if (!highui.getWidth().hasValue())
      return highui;
    if (lowui.getWidth().getValue() > highui.getWidth().getValue())
      return low;
    return high;
  }

  if (auto lowsi = low.dyn_cast<SIntType>()) {
    if (!lowsi.getWidth().hasValue())
      return lowsi;
    auto highsi = high.cast<SIntType>();
    if (!highsi.getWidth().hasValue())
      return highsi;
    if (lowsi.getWidth().getValue() > highsi.getWidth().getValue())
      return low;
    return high;
  }

  // FIXME: Should handle bundles and other things.
  return {};
}

FIRRTLType PadPrimOp::getResultType(FIRRTLType input, int32_t amount) {
  auto inputi = input.dyn_cast<IntType>();
  if (amount < 0 || !inputi)
    return {};

  int32_t width = inputi.getWidthOrSentinel();
  if (width == -1)
    return input;

  width = std::max(width, amount);
  return IntType::get(input.getContext(), inputi.isSigned(), width);
}

FIRRTLType ShlPrimOp::getResultType(FIRRTLType input, int32_t amount) {
  auto inputi = input.dyn_cast<IntType>();
  if (amount < 0 || !inputi)
    return {};

  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1)
    width += amount;

  return IntType::get(input.getContext(), inputi.isSigned(), width);
}

FIRRTLType ShrPrimOp::getResultType(FIRRTLType input, int32_t amount) {
  auto inputi = input.dyn_cast<IntType>();
  if (amount < 0 || !inputi)
    return {};

  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1)
    width = std::max(1, width - amount);

  return IntType::get(input.getContext(), inputi.isSigned(), width);
}

FIRRTLType TailPrimOp::getResultType(FIRRTLType input, int32_t amount) {
  auto inputi = input.dyn_cast<IntType>();
  if (amount < 0 || !inputi)
    return {};

  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1) {
    // TODO(firrtl-spec): zero bit integers are not allowed, so the amount
    // cannot equal the width.
    if (width <= amount)
      return {};
    width -= amount;
  }

  return IntType::get(input.getContext(), inputi.isSigned(), width);
}

//===----------------------------------------------------------------------===//
// Conversions to/from fixed-width signless integer types in standard dialect.
//===----------------------------------------------------------------------===//

static LogicalResult verifyStdIntCast(StdIntCast cast) {
  // Either the input or result must have signless standard integer type, the
  // other must be a FIRRTL type that lowers to one, and their widths must
  // match.
  FIRRTLType firType;
  IntegerType integerType;
  if ((firType = cast.getOperand().getType().dyn_cast<FIRRTLType>())) {
    integerType = cast.getType().dyn_cast<IntegerType>();
    if (!integerType) {
      cast.emitError("result type must be a signless integer");
      return failure();
    }
  } else if ((firType = cast.getType().dyn_cast<FIRRTLType>())) {
    integerType = cast.getOperand().getType().dyn_cast<IntegerType>();
    if (!integerType) {
      cast.emitError("operand type must be a signless integer");
      return failure();
    }
  } else {
    cast.emitError("either source or result type must be integer type");
    return failure();
  }

  int32_t intWidth = firType.getBitWidthOrSentinel();
  if (intWidth == -2)
    return cast.emitError("firrtl type isn't simple bit type");
  if (intWidth == -1)
    return cast.emitError("SInt/UInt type must have a width"), failure();
  if (!integerType.isSignless())
    return cast.emitError("standard integer type must be signless"), failure();
  if (unsigned(intWidth) != integerType.getWidth())
    return cast.emitError("source and result width must match"), failure();

  return success();
}

//===----------------------------------------------------------------------===//
// TblGen Generated Logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/FIRRTL/FIRRTL.cpp.inc"
