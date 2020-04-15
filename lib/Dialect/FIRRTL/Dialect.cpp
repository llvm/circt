//===- Dialect.cpp - Implement the FIRRTL dialect -------------------------===//
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "spt/Dialect/FIRRTL/IR/Ops.h"
#include "spt/Dialect/FIRRTL/IR/Types.h"
#include "llvm/ADT/StringSwitch.h"

using namespace spt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// If the specified attribute set contains the firrtl.name attribute, return it.
static StringAttr getModuleFIRRTLNameAttr(ArrayRef<NamedAttribute> attrs) {
  for (auto &argAttr : attrs) {
    // FIXME: We currently use firrtl.name instead of name because this makes
    // the FunctionLike handling in MLIR core happier.  It otherwise doesn't
    // allow attributes on module parameters.
    if (argAttr.first != "firrtl.name")
      continue;

    return argAttr.second.dyn_cast<StringAttr>();
  }

  return StringAttr();
}

namespace {

// We implement the OpAsmDialectInterface so that FIRRTL dialect operations
// automatically interpret the name attribute on function arguments and
// on operations as their SSA name.
struct FIRRTLOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  /// Get a special name to use when printing the given operation. See
  /// OpAsmInterface.td#getAsmResultNames for usage details and documentation.
  void getAsmResultNames(Operation *op,
                         OpAsmSetValueNameFn setNameFn) const override {
    // Many firrtl dialect operations have an optional 'name' attribute.  If
    // present, use it.
    if (op->getNumResults() > 0)
      if (auto nameAttr = op->getAttrOfType<StringAttr>("name"))
        setNameFn(op->getResult(0), nameAttr.getValue());

    // For constants in particular, propagate the value into the result name to
    // make it easier to read the IR.
    if (auto constant = dyn_cast<ConstantOp>(op)) {
      auto intTy = constant.getType().dyn_cast<IntType>();

      // Otherwise, build a complex name with the value and type.
      SmallString<32> specialNameBuffer;
      llvm::raw_svector_ostream specialName(specialNameBuffer);
      specialName << 'c';
      if (intTy) {
        if (!intTy.isSigned() || !constant.value().isNegative())
          constant.value().print(specialName, /*isSigned:*/ false);
        else {
          specialName << 'm';
          (-constant.value()).print(specialName, /*isSigned:*/ false);
        }

        specialName << (intTy.isSigned() ? "_si" : "_ui");
        auto width = intTy.getWidthOrSentinel();
        if (width != -1)
          specialName << width;
      } else {
        constant.value().print(specialName, /*isSigned:*/ false);
      }
      setNameFn(constant.getResult(), specialName.str());
    }
  }

  /// Get a special name to use when printing the entry block arguments of the
  /// region contained by an operation in this dialect.
  void getAsmBlockArgumentNames(Block *block,
                                OpAsmSetValueNameFn setNameFn) const override {
    // Check to see if the operation containing the arguments has 'firrtl.name'
    // attributes for them.  If so, use that as the name.
    auto *parentOp = block->getParentOp();

    for (size_t i = 0, e = block->getNumArguments(); i != e; ++i) {
      // Scan for a 'firrtl.name' attribute.
      if (auto str = getModuleFIRRTLNameAttr(impl::getArgAttrs(parentOp, i)))
        setNameFn(block->getArgument(i), str.getValue());
    }
  }
};
} // end anonymous namespace

FIRRTLDialect::FIRRTLDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {

  // Register types.
  addTypes<SIntType, UIntType, ClockType, ResetType, AsyncResetType, AnalogType,
           // Derived Types
           FlipType, BundleType, FVectorType>();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "spt/Dialect/FIRRTL/IR/FIRRTL.cpp.inc"
      >();

  // Register interface implementations.
  addInterfaces<FIRRTLOpAsmDialectInterface>();
}

FIRRTLDialect::~FIRRTLDialect() {}

void FIRRTLDialect::printType(Type type, DialectAsmPrinter &os) const {
  type.cast<FIRRTLType>().print(os.getStream());
}

//===----------------------------------------------------------------------===//
// CircuitOp
//===----------------------------------------------------------------------===//

void CircuitOp::build(Builder *builder, OperationState &result,
                      StringAttr name) {
  // Add an attribute for the name.
  result.addAttribute(builder->getIdentifier("name"), name);

  // Create a region and a block for the body.  The argument of the region is
  // the loop induction variable.
  Region *bodyRegion = result.addRegion();
  Block *body = new Block();
  bodyRegion->push_back(body);
  CircuitOp::ensureTerminator(*bodyRegion, *builder, result.location);
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
    results.push_back(
        {getModuleFIRRTLNameAttr(argAttrs), argTypes[i].cast<FIRRTLType>()});
  }
}

static void buildModule(Builder *builder, OperationState &result,
                        StringAttr name,
                        ArrayRef<std::pair<StringAttr, FIRRTLType>> ports) {
  using namespace mlir::impl;

  // Add an attribute for the name.
  result.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), name);

  SmallVector<Type, 4> argTypes;
  for (auto elt : ports)
    argTypes.push_back(elt.second);

  // Record the argument and result types as an attribute.
  auto type = builder->getFunctionType(argTypes, /*resultTypes*/ {});
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

  // Record the names of the arguments if present.
  SmallString<8> attrNameBuf;
  for (size_t i = 0, e = ports.size(); i != e; ++i) {
    if (ports[i].first.getValue().empty())
      continue;

    auto argAttr =
        NamedAttribute(builder->getIdentifier("firrtl.name"), ports[i].first);

    result.addAttribute(getArgAttrName(i, attrNameBuf),
                        builder->getDictionaryAttr(argAttr));
  }

  result.addRegion();
}

void FModuleOp::build(Builder *builder, OperationState &result, StringAttr name,
                      ArrayRef<std::pair<StringAttr, FIRRTLType>> ports) {
  buildModule(builder, result, name, ports);

  // Create a region and a block for the body.
  auto *bodyRegion = result.regions[0].get();
  Block *body = new Block();
  bodyRegion->push_back(body);

  // Add arguments to the body block.
  for (auto elt : ports)
    body->addArgument(elt.second);

  FModuleOp::ensureTerminator(*bodyRegion, *builder, result.location);
}

void FExtModuleOp::build(Builder *builder, OperationState &result,
                         StringAttr name,
                         ArrayRef<std::pair<StringAttr, FIRRTLType>> ports) {
  buildModule(builder, result, name, ports);
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
      if (auto nameAttr = getModuleFIRRTLNameAttr(argAttrs)) {

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

static ParseResult parseFModuleOp(OpAsmParser &parser, OperationState &result) {
  using namespace mlir::impl;

  // TODO: Should refactor mlir::impl::parseFunctionLikeOp to allow these
  // customizations for implicit argument names.  Need to not print the
  // terminator.

  SmallVector<OpAsmParser::OperandType, 4> entryArgs;
  SmallVector<SmallVector<NamedAttribute, 2>, 4> argAttrs;
  SmallVector<SmallVector<NamedAttribute, 2>, 4> resultAttrs;
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

  if (!body->empty())
    FModuleOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

static ParseResult parseFExtModuleOp(OpAsmParser &parser,
                                     OperationState &result) {
  return parseFModuleOp(parser, result);
}
//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

void WhenOp::createElseRegion() {
  assert(!hasElseRegion() && "already has an else region");
  OpBuilder builder(&elseRegion());
  WhenOp::ensureTerminator(elseRegion(), builder, getLoc());
}

void WhenOp::build(Builder *builder, OperationState &result, Value condition,
                   bool withElseRegion) {
  result.addOperands(condition);

  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();
  WhenOp::ensureTerminator(*thenRegion, *builder, result.location);
  if (withElseRegion)
    WhenOp::ensureTerminator(*elseRegion, *builder, result.location);
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

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return valueAttr();
}

/// Build a ConstantOp from an APInt and a FIRRTL type, handling the attribute
/// formation for the 'value' attribute.
void ConstantOp::build(Builder *builder, OperationState &result,
                       FIRRTLType type, const APInt &value,
                       Optional<StringAttr> name) {
  int32_t width = -1;
  IntegerType::SignednessSemantics signedness;

  APInt valueToUse = value;
  if (auto sint = type.dyn_cast<SIntType>()) {
    signedness = IntegerType::Signed;
    width = sint.getWidthOrSentinel();

    // TODO: 1024 bits should be enough for anyone??  The issue here is that
    // the constant type uniquer does not like constants with multiple bitwidth
    // but the same MLIR type.  We can't just use unit type or index type or
    // something like that to represent the unwidth'd case.
    valueToUse = valueToUse.sextOrTrunc(width != -1 ? width : 1024);
  } else {
    assert(type.isa<UIntType>());
    signedness = IntegerType::Unsigned;
    width = type.cast<UIntType>().getWidthOrSentinel();

    // TODO: 1024 bits should be enough for anyone??
    valueToUse = valueToUse.zextOrTrunc(width != -1 ? width : 1024);
  }

  Type attrType =
      IntegerType::get(valueToUse.getBitWidth(), signedness, type.getContext());
  auto attr = builder->getIntegerAttr(attrType, valueToUse);

  auto nameAttr = name.hasValue() ? name.getValue() : StringAttr();
  return build(builder, result, type, attr, nameAttr);
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
  if (!lhsi || lhsi.getKind() != rhs.getKind())
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
  if (lhs.getKind() != rhs.getKind())
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
  if (lhs.getKind() != rhs.getKind())
    return {};

  int32_t width = -1;
  if (auto lu = lhs.dyn_cast<UIntType>()) {
    if (lu.getWidth().hasValue())
      width = lu.getWidth().getValue();
    return UIntType::get(lhs.getContext(), width);
  }
  if (auto ls = lhs.dyn_cast<SIntType>()) {
    if (ls.getWidth().hasValue())
      width = ls.getWidth().getValue() + 1;
    return SIntType::get(lhs.getContext(), width);
  }
  return {};
}

FIRRTLType firrtl::getRemResult(FIRRTLType lhs, FIRRTLType rhs) {
  if (lhs.getKind() != rhs.getKind())
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

FIRRTLType HeadPrimOp::getResultType(FIRRTLType input, int32_t amount) {
  auto inputi = input.dyn_cast<IntType>();
  if (amount < 0 || !inputi)
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
  if (high.getKind() != low.getKind())
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
    if (width < amount)
      return {};
    width -= amount;
  }

  return IntType::get(input.getContext(), inputi.isSigned(), width);
}

#include "spt/Dialect/FIRRTL/IR/FIRRTLEnums.cpp.inc"

#define GET_OP_CLASSES
#include "spt/Dialect/FIRRTL/IR/FIRRTL.cpp.inc"
