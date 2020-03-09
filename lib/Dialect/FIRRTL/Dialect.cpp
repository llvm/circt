//===- Dialect.cpp - Implement the FIRRTL dialect -------------------------===//
//
//===----------------------------------------------------------------------===//

#include "spt/Dialect/FIRRTL/IR/Ops.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace spt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// If the specified attribute set contains the firrtl.name attribute, return it.
static StringAttr getFIRRTLNameAttr(ArrayRef<NamedAttribute> attrs) {
  for (auto &argAttr : attrs) {
    if (!argAttr.first.is("firrtl.name"))
      continue;
  
    return argAttr.second.dyn_cast<StringAttr>();
  }
  
  return StringAttr();
}

namespace {

// We implement the OpAsmDialectInterface so that FIRRTL dialect operations
// automatically interpret the firrtl.name attribute on function arguments and
// on operations as their SSA name.
struct FIRRTLOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  
  /// Get a special name to use when printing the given operation. See
  /// OpAsmInterface.td#getAsmResultNames for usage details and documentation.
  void getAsmResultNames(Operation *op,
                         OpAsmSetValueNameFn setNameFn) const override {
    if (op->getNumResults() > 0)
      if (auto nameAttr = op->getAttrOfType<StringAttr>("firrtl.name"))
        setNameFn(op->getResult(0), nameAttr.getValue());
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
      if (auto str = getFIRRTLNameAttr(impl::getArgAttrs(parentOp, i)))
        setNameFn(block->getArgument(i), str.getValue());
    }
  }
};
} // end anonymous namespace

FIRRTLDialect::FIRRTLDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addTypes<UIntType>();
  addOperations<
#define GET_OP_LIST
#include "spt/Dialect/FIRRTL/IR/FIRRTL.cpp.inc"
      >();
      
  addInterfaces<FIRRTLOpAsmDialectInterface>();
}

FIRRTLDialect::~FIRRTLDialect() {
}

//===----------------------------------------------------------------------===//
// Type Implementations.
//===----------------------------------------------------------------------===//

/// Parse a type registered to this dialect.
Type FIRRTLDialect::parseType(DialectAsmParser &parser) const {
  StringRef tyData = parser.getFullSymbolSpec();
  
  if (tyData == "uint")
    return UIntType::get(getContext());
  
  parser.emitError(parser.getNameLoc(), "unknown firrtl type");
  return Type();
}

void FIRRTLDialect::printType(Type type, DialectAsmPrinter &os) const {
  auto uintType = type.dyn_cast<UIntType>();
  assert(uintType && "printing wrong type");
  os.getStream() << "uint";
}

//===----------------------------------------------------------------------===//
// Module and Circuit Ops.
//===----------------------------------------------------------------------===//

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
  if (parser.parseRegion(*body, /*regionArgs*/{}, /*argTypes*/{}))
    return failure();

  CircuitOp::ensureTerminator(*body, parser.getBuilder(), result.location);

  return success();
}


// TODO: This ia a clone of mlir::impl::printFunctionSignature, refactor it to
// allow this customization.
static void printFunctionSignature2(OpAsmPrinter &p, Operation *op,
                                    ArrayRef<Type> argTypes,
                                    bool isVariadic,
                                    ArrayRef<Type> resultTypes) {
  Region &body = op->getRegion(0);
  bool isExternal = body.empty();

  p << '(';
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    if (i > 0)
      p << ", ";

    if (!isExternal) {
      p.printOperand(body.front().getArgument(i));
      p << ": ";
    }

    p.printType(argTypes[i]);
    
    auto argAttrs = ::mlir::impl::getArgAttrs(op, i);
    
    // If the argument has the firrtl.name attribute, and if it was used by the
    // printer exactly (not name mangled with a suffix etc) then we can omit
    // the firrtl.name attribute from the argument attribute dictionary.
    if (auto nameAttr = getFIRRTLNameAttr(argAttrs)) {
      // FIXME: Need a way to verify that nameAttr is matched exactly.  We can't
      // get the "used" name out of OpAsmPrinter.
      p.printOptionalAttrDict(argAttrs, {"firrtl.name"});
    } else {
      p.printOptionalAttrDict(argAttrs);
    }
  }

  if (isVariadic) {
    if (!argTypes.empty())
      p << ", ";
    p << "...";
  }

  p << ')';
}


static void print(OpAsmPrinter &p, FModuleOp op) {
  using namespace mlir::impl;

  FunctionType fnType = op.getType();
  auto argTypes = fnType.getInputs();
  auto resultTypes = fnType.getResults();
  
  // TODO: Should refactor mlir::impl::printFunctionLikeOp to allow these
  // customizations.  Need to not print the terminator.
  
  // Print the operation and the function name.
  auto funcName =
      op.getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName())
          .getValue();
  p << op.getOperationName() << ' ';
  p.printSymbolName(funcName);

  printFunctionSignature2(p, op, argTypes, /*isVariadic*/false, resultTypes);
  printFunctionAttributes(p, op, argTypes.size(), resultTypes.size());

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
  if (parseFunctionSignature(parser, /*allowVariadic*/false, entryArgs, argTypes,
                             argAttrs, isVariadic, resultTypes, resultAttrs))
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
  // text file is a load-bearing part of the IR, but we don't want the verbosity
  // in dumps of including it explicitly in the attribute dictionary.
  for (size_t i = 0, e = argAttrs.size(); i != e; ++i) {
    auto &arg = entryArgs[i];
    auto &attrs = argAttrs[i];
    
    // If an explicit name attribute was present, don't add the implicit one.
    bool hasNameAttr = false;
    for (auto &elt : attrs)
      if (elt.first.str() == "firrtl.name")
        hasNameAttr = true;
    if (hasNameAttr)
      continue;
    
    // The name of an argument is of the form "%42" or "%id", and since parsing
    // succeeded, we know it always has one character.
    assert(arg.name.size() > 1 && arg.name[0] == '%' &&
           "Unknown MLIR name");
    if (isdigit(arg.name[1]))
      continue;
    
    auto nameAttr = StringAttr::get(arg.name.drop_front(), context);
    attrs.push_back({ Identifier::get("firrtl.name", context), nameAttr });
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
#define GET_OP_CLASSES
#include "spt/Dialect/FIRRTL/IR/FIRRTL.cpp.inc"
