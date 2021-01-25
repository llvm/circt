//===- FIRRTLOps.cpp - Implement the FIRRTL operations --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the FIRRTL ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/RTL/RTLTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "llvm/ADT/DenseMap.h"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// VERIFY_RESULT_TYPE / VERIFY_RESULT_TYPE_RET
//===----------------------------------------------------------------------===//

// These macros are used to implement verifier hooks that check that the result
// type of a primitive matches what is returned by the getResultType() static
// method.  This should go away when/if this bug is implemented:
//    https://bugs.llvm.org/show_bug.cgi?id=48645
#define VERIFY_RESULT_TYPE(...)                                                \
  {                                                                            \
    auto resType = getResultType(__VA_ARGS__, getLoc());                       \
    if (!resType)                                                              \
      return failure(); /*already diagnosed the error*/                        \
    if (resType != getType())                                                  \
      return emitOpError("result type should be ") << resType;                 \
  }

// This is the same as VERIFY_RESULT_TYPE but return success if the result type
// matches.  This is useful as the last thing in a verify hook.
#define VERIFY_RESULT_TYPE_RET(...)                                            \
  VERIFY_RESULT_TYPE(__VA_ARGS__);                                             \
  return success();

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

static LogicalResult verifyCircuitOp(CircuitOp circuit) {
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

  // Store a mapping of defname to either the first external module
  // that defines it or, preferentially, the first external module
  // that defines it and has no parameters.
  llvm::DenseMap<Attribute, FExtModuleOp> defnameMap;

  // Verify external modules.
  for (auto &op : *circuit.getBody()) {
    auto extModule = dyn_cast<FExtModuleOp>(op);
    if (!extModule)
      continue;

    auto defname = extModule.defnameAttr();
    if (!defname)
      continue;

    // Check that this extmodule's defname does not conflict with
    // the symbol name of any module.
    auto collidingModule = circuit.lookupSymbol(defname.getValue());
    if (isa_and_nonnull<FModuleOp>(collidingModule)) {
      auto diag =
          op.emitOpError()
          << "attribute 'defname' with value " << defname
          << " conflicts with the name of another module in the circuit";
      diag.attachNote(collidingModule->getLoc())
          << "previous module declared here";
      return failure();
    }

    // Find an optional extmodule with a defname collision. Update
    // the defnameMap if this is the first extmodule with that
    // defname or if the current extmodule takes no parameters and
    // the collision does. The latter condition improves later
    // extmodule verification as checking against a parameterless
    // module is stricter.
    FExtModuleOp collidingExtModule;
    if (auto &value = defnameMap[defname]) {
      collidingExtModule = value;
      if (value.parameters() && !extModule.parameters())
        value = extModule;
    } else {
      value = extModule;
      // Go to the next extmodule if no extmodule with the same
      // defname was found.
      continue;
    }

    // Check that the number of ports is exactly the same.
    SmallVector<ModulePortInfo, 8> ports;
    SmallVector<ModulePortInfo, 8> collidingPorts;
    extModule.getPortInfo(ports);
    collidingExtModule.getPortInfo(collidingPorts);
    if (ports.size() != collidingPorts.size()) {
      auto diag = op.emitOpError()
                  << "with 'defname' attribute " << defname << " has "
                  << ports.size()
                  << " ports which is different from a previously defined "
                     "extmodule with the same 'defname' which has "
                  << collidingPorts.size() << " ports";
      diag.attachNote(collidingExtModule.getLoc())
          << "previous extmodule definition occurred here";
      return failure();
    }

    // Check that ports match for name and type. Since parameters
    // *might* affect widths, ignore widths if either module has
    // parameters. Note that this allows for misdetections, but
    // has zero false positives.
    for (auto p : llvm::zip(ports, collidingPorts)) {
      StringAttr aName = std::get<0>(p).name, bName = std::get<1>(p).name;
      FIRRTLType aType = std::get<0>(p).type, bType = std::get<1>(p).type;

      if (extModule.parameters() || collidingExtModule.parameters()) {
        aType = aType.getWidthlessType();
        bType = bType.getWidthlessType();
      }
      if (aName != bName) {
        auto diag = op.emitOpError()
                    << "with 'defname' attribute " << defname
                    << " has a port with name " << aName
                    << " which does not match the name of the port "
                    << "in the same position of a previously defined "
                    << "extmodule with the same 'defname', expected port "
                       "to have name "
                    << bName;
        diag.attachNote(collidingExtModule.getLoc())
            << "previous extmodule definition occurred here";
        return failure();
      }
      if (aType != bType) {
        auto diag = op.emitOpError()
                    << "with 'defname' attribute " << defname
                    << " has a port with name " << aName
                    << " which has a different type " << aType
                    << " which does not match the type of the port in "
                       "the same position of a previously defined "
                       "extmodule with the same 'defname', expected port "
                       "to have type "
                    << bType;
        diag.attachNote(collidingExtModule.getLoc())
            << "previous extmodule definition occurred here";
        return failure();
      }
    }
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
    auto type = argTypes[i].cast<FIRRTLType>();
    results.push_back({getFIRRTLNameAttr(argAttrs), type});
  }
}

static void buildModule(OpBuilder &builder, OperationState &result,
                        StringAttr name, ArrayRef<ModulePortInfo> ports) {
  using namespace mlir::impl;

  // Add an attribute for the name.
  result.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), name);

  SmallVector<Type, 4> argTypes;
  for (auto elt : ports)
    argTypes.push_back(elt.type);

  // Record the argument and result types as an attribute.
  auto type = builder.getFunctionType(argTypes, /*resultTypes*/ {});
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

  // Record the names of the arguments if present.
  SmallString<8> attrNameBuf;
  for (size_t i = 0, e = ports.size(); i != e; ++i) {
    if (ports[i].getName().empty())
      continue;

    auto argAttr =
        NamedAttribute(builder.getIdentifier("firrtl.name"), ports[i].name);

    result.addAttribute(getArgAttrName(i, attrNameBuf),
                        builder.getDictionaryAttr(argAttr));
  }

  result.addRegion();
}

void FModuleOp::build(OpBuilder &builder, OperationState &result,
                      StringAttr name, ArrayRef<ModulePortInfo> ports) {
  buildModule(builder, result, name, ports);

  // Create a region and a block for the body.
  auto *bodyRegion = result.regions[0].get();
  Block *body = new Block();
  bodyRegion->push_back(body);

  // Add arguments to the body block.
  for (auto elt : ports)
    body->addArgument(elt.type);

  FModuleOp::ensureTerminator(*bodyRegion, builder, result.location);
}

void FExtModuleOp::build(OpBuilder &builder, OperationState &result,
                         StringAttr name, ArrayRef<ModulePortInfo> ports,
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
  if (!isExtModule) {
    if (parser.parseRegion(*body, entryArgs,
                           entryArgs.empty() ? ArrayRef<Type>() : argTypes))
      return failure();

    FModuleOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  }
  return success();
}

static ParseResult parseFExtModuleOp(OpAsmParser &parser,
                                     OperationState &result) {
  return parseFModuleOp(parser, result, /*isExtModule:*/ true);
}

static LogicalResult verifyModuleSignature(Operation *op) {
  for (auto argType : getModuleType(op).getInputs())
    if (!argType.isa<FIRRTLType>())
      return op->emitOpError("all module ports must be firrtl types");
  return success();
}

static LogicalResult verifyFModuleOp(FModuleOp op) {
  // The parent op must be a circuit op.
  auto parentOp = dyn_cast_or_null<CircuitOp>(op->getParentOp());
  if (!parentOp)
    return op.emitOpError("should be embedded into a 'firrtl.circuit'");

  // Verify the module signature.
  return verifyModuleSignature(op);
}

static LogicalResult verifyFExtModuleOp(FExtModuleOp op) {
  // Verify the module signature.
  if (failed(verifyModuleSignature(op)))
    return failure();

  auto paramDictOpt = op.parameters();
  if (!paramDictOpt)
    return success();

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

  return success();
}

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

/// Lookup the module or extmodule for the symbol.  This returns null on
/// invalid IR.
Operation *InstanceOp::getReferencedModule() {
  auto circuit = (*this)->getParentOfType<CircuitOp>();
  if (!circuit)
    return nullptr;

  return circuit.lookupSymbol(moduleName());
}

StringAttr InstanceOp::getPortName(size_t resultNo) {
  return portNames()[resultNo].cast<StringAttr>();
}

Value InstanceOp::getPortNamed(StringRef name) {
  auto namesArray = portNames();
  for (size_t i = 0, e = namesArray.size(); i != e; ++i) {
    if (namesArray[i].cast<StringAttr>().getValue() == name) {
      assert(i < getNumResults() && " names array out of sync with results");
      return getResult(i);
    }
  }
  return Value();
}

Value InstanceOp::getPortNamed(StringAttr name) {
  auto namesArray = portNames();
  for (size_t i = 0, e = namesArray.size(); i != e; ++i) {
    if (namesArray[i] == name) {
      assert(i < getNumResults() && " names array out of sync with results");
      return getResult(i);
    }
  }
  return Value();
}

/// Verify the correctness of an InstanceOp.
static LogicalResult verifyInstanceOp(InstanceOp instance) {

  // Check that this instance is inside a module.
  auto module = instance->getParentOfType<FModuleOp>();
  if (!module) {
    instance.emitOpError("should be embedded in a 'firrtl.module'");
    return failure();
  }

  auto *referencedModule = instance.getReferencedModule();
  if (!referencedModule) {
    instance.emitOpError("invalid symbol reference");
    return failure();
  }

  // Check that this instance doesn't recursively instantiate its wrapping
  // module.
  if (referencedModule == module) {
    auto diag = instance.emitOpError()
                << "is a recursive instantiation of its containing module";
    diag.attachNote(module.getLoc()) << "containing module declared here";
    return failure();
  }

  SmallVector<ModulePortInfo> modulePorts;
  getModulePortInfo(referencedModule, modulePorts);

  // Check that result types are consistent with the referenced module's ports.
  size_t numResults = instance.getNumResults();
  if (numResults != modulePorts.size()) {
    auto diag = instance.emitOpError()
                << "has a wrong number of results; expected "
                << modulePorts.size() << " but got " << numResults;
    diag.attachNote(referencedModule->getLoc())
        << "original module declared here";
    return failure();
  }

  // Check that the names array is the right length.
  if (instance.portNames().size() != instance.getNumResults()) {
    instance.emitOpError("incorrect number of port names");
    return failure();
  }

  for (size_t i = 0; i != numResults; i++) {
    auto result = instance.getPortNamed(modulePorts[i].name);
    if (!result) {
      auto diag = instance.emitOpError()
                  << "is missing a port named '" << modulePorts[i].name
                  << "' expected by referenced module";
      diag.attachNote(referencedModule->getLoc())
          << "original module declared here";
      return failure();
    }

    auto resultType = result.getType();
    auto expectedType = FlipType::get(modulePorts[i].type);
    if (resultType != expectedType) {
      auto diag = instance.emitOpError()
                  << "result type for " << modulePorts[i].name << " must be "
                  << expectedType << ", but got " << resultType;

      diag.attachNote(referencedModule->getLoc())
          << "original module declared here";
      return failure();
    }
  }

  return success();
}

/// Return the type of a mem given a list of named ports and their kind.
/// This returns a null type if there are duplicate port names.
BundleType
MemOp::getTypeForPortList(uint64_t depth, FIRRTLType dataType,
                          ArrayRef<std::pair<Identifier, PortKind>> portList) {
  assert(dataType.isPassive() && "mem can only have passive datatype");

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
  auto addressType =
      UIntType::get(context, std::max(1U, llvm::Log2_64_Ceil(depth)));

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

    memFields.push_back(
        {port.first, FlipType::get(BundleType::get(portFields, context))});
  }

  return BundleType::get(memFields, context).cast<BundleType>();
}

/// Return the kind of port this is given the port type from a 'mem' decl.
static Optional<MemOp::PortKind> getMemPortKindFromType(FIRRTLType type) {
  auto portType = type.dyn_cast<BundleType>();
  if (!portType) {
    if (auto flipType = type.dyn_cast<FlipType>())
      portType = flipType.getElementType().dyn_cast<BundleType>();
    if (!portType)
      return None;
  }
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
  // Each entry in the bundle is a port.
  for (size_t i = 0, e = getNumResults(); i != e; ++i) {
    auto elt = getResult(i);
    // Each port is a bundle.
    auto kind = getMemPortKindFromType(elt.getType().cast<FIRRTLType>());
    assert(kind.hasValue() && "unknown port type!");
    result.push_back({Identifier::get(getPortNameStr(i), elt.getContext()),
                      kind.getValue()});
  }
}

/// Return the kind of the specified port or None if the name is invalid.
Optional<MemOp::PortKind> MemOp::getPortKind(StringRef portName) {
  auto elt = getPortNamed(portName);
  if (!elt)
    return None;
  return getMemPortKindFromType(elt.getType().cast<FIRRTLType>());
}

/// Return the data-type field of the memory, the type of each element.
FIRRTLType MemOp::getDataTypeOrNull() {
  // Mems with no read/write ports are legal.
  if (getNumResults() == 0)
    return {};

  return getResult(0)
      .getType()
      .cast<FIRRTLType>()
      .getPassiveType()
      .cast<BundleType>()
      .getElementType("data");
}

StringAttr MemOp::getPortName(size_t resultNo) {
  return portNames()[resultNo].cast<StringAttr>();
}

Value MemOp::getPortNamed(StringRef name) {
  auto namesArray = portNames();
  for (size_t i = 0, e = namesArray.size(); i != e; ++i) {
    if (namesArray[i].cast<StringAttr>().getValue() == name) {
      assert(i < getNumResults() && " names array out of sync with results");
      return getResult(i);
    }
  }
  return Value();
}

Value MemOp::getPortNamed(StringAttr name) {
  auto namesArray = portNames();
  for (size_t i = 0, e = namesArray.size(); i != e; ++i) {
    if (namesArray[i] == name) {
      assert(i < getNumResults() && " names array out of sync with results");
      return getResult(i);
    }
  }
  return Value();
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

static LogicalResult verifyConnectOp(ConnectOp connect) {
  FIRRTLType destType =
      connect.dest().getType().cast<FIRRTLType>().getPassiveType();
  FIRRTLType srcType =
      connect.src().getType().cast<FIRRTLType>().getPassiveType();

  // Analog types cannot be connected and must be attached.
  if (destType.isa<AnalogType>() || srcType.isa<AnalogType>())
    return connect.emitError("analog types may not be connected");

  // Destination and source types must be equivalent.
  if (!areTypesEquivalent(destType, srcType))
    return connect.emitError("type mismatch between destination ")
           << destType << " and source " << srcType;

  // Destination bitwidth must be greater than or equal to source bitwidth.
  int32_t destWidth = destType.getBitWidthOrSentinel();
  int32_t srcWidth = srcType.getBitWidthOrSentinel();
  if (destWidth > -1 && srcWidth > -1 && destWidth < srcWidth)
    return connect.emitError("destination width ")
           << destWidth << " is not greater than or equal to source width "
           << srcWidth;

  // TODO(mikeurbach): verify destination flow is sink or duplex.
  // TODO(mikeurbach): verify source flow is source or duplex.
  return success();
}

void WhenOp::createElseRegion() {
  assert(!hasElseRegion() && "already has an else region");
  OpBuilder builder(&elseRegion());
  WhenOp::ensureTerminator(elseRegion(), builder, getLoc());
}

void WhenOp::build(OpBuilder &builder, OperationState &result, Value condition,
                   bool withElseRegion, std::function<void()> thenCtor,
                   std::function<void()> elseCtor) {
  result.addOperands(condition);

  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();
  WhenOp::ensureTerminator(*thenRegion, builder, result.location);

  if (thenCtor) {
    auto oldIP = &*builder.getInsertionPoint();
    builder.setInsertionPointToStart(&*thenRegion->begin());
    thenCtor();
    builder.setInsertionPoint(oldIP);
  }

  if (withElseRegion) {
    WhenOp::ensureTerminator(*elseRegion, builder, result.location);

    if (elseCtor) {
      auto oldIP = &*builder.getInsertionPoint();
      builder.setInsertionPointToStart(&*elseRegion->begin());
      elseCtor();
      builder.setInsertionPoint(oldIP);
    }
  }
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
  if (width != -1 && (int)constant.value().getBitWidth() != width)
    return constant.emitError(
        "firrtl.constant attribute bitwidth doesn't match return type");

  return success();
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
      IntegerType::get(type.getContext(), value.getBitWidth(), signedness);
  auto attr = builder.getIntegerAttr(attrType, value);
  return build(builder, result, type, attr);
}

// Return the result of a subfield operation.
FIRRTLType SubfieldOp::getResultType(FIRRTLType inType, StringRef fieldName,
                                     Location loc) {
  if (auto bundleType = inType.dyn_cast<BundleType>()) {
    for (auto &elt : bundleType.getElements()) {
      if (elt.name == fieldName)
        return elt.type;
    }
    mlir::emitError(loc, "unknown field '")
        << fieldName << "' in bundle type " << inType;
    return {};
  }

  if (auto flipType = inType.dyn_cast<FlipType>())
    if (auto subType = getResultType(flipType.getElementType(), fieldName, loc))
      return FlipType::get(subType);

  mlir::emitError(loc, "subfield requires bundle operand");
  return {};
}

FIRRTLType SubindexOp::getResultType(FIRRTLType inType, unsigned fieldIdx,
                                     Location loc) {
  if (auto vectorType = inType.dyn_cast<FVectorType>()) {
    if (fieldIdx < vectorType.getNumElements())
      return vectorType.getElementType();
    mlir::emitError(loc, "out of range index '")
        << fieldIdx << "' in vector type " << inType;
    return {};
  }

  if (auto flipType = inType.dyn_cast<FlipType>())
    if (auto subType = getResultType(flipType.getElementType(), fieldIdx, loc))
      return FlipType::get(subType);

  mlir::emitError(loc, "subindex requires vector operand");
  return {};
}

FIRRTLType SubaccessOp::getResultType(FIRRTLType inType, FIRRTLType indexType,
                                      Location loc) {
  if (!indexType.isa<UIntType>()) {
    mlir::emitError(loc, "subaccess index must be UInt type, not ")
        << indexType;
    return {};
  }

  if (auto vectorType = inType.dyn_cast<FVectorType>())
    return vectorType.getElementType();

  if (auto flipType = inType.dyn_cast<FlipType>())
    if (auto subType = getResultType(flipType.getElementType(), indexType, loc))
      return FlipType::get(subType);

  mlir::emitError(loc, "subaccess requires vector operand, not ") << inType;
  return {};
}

//===----------------------------------------------------------------------===//
// Binary Primitives
//===----------------------------------------------------------------------===//

/// If LHS and RHS are both UInt or SInt types, the return true and fill in the
/// width of them if known.  If unknown, return -1 for the widths.
///
/// On failure, this reports and error and returns false.  This function should
/// not be used if you don't want an error reported.
static bool isSameIntTypeKind(FIRRTLType lhs, FIRRTLType rhs, int32_t &lhsWidth,
                              int32_t &rhsWidth, Location loc) {
  // Must have two integer types with the same signedness.
  auto lhsi = lhs.dyn_cast<IntType>();
  auto rhsi = rhs.dyn_cast<IntType>();
  if (!lhsi || !rhsi || lhsi.isSigned() != rhsi.isSigned()) {
    if (lhsi && !rhsi)
      mlir::emitError(loc, "second operand must be an integer type, not ")
          << rhs;
    else if (!lhsi && rhsi)
      mlir::emitError(loc, "first operand must be an integer type, not ")
          << lhs;
    else if (!lhsi && !rhsi)
      mlir::emitError(loc, "operands must be integer types, not ")
          << lhs << " and " << rhs;
    else
      mlir::emitError(loc, "operand signedness must match");
    return false;
  }

  lhsWidth = lhsi.getWidthOrSentinel();
  rhsWidth = rhs.cast<IntType>().getWidthOrSentinel();
  return true;
}

static FIRRTLType getAddSubResult(FIRRTLType lhs, FIRRTLType rhs,
                                  Location loc) {
  int32_t lhsWidth, rhsWidth, resultWidth = -1;
  if (!isSameIntTypeKind(lhs, rhs, lhsWidth, rhsWidth, loc))
    return {};

  if (lhsWidth != -1 && rhsWidth != -1)
    resultWidth = std::max(lhsWidth, rhsWidth) + 1;
  return IntType::get(lhs.getContext(), lhs.isa<SIntType>(), resultWidth);
}

FIRRTLType AddPrimOp::getResultType(FIRRTLType lhs, FIRRTLType rhs,
                                    Location loc) {
  return getAddSubResult(lhs, rhs, loc);
}

FIRRTLType SubPrimOp::getResultType(FIRRTLType lhs, FIRRTLType rhs,
                                    Location loc) {
  return getAddSubResult(lhs, rhs, loc);
}

FIRRTLType MulPrimOp::getResultType(FIRRTLType lhs, FIRRTLType rhs,
                                    Location loc) {
  int32_t lhsWidth, rhsWidth, resultWidth = -1;
  if (!isSameIntTypeKind(lhs, rhs, lhsWidth, rhsWidth, loc))
    return {};

  if (lhsWidth != -1 && rhsWidth != -1)
    resultWidth = lhsWidth + rhsWidth;

  return IntType::get(lhs.getContext(), lhs.isa<SIntType>(), resultWidth);
}

FIRRTLType DivPrimOp::getResultType(FIRRTLType lhs, FIRRTLType rhs,
                                    Location loc) {
  int32_t lhsWidth, rhsWidth;
  if (!isSameIntTypeKind(lhs, rhs, lhsWidth, rhsWidth, loc))
    return {};

  // For unsigned, the width is the width of the numerator on the LHS.
  if (lhs.isa<UIntType>())
    return UIntType::get(lhs.getContext(), lhsWidth);

  // For signed, the width is the width of the numerator on the LHS, plus 1.
  int32_t resultWidth = lhsWidth != -1 ? lhsWidth + 1 : -1;
  return SIntType::get(lhs.getContext(), resultWidth);
}

FIRRTLType RemPrimOp::getResultType(FIRRTLType lhs, FIRRTLType rhs,
                                    Location loc) {
  int32_t lhsWidth, rhsWidth, resultWidth = -1;
  if (!isSameIntTypeKind(lhs, rhs, lhsWidth, rhsWidth, loc))
    return {};

  if (lhsWidth != -1 && rhsWidth != -1)
    resultWidth = std::min(lhsWidth, rhsWidth);
  return IntType::get(lhs.getContext(), lhs.isa<SIntType>(), resultWidth);
}

static FIRRTLType getBitwiseBinaryResult(FIRRTLType lhs, FIRRTLType rhs,
                                         Location loc) {
  int32_t lhsWidth, rhsWidth, resultWidth = -1;
  if (!isSameIntTypeKind(lhs, rhs, lhsWidth, rhsWidth, loc))
    return {};

  if (lhsWidth != -1 && rhsWidth != -1)
    resultWidth = std::max(lhsWidth, rhsWidth);
  return UIntType::get(lhs.getContext(), resultWidth);
}

FIRRTLType AndPrimOp::getResultType(FIRRTLType lhs, FIRRTLType rhs,
                                    Location loc) {
  return getBitwiseBinaryResult(lhs, rhs, loc);
}
FIRRTLType OrPrimOp::getResultType(FIRRTLType lhs, FIRRTLType rhs,
                                   Location loc) {
  return getBitwiseBinaryResult(lhs, rhs, loc);
}
FIRRTLType XorPrimOp::getResultType(FIRRTLType lhs, FIRRTLType rhs,
                                    Location loc) {
  return getBitwiseBinaryResult(lhs, rhs, loc);
}

static FIRRTLType getCompareResult(FIRRTLType lhs, FIRRTLType rhs,
                                   Location loc) {
  int32_t lhsWidth, rhsWidth;
  if (!isSameIntTypeKind(lhs, rhs, lhsWidth, rhsWidth, loc))
    return {};

  return UIntType::get(lhs.getContext(), 1);
}

FIRRTLType LEQPrimOp::getResultType(FIRRTLType lhs, FIRRTLType rhs,
                                    Location loc) {
  return getCompareResult(lhs, rhs, loc);
}
FIRRTLType LTPrimOp::getResultType(FIRRTLType lhs, FIRRTLType rhs,
                                   Location loc) {
  return getCompareResult(lhs, rhs, loc);
}
FIRRTLType GEQPrimOp::getResultType(FIRRTLType lhs, FIRRTLType rhs,
                                    Location loc) {
  return getCompareResult(lhs, rhs, loc);
}
FIRRTLType GTPrimOp::getResultType(FIRRTLType lhs, FIRRTLType rhs,
                                   Location loc) {
  return getCompareResult(lhs, rhs, loc);
}
FIRRTLType EQPrimOp::getResultType(FIRRTLType lhs, FIRRTLType rhs,
                                   Location loc) {
  return getCompareResult(lhs, rhs, loc);
}
FIRRTLType NEQPrimOp::getResultType(FIRRTLType lhs, FIRRTLType rhs,
                                    Location loc) {
  return getCompareResult(lhs, rhs, loc);
}

FIRRTLType CatPrimOp::getResultType(FIRRTLType lhs, FIRRTLType rhs,
                                    Location loc) {
  int32_t lhsWidth, rhsWidth, resultWidth = -1;
  if (!isSameIntTypeKind(lhs, rhs, lhsWidth, rhsWidth, loc))
    return {};

  if (lhsWidth != -1 && rhsWidth != -1)
    resultWidth = lhsWidth + rhsWidth;
  return UIntType::get(lhs.getContext(), resultWidth);
}

FIRRTLType DShlPrimOp::getResultType(FIRRTLType lhs, FIRRTLType rhs,
                                     Location loc) {
  auto lhsi = lhs.dyn_cast<IntType>();
  auto rhsui = rhs.dyn_cast<UIntType>();
  if (!rhsui || !lhsi) {
    mlir::emitError(loc,
                    "first operand should be integer, second unsigned int");
    return {};
  }

  // If the left or right has unknown result type, then the operation does too.
  auto width = lhsi.getWidthOrSentinel();
  if (width == -1 || !rhsui.getWidth().hasValue())
    width = -1;
  else
    width = width + (1 << rhsui.getWidth().getValue()) - 1;
  return IntType::get(lhs.getContext(), lhsi.isSigned(), width);
}

FIRRTLType DShlwPrimOp::getResultType(FIRRTLType lhs, FIRRTLType rhs,
                                      Location loc) {
  if (!lhs.isa<IntType>() || !rhs.isa<UIntType>()) {
    mlir::emitError(loc,
                    "first operand should be integer, second unsigned int");
    return {};
  }
  return lhs;
}

FIRRTLType DShrPrimOp::getResultType(FIRRTLType lhs, FIRRTLType rhs,
                                     Location loc) {
  if (!lhs.isa<IntType>() || !rhs.isa<UIntType>()) {
    mlir::emitError(loc,
                    "first operand should be integer, second unsigned int");
    return {};
  }
  return lhs;
}

FIRRTLType ValidIfPrimOp::getResultType(FIRRTLType lhs, FIRRTLType rhs,
                                        Location loc) {
  auto lhsUInt = lhs.dyn_cast<UIntType>();
  if (!lhsUInt) {
    mlir::emitError(loc, "first operand should have UInt type");
    return {};
  }
  auto lhsWidth = lhsUInt.getWidthOrSentinel();
  if (lhsWidth != -1 && lhsWidth != 1) {
    mlir::emitError(loc, "first operand should have 'uint<1>' type");
    return {};
  }
  return rhs;
}

//===----------------------------------------------------------------------===//
// Unary Primitives
//===----------------------------------------------------------------------===//

FIRRTLType AsSIntPrimOp::getResultType(FIRRTLType input, Location loc) {
  int32_t width = input.getBitWidthOrSentinel();
  if (width == -2) {
    mlir::emitError(loc, "operand must be a scalar type");
    return {};
  }

  return SIntType::get(input.getContext(), width);
}

FIRRTLType AsUIntPrimOp::getResultType(FIRRTLType input, Location loc) {
  int32_t width = input.getBitWidthOrSentinel();
  if (width == -2) {
    mlir::emitError(loc, "operand must be a scalar type");
    return {};
  }

  return UIntType::get(input.getContext(), width);
}

FIRRTLType AsAsyncResetPrimOp::getResultType(FIRRTLType input, Location loc) {
  int32_t width = input.getBitWidthOrSentinel();
  if (width == -2 || width == 0 || width > 1) {
    mlir::emitError(loc, "operand must be single bit scalar type");
    return {};
  }
  return AsyncResetType::get(input.getContext());
}

FIRRTLType AsClockPrimOp::getResultType(FIRRTLType input, Location loc) {
  int32_t width = input.getBitWidthOrSentinel();
  if (width == -2 || width == 0 || width > 1) {
    mlir::emitError(loc, "operand must be single bit scalar type");
    return {};
  }
  return ClockType::get(input.getContext());
}

FIRRTLType CvtPrimOp::getResultType(FIRRTLType input, Location loc) {
  if (auto uiType = input.dyn_cast<UIntType>()) {
    auto width = uiType.getWidthOrSentinel();
    if (width != -1)
      ++width;
    return SIntType::get(input.getContext(), width);
  }

  if (input.isa<SIntType>())
    return input;

  mlir::emitError(loc, "operand must have integer type");
  return {};
}

FIRRTLType NegPrimOp::getResultType(FIRRTLType input, Location loc) {
  auto inputi = input.dyn_cast<IntType>();
  if (!inputi) {
    mlir::emitError(loc, "operand must have integer type");

    return {};
  }
  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1)
    ++width;
  return SIntType::get(input.getContext(), width);
}

FIRRTLType NotPrimOp::getResultType(FIRRTLType input, Location loc) {
  auto inputi = input.dyn_cast<IntType>();
  if (!inputi) {
    mlir::emitError(loc, "operand must have integer type");

    return {};
  }
  return UIntType::get(input.getContext(), inputi.getWidthOrSentinel());
}

static FIRRTLType getReductionResult(FIRRTLType input, Location loc) {
  if (!input.isa<IntType>()) {
    mlir::emitError(loc, "operand must have integer type");
    return {};
  }
  return UIntType::get(input.getContext(), 1);
}

FIRRTLType AndRPrimOp::getResultType(FIRRTLType input, Location loc) {
  return getReductionResult(input, loc);
}
FIRRTLType OrRPrimOp::getResultType(FIRRTLType input, Location loc) {
  return getReductionResult(input, loc);
}
FIRRTLType XorRPrimOp::getResultType(FIRRTLType input, Location loc) {
  return getReductionResult(input, loc);
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

FIRRTLType BitsPrimOp::getResultType(FIRRTLType input, int32_t high,
                                     int32_t low, Location loc) {
  auto inputi = input.dyn_cast<IntType>();

  if (!inputi) {
    mlir::emitError(loc, "input type should be the int type but got ") << input;
    return {};
  }

  // High must be >= low and both most be non-negative.
  if (high < low) {
    mlir::emitError(loc,
                    "high must be equal or greater than low, but got high = ")
        << high << ", low = " << low;
    return {};
  }

  if (low < 0) {
    mlir::emitError(loc, "low must be non-negative but got ") << low;
    return {};
  }

  // If the input has staticly known width, check it.  Both and low must be
  // strictly less than width.
  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1 && high >= width) {
    mlir::emitError(loc)
        << "high must be smaller than the width of input, but got high = "
        << high << ", width = " << width;
    return {};
  }

  return UIntType::get(input.getContext(), high - low + 1);
}

void BitsPrimOp::build(OpBuilder &builder, OperationState &result, Value input,
                       unsigned high, unsigned low) {
  auto type = getResultType(input.getType().cast<FIRRTLType>(), high, low,
                            result.location);
  assert(type && "invalid inputs building BitsPrimOp!");
  build(builder, result, type, input, high, low);
}

FIRRTLType HeadPrimOp::getResultType(FIRRTLType input, int32_t amount,
                                     Location loc) {
  auto inputi = input.dyn_cast<IntType>();
  if (amount < 0 || !inputi) {
    mlir::emitError(loc,
                    "operand must have integer type and amount must be >= 0");
    return {};
  }

  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1 && amount > width) {
    mlir::emitError(loc, "amount larger than input width");
    return {};
  }

  width = std::max(width, amount);
  return UIntType::get(input.getContext(), amount);
}

FIRRTLType MuxPrimOp::getResultType(FIRRTLType sel, FIRRTLType high,
                                    FIRRTLType low, Location loc) {
  // Sel needs to be a one bit uint or an unknown width uint.
  auto selui = sel.dyn_cast<UIntType>();
  int32_t selWidth = selui.getBitWidthOrSentinel();
  if (!selui || selWidth == 0 || selWidth > 1) {
    mlir::emitError(loc, "selector must be UInt or UInt<1>");
    return {};
  }

  // TODO: Should use a more general type equivalence operator.
  if (high == low)
    return low;

  // The base types need to be equivalent.
  if (high.getTypeID() != low.getTypeID()) {
    mlir::emitError(loc, "true and false value must have same type");
    return {};
  }

  if (low.isa<IntType>()) {
    // Two different Int types can be compatible.  If either has unknown width,
    // then return it.  If both are known but different width, then return the
    // larger one.
    int32_t highWidth = high.getBitWidthOrSentinel();
    int32_t lowWidth = low.getBitWidthOrSentinel();
    if (lowWidth == -1)
      return low;
    if (highWidth == -1)
      return high;
    return lowWidth > highWidth ? low : high;
  }

  // FIXME: Should handle bundles and other things.
  mlir::emitError(loc, "unknown types to mux");
  return {};
}

FIRRTLType PadPrimOp::getResultType(FIRRTLType input, int32_t amount,
                                    Location loc) {
  auto inputi = input.dyn_cast<IntType>();
  if (amount < 0 || !inputi) {
    mlir::emitError(loc, "input must be integer and amount must be >= 0");
    return {};
  }

  int32_t width = inputi.getWidthOrSentinel();
  if (width == -1)
    return input;

  width = std::max(width, amount);
  return IntType::get(input.getContext(), inputi.isSigned(), width);
}

FIRRTLType ShlPrimOp::getResultType(FIRRTLType input, int32_t amount,
                                    Location loc) {
  auto inputi = input.dyn_cast<IntType>();
  if (amount < 0 || !inputi) {
    mlir::emitError(loc, "input must be integer and amount must be >= 0");
    return {};
  }

  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1)
    width += amount;

  return IntType::get(input.getContext(), inputi.isSigned(), width);
}

FIRRTLType ShrPrimOp::getResultType(FIRRTLType input, int32_t amount,
                                    Location loc) {
  auto inputi = input.dyn_cast<IntType>();
  if (amount < 0 || !inputi) {
    mlir::emitError(loc, "input must be integer and amount must be >= 0");
    return {};
  }

  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1)
    width = std::max(1, width - amount);

  return IntType::get(input.getContext(), inputi.isSigned(), width);
}

FIRRTLType TailPrimOp::getResultType(FIRRTLType input, int32_t amount,
                                     Location loc) {
  auto inputi = input.dyn_cast<IntType>();
  if (amount < 0 || !inputi) {
    mlir::emitError(loc, "input must be integer and amount must be >= 0");
    return {};
  }

  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1) {
    if (width < amount) {
      mlir::emitError(loc, "amount must be less than or equal operand width");
      return {};
    }
    width -= amount;
  }

  return IntType::get(input.getContext(), false, width);
}

//===----------------------------------------------------------------------===//
// Conversions to/from fixed-width signless integer types in standard dialect.
//===----------------------------------------------------------------------===//

static LogicalResult verifyStdIntCastOp(StdIntCastOp cast) {
  // Either the input or result must have signless standard integer type, the
  // other must be a FIRRTL type that lowers to one, and their widths must
  // match.
  FIRRTLType firType;
  IntegerType integerType;
  if ((firType = cast.getOperand().getType().dyn_cast<FIRRTLType>())) {
    integerType = cast.getType().dyn_cast<IntegerType>();
    if (!integerType)
      return cast.emitError("result type must be a signless integer");
  } else if ((firType = cast.getType().dyn_cast<FIRRTLType>())) {
    integerType = cast.getOperand().getType().dyn_cast<IntegerType>();
    if (!integerType)
      return cast.emitError("operand type must be a signless integer");
  } else {
    return cast.emitError("either source or result type must be integer type");
  }

  int32_t intWidth = firType.getBitWidthOrSentinel();
  if (intWidth == -2)
    return cast.emitError("firrtl type isn't simple bit type");
  if (intWidth == -1)
    return cast.emitError("SInt/UInt type must have a width");
  if (!integerType.isSignless())
    return cast.emitError("standard integer type must be signless");
  if (unsigned(intWidth) != integerType.getWidth())
    return cast.emitError("source and result width must match");

  return success();
}

static LogicalResult verifyAnalogInOutCastOp(AnalogInOutCastOp cast) {
  AnalogType firType;
  rtl::InOutType inoutType;

  if ((firType = cast.getOperand().getType().dyn_cast<AnalogType>())) {
    inoutType = cast.getType().dyn_cast<rtl::InOutType>();
    if (!inoutType)
      return cast.emitError("result type must be an inout type");
  } else if ((firType = cast.getType().dyn_cast<AnalogType>())) {
    inoutType = cast.getOperand().getType().dyn_cast<rtl::InOutType>();
    if (!inoutType)
      return cast.emitError("operand type must be an inout type");
  } else {
    return cast.emitError("either source or result type must be analog type");
  }

  // The inout type must wrap an integer.
  auto integerType = inoutType.getElementType().dyn_cast<IntegerType>();
  if (!integerType)
    return cast.emitError("inout type must wrap an integer");

  int32_t width = firType.getBitWidthOrSentinel();
  if (width == -2)
    return cast.emitError("firrtl type isn't simple bit type");
  if (width == -1)
    return cast.emitError("Analog type must have a width");
  if (!integerType.isSignless())
    return cast.emitError("standard integer type must be signless");
  if (unsigned(width) != integerType.getWidth())
    return cast.emitError("source and result width must match");

  return success();
}

void AsPassivePrimOp::build(OpBuilder &builder, OperationState &result,
                            Value input) {
  result.addOperands(input);
  result.addTypes(input.getType().cast<FIRRTLType>().getPassiveType());
}

//===----------------------------------------------------------------------===//
// ImplicitSSAName Custom Directive
//===----------------------------------------------------------------------===//

static ParseResult parseImplicitSSAName(OpAsmParser &parser,
                                        NamedAttrList &resultAttrs) {

  if (parser.parseOptionalAttrDict(resultAttrs))
    return failure();

  // If the attribute dictionary contains no 'name' attribute, infer it from
  // the SSA name (if specified).
  bool hadName = llvm::any_of(
      resultAttrs, [](NamedAttribute attr) { return attr.first == "name"; });

  // If there was no name specified, check to see if there was a useful name
  // specified in the asm file.
  if (hadName)
    return success();

  auto resultName = parser.getResultName(0);
  if (!resultName.first.empty() && !isdigit(resultName.first[0])) {
    StringRef name = resultName.first;
    auto nameAttr = parser.getBuilder().getStringAttr(name);
    auto *context = parser.getBuilder().getContext();
    resultAttrs.push_back({Identifier::get("name", context), nameAttr});
  }

  return success();
}

static void printImplicitSSAName(OpAsmPrinter &p, Operation *op,
                                 DictionaryAttr attr) {
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
}

//===----------------------------------------------------------------------===//
// TblGen Generated Logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/FIRRTL/FIRRTL.cpp.inc"
