//===- SVOps.cpp - Implement the SV operations ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the SV ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/RTL/RTLTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallString.h"

using namespace circt;
using namespace sv;

/// Return true if the specified operation is an expression.
bool sv::isExpression(Operation *op) {
  return isa<sv::TextualValueOp>(op) || isa<sv::GetModportOp>(op) ||
         isa<sv::ReadInterfaceSignalOp>(op);
}

//===----------------------------------------------------------------------===//
// RegOp
//===----------------------------------------------------------------------===//

void RegOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                  Type elementType, StringAttr name) {
  if (name)
    odsState.addAttribute("name", name);

  odsState.addTypes(rtl::InOutType::get(elementType));
}

/// Suggest a name for each result value based on the saved result names
/// attribute.
void RegOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // If the wire has an optional 'name' attribute, use it.
  if (auto nameAttr = (*this)->getAttrOfType<StringAttr>("name"))
    setNameFn(getResult(), nameAttr.getValue());
}

static void printRegOp(OpAsmPrinter &p, Operation *op) {
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

static ParseResult parseRegOp(OpAsmParser &parser, OperationState &result) {
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

//===----------------------------------------------------------------------===//
// Control flow like-operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// IfDefOp

void IfDefOp::build(OpBuilder &odsBuilder, OperationState &result,
                    StringRef cond, std::function<void()> thenCtor,
                    std::function<void()> elseCtor) {
  result.addAttribute("cond", odsBuilder.getStringAttr(cond));
  Region *thenRegion = result.addRegion();
  IfDefOp::ensureTerminator(*thenRegion, odsBuilder, result.location);

  // Fill in the body of the #ifdef.
  if (thenCtor) {
    auto oldIP = &*odsBuilder.getInsertionPoint();
    odsBuilder.setInsertionPointToStart(&*thenRegion->begin());
    thenCtor();
    odsBuilder.setInsertionPoint(oldIP);
  }

  Region *elseRegion = result.addRegion();
  if (elseCtor) {
    IfDefOp::ensureTerminator(*elseRegion, odsBuilder, result.location);
    auto oldIP = &*odsBuilder.getInsertionPoint();
    odsBuilder.setInsertionPointToStart(&*elseRegion->begin());
    elseCtor();
    odsBuilder.setInsertionPoint(oldIP);
  }
}

//===----------------------------------------------------------------------===//
// IfOp

void IfOp::build(OpBuilder &odsBuilder, OperationState &result, Value cond,
                 std::function<void()> thenCtor,
                 std::function<void()> elseCtor) {
  result.addOperands(cond);
  Region *body = result.addRegion();
  IfOp::ensureTerminator(*body, odsBuilder, result.location);

  // Fill in the body of the #ifdef.
  if (thenCtor) {
    auto oldIP = &*odsBuilder.getInsertionPoint();
    odsBuilder.setInsertionPointToStart(&*body->begin());
    thenCtor();
    odsBuilder.setInsertionPoint(oldIP);
  }

  Region *elseRegion = result.addRegion();
  if (elseCtor) {
    IfDefOp::ensureTerminator(*elseRegion, odsBuilder, result.location);
    auto oldIP = &*odsBuilder.getInsertionPoint();
    odsBuilder.setInsertionPointToStart(&*elseRegion->begin());
    elseCtor();
    odsBuilder.setInsertionPoint(oldIP);
  }
}

//===----------------------------------------------------------------------===//
// AlwaysOp

AlwaysOp::Condition AlwaysOp::getCondition(size_t idx) {
  return Condition{EventControl(events()[idx].cast<IntegerAttr>().getInt()),
                   getOperand(idx)};
}

void AlwaysOp::build(OpBuilder &odsBuilder, OperationState &result,
                     ArrayRef<EventControl> events, ArrayRef<Value> clocks,
                     std::function<void()> bodyCtor) {
  assert(events.size() == clocks.size() &&
         "mismatch between event and clock list");

  SmallVector<Attribute> eventAttrs;
  for (auto event : events)
    eventAttrs.push_back(
        odsBuilder.getI32IntegerAttr(static_cast<int32_t>(event)));
  result.addAttribute("events", odsBuilder.getArrayAttr(eventAttrs));
  result.addOperands(clocks);

  // Set up the body.
  Region *body = result.addRegion();
  AlwaysOp::ensureTerminator(*body, odsBuilder, result.location);

  // Fill in the body of the #ifdef.
  if (bodyCtor) {
    auto oldIP = &*odsBuilder.getInsertionPoint();
    odsBuilder.setInsertionPointToStart(&*body->begin());
    bodyCtor();
    odsBuilder.setInsertionPoint(oldIP);
  }
}

/// Ensure that the symbol being instantiated exists and is an InterfaceOp.
static LogicalResult verifyAlwaysOp(AlwaysOp op) {
  if (op.events().size() != op.getNumOperands())
    return op.emitError("different number of operands and events");
  return success();
}

static ParseResult
parseEventList(OpAsmParser &p, Attribute &eventsAttr,
               SmallVectorImpl<OpAsmParser::OperandType> &clocksOperands) {

  // Parse zero or more conditions intoevents and clocksOperands.
  SmallVector<Attribute> events;

  auto loc = p.getCurrentLocation();
  StringRef keyword;
  if (!p.parseOptionalKeyword(&keyword)) {
    while (1) {
      auto kind = symbolizeEventControl(keyword);
      if (!kind.hasValue())
        return p.emitError(loc, "expected 'posedge', 'negedge', or 'edge'");
      auto eventEnum = static_cast<int32_t>(kind.getValue());
      events.push_back(p.getBuilder().getI32IntegerAttr(eventEnum));

      clocksOperands.push_back({});
      if (p.parseOperand(clocksOperands.back()))
        return failure();

      if (failed(p.parseOptionalComma()))
        break;
      if (p.parseKeyword(&keyword))
        return failure();
    }
  }
  eventsAttr = p.getBuilder().getArrayAttr(events);
  return success();
}

static void printEventList(OpAsmPrinter &p, AlwaysOp op, ArrayAttr portsAttr,
                           OperandRange operands) {
  for (size_t i = 0, e = op.getNumConditions(); i != e; ++i) {
    if (i != 0)
      p << ", ";
    auto cond = op.getCondition(i);
    p << stringifyEventControl(cond.event);
    p << ' ';
    p.printOperand(cond.value);
  }
}

//===----------------------------------------------------------------------===//
// AlwaysFFOp

void AlwaysFFOp::build(OpBuilder &odsBuilder, OperationState &result,
                       EventControl clockEdge, Value clock,
                       std::function<void()> bodyCtor) {
  result.addAttribute("clockEdge", odsBuilder.getI32IntegerAttr(
                                       static_cast<int32_t>(clockEdge)));
  result.addOperands(clock);
  result.addAttribute(
      "resetStyle",
      odsBuilder.getI32IntegerAttr(static_cast<int32_t>(ResetType::NoReset)));

  // Set up the body.
  Region *bodyBlk = result.addRegion();
  AlwaysFFOp::ensureTerminator(*bodyBlk, odsBuilder, result.location);

  if (bodyCtor) {
    auto oldIP = &*odsBuilder.getInsertionPoint();
    odsBuilder.setInsertionPointToStart(&*bodyBlk->begin());
    bodyCtor();
    odsBuilder.setInsertionPoint(oldIP);
  }

  // Set up the reset region.
  result.addRegion();
}

void AlwaysFFOp::build(OpBuilder &odsBuilder, OperationState &result,
                       EventControl clockEdge, Value clock,
                       ResetType resetStyle, EventControl resetEdge,
                       Value reset, std::function<void()> resetCtor,
                       std::function<void()> bodyCtor) {
  result.addAttribute("clockEdge", odsBuilder.getI32IntegerAttr(
                                       static_cast<int32_t>(clockEdge)));
  result.addOperands(clock);
  result.addAttribute("resetStyle", odsBuilder.getI32IntegerAttr(
                                        static_cast<int32_t>(resetStyle)));
  result.addAttribute("resetEdge", odsBuilder.getI32IntegerAttr(
                                       static_cast<int32_t>(resetEdge)));
  result.addOperands(reset);

  // Set up the body.
  Region *bodyRegion = result.addRegion();

  if (bodyCtor) {
    AlwaysFFOp::ensureTerminator(*bodyRegion, odsBuilder, result.location);
    auto oldIP = &*odsBuilder.getInsertionPoint();
    odsBuilder.setInsertionPointToStart(&*bodyRegion->begin());
    bodyCtor();
    odsBuilder.setInsertionPoint(oldIP);
  }

  // Set up the reset.
  Region *resetRegion = result.addRegion();

  if (resetCtor) {
    AlwaysFFOp::ensureTerminator(*resetRegion, odsBuilder, result.location);
    auto oldIP = &*odsBuilder.getInsertionPoint();
    odsBuilder.setInsertionPointToStart(&*resetRegion->begin());
    resetCtor();
    odsBuilder.setInsertionPoint(oldIP);
  }
}

//===----------------------------------------------------------------------===//
// InitialOp
//===----------------------------------------------------------------------===//

void InitialOp::build(OpBuilder &odsBuilder, OperationState &result,
                      std::function<void()> bodyCtor) {
  Region *body = result.addRegion();
  InitialOp::ensureTerminator(*body, odsBuilder, result.location);

  // Fill in the body of the #ifdef.
  if (bodyCtor) {
    auto oldIP = &*odsBuilder.getInsertionPoint();
    odsBuilder.setInsertionPointToStart(&*body->begin());
    bodyCtor();
    odsBuilder.setInsertionPoint(oldIP);
  }
}
//===----------------------------------------------------------------------===//
// TypeDecl operations
//===----------------------------------------------------------------------===//

ModportType InterfaceOp::getModportType(StringRef modportName) {
  InterfaceModportOp modportOp = lookupSymbol<InterfaceModportOp>(modportName);
  assert(modportOp && "Modport symbol not found.");
  return ModportType::get(getContext(),
                          SymbolRefAttr::get(modportName, getContext()));
}

Type InterfaceOp::getSignalType(StringRef signalName) {
  InterfaceSignalOp signal = lookupSymbol<InterfaceSignalOp>(signalName);
  assert(signal && "Interface signal symbol not found.");
  return signal.type();
}

static ParseResult parseModportStructs(OpAsmParser &parser,
                                       ArrayAttr &portsAttr) {
  if (parser.parseLParen())
    return failure();

  auto context = parser.getBuilder().getContext();

  SmallVector<Attribute, 8> ports;
  do {
    StringAttr direction;
    FlatSymbolRefAttr signal;
    if (parser.parseAttribute(direction) || parser.parseAttribute(signal))
      return failure();

    ports.push_back(ModportStructAttr::get(direction, signal, context));
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRParen())
    return failure();

  portsAttr = ArrayAttr::get(ports, context);
  return success();
}

static void printModportStructs(OpAsmPrinter &p, Operation *,
                                ArrayAttr portsAttr) {
  p << " (";
  llvm::interleaveComma(portsAttr, p, [&](Attribute attr) {
    auto port = attr.cast<ModportStructAttr>();
    p << port.direction();
    p << ' ';
    p.printSymbolName(port.signal().getRootReference());
  });
  p << ')';
}

void InterfaceSignalOp::build(mlir::OpBuilder &builder,
                              ::mlir::OperationState &state, StringRef name,
                              mlir::Type type) {
  build(builder, state, name, mlir::TypeAttr::get(type));
}

void InterfaceModportOp::build(OpBuilder &builder, OperationState &state,
                               StringRef name, ArrayRef<StringRef> inputs,
                               ArrayRef<StringRef> outputs) {
  auto *ctxt = builder.getContext();
  SmallVector<Attribute, 8> directions;
  StringAttr inputDir = StringAttr::get("input", ctxt);
  StringAttr outputDir = StringAttr::get("output", ctxt);
  for (auto input : inputs)
    directions.push_back(ModportStructAttr::get(
        inputDir, SymbolRefAttr::get(input, ctxt), ctxt));
  for (auto output : outputs)
    directions.push_back(ModportStructAttr::get(
        outputDir, SymbolRefAttr::get(output, ctxt), ctxt));
  build(builder, state, name, ArrayAttr::get(directions, ctxt));
}

/// Ensure that the symbol being instantiated exists and is an InterfaceOp.
static LogicalResult verifyInterfaceInstanceOp(InterfaceInstanceOp op) {
  auto symtable = SymbolTable::getNearestSymbolTable(op);
  if (!symtable)
    return op.emitError("sv.interface.instance must exist within a region "
                        "which has a symbol table.");
  auto ifaceTy = op.getType();
  auto referencedOp =
      SymbolTable::lookupSymbolIn(symtable, ifaceTy.getInterface());
  if (!referencedOp)
    return op.emitError("Symbol not found: ") << ifaceTy.getInterface() << ".";
  if (!isa<InterfaceOp>(referencedOp))
    return op.emitError("Symbol ")
           << ifaceTy.getInterface() << " is not an InterfaceOp.";
  return success();
}

/// Ensure that the symbol being instantiated exists and is an
/// InterfaceModportOp.
static LogicalResult verifyGetModportOp(GetModportOp op) {
  auto symtable = SymbolTable::getNearestSymbolTable(op);
  if (!symtable)
    return op.emitError("sv.interface.instance must exist within a region "
                        "which has a symbol table.");
  auto ifaceTy = op.getType();
  auto referencedOp =
      SymbolTable::lookupSymbolIn(symtable, ifaceTy.getModport());
  if (!referencedOp)
    return op.emitError("Symbol not found: ") << ifaceTy.getModport() << ".";
  if (!isa<InterfaceModportOp>(referencedOp))
    return op.emitError("Symbol ")
           << ifaceTy.getModport() << " is not an InterfaceModportOp.";
  return success();
}

void GetModportOp::build(OpBuilder &builder, OperationState &state, Value value,
                         StringRef field) {
  auto ifaceTy = value.getType().dyn_cast<InterfaceType>();
  assert(ifaceTy && "GetModportOp expects an InterfaceType.");
  auto fieldAttr = SymbolRefAttr::get(field, builder.getContext());
  auto modportSym =
      SymbolRefAttr::get(ifaceTy.getInterface().getRootReference(), {fieldAttr},
                         builder.getContext());
  build(builder, state, {ModportType::get(builder.getContext(), modportSym)},
        {value}, fieldAttr);
}

void ReadInterfaceSignalOp::build(OpBuilder &builder, OperationState &state,
                                  Value iface, StringRef signalName) {
  auto ifaceTy = iface.getType().dyn_cast<InterfaceType>();
  assert(ifaceTy && "ReadInterfaceSignalOp expects an InterfaceType.");
  auto fieldAttr = SymbolRefAttr::get(signalName, builder.getContext());
  InterfaceOp ifaceDefOp = SymbolTable::lookupNearestSymbolFrom<InterfaceOp>(
      iface.getDefiningOp(), ifaceTy.getInterface());
  assert(ifaceDefOp &&
         "ReadInterfaceSignalOp could not resolve an InterfaceOp.");
  build(builder, state, {ifaceDefOp.getSignalType(signalName)}, {iface},
        fieldAttr);
}

ParseResult parseIfaceTypeAndSignal(OpAsmParser &p, Type &ifaceTy,
                                    FlatSymbolRefAttr &signalName) {
  SymbolRefAttr fullSym;
  if (p.parseAttribute(fullSym) || fullSym.getNestedReferences().size() != 1)
    return failure();

  auto *ctxt = p.getBuilder().getContext();
  ifaceTy = InterfaceType::get(
      ctxt, FlatSymbolRefAttr::get(fullSym.getRootReference(), ctxt));
  signalName = FlatSymbolRefAttr::get(fullSym.getLeafReference(), ctxt);
  return success();
}

void printIfaceTypeAndSignal(OpAsmPrinter &p, Operation *op, Type type,
                             FlatSymbolRefAttr signalName) {
  InterfaceType ifaceTy = type.dyn_cast<InterfaceType>();
  assert(ifaceTy && "Expected an InterfaceType");
  auto sym = SymbolRefAttr::get(ifaceTy.getInterface().getRootReference(),
                                {signalName}, op->getContext());
  p << sym;
}

LogicalResult verifySignalExists(Value ifaceVal, FlatSymbolRefAttr signalName) {
  auto ifaceTy = ifaceVal.getType().dyn_cast<InterfaceType>();
  if (!ifaceTy)
    return failure();
  InterfaceOp iface = SymbolTable::lookupNearestSymbolFrom<InterfaceOp>(
      ifaceVal.getDefiningOp(), ifaceTy.getInterface());
  if (!iface)
    return failure();
  InterfaceSignalOp signal = iface.lookupSymbol<InterfaceSignalOp>(signalName);
  if (!signal)
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// Other ops.
//===----------------------------------------------------------------------===//

static LogicalResult verifyAliasOp(AliasOp op) {
  // Must have at least two operands.
  if (op.operands().size() < 2)
    return op.emitOpError("alias must have at least two operands");

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/SV/SV.cpp.inc"
#include "circt/Dialect/SV/SVEnums.cpp.inc"
#include "circt/Dialect/SV/SVStructs.cpp.inc"
