//===- DebugOps.cpp - Debug dialect operations ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Debug/DebugOps.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/DenseSet.h"

using namespace circt;
using namespace debug;
using namespace mlir;

/// Check that `ext` maps language names to strings.
static LogicalResult verifyExt(Operation *op, DictionaryAttr ext) {
  if (!ext)
    return success();

  for (auto entry : ext) {
    auto lang = entry.getName().getValue();
    if (!isExtLanguage(lang))
      return op->emitOpError("ext key '")
             << lang << "' is not a lowercase language name";

    auto json = dyn_cast<StringAttr>(entry.getValue());
    if (!json)
      return op->emitOpError("ext value of '") << lang << "' must be a string";
  }

  return success();
}

/// Check the type of each discardable attribute in `DebugDialect.td`.
LogicalResult DebugDialect::verifyOperationAttribute(Operation *op,
                                                     NamedAttribute attr) {
  auto name = attr.getName().getValue();
  auto value = attr.getValue();

  if (name == kDbgMergedFromAttr) {
    if (!isa<StringAttr>(value))
      return op->emitOpError("'") << name << "' must be a string";
    return success();
  }

  if (name == kDbgLoweredAttr || name == kDbgArtificialAttr ||
      name == kDbgMemoryAttr) {
    if (!isa<UnitAttr>(value))
      return op->emitOpError("'") << name << "' must be a unit attribute";
    return success();
  }

  if (name == kDbgModuleLocsAttr) {
    auto locs = dyn_cast<DictionaryAttr>(value);
    if (!locs || !llvm::all_of(locs, [](NamedAttribute entry) {
          return isa<LocationAttr>(entry.getValue());
        }))
      return op->emitOpError("'")
             << name << "' must map module names to locations";
    return success();
  }

  if (name == kDbgInstanceAttr) {
    auto entries = dyn_cast<ArrayAttr>(value);
    if (!entries || !llvm::all_of(entries, llvm::IsaPred<InstanceAttr>))
      return op->emitOpError("'")
             << name << "' must be a list of #dbg.instance";
    for (auto inst : entries.getAsRange<InstanceAttr>()) {
      if (auto path = inst.getPath();
          path && !llvm::all_of(path, llvm::IsaPred<StringAttr>))
        return op->emitOpError("'") << name << "' path must hold strings";
      if (failed(verifyExt(op, inst.getExt())))
        return failure();
    }
    return success();
  }

  if (name != kDbgModuleInfoAttr)
    return op->emitOpError("has unknown attribute '") << name << "'";

  auto info = dyn_cast<DictionaryAttr>(value);
  if (!info)
    return op->emitOpError("'")
           << kDbgModuleInfoAttr << "' must be a dictionary";

  auto infoName = info.get("name");
  if (infoName && !isa<StringAttr>(infoName))
    return op->emitOpError("'")
           << kDbgModuleInfoAttr << "' name must be a string";

  auto ext = info.get("ext");
  if (ext && !isa<DictionaryAttr>(ext))
    return op->emitOpError("'")
           << kDbgModuleInfoAttr << "' ext must be a dictionary";

  return verifyExt(op, cast_if_present<DictionaryAttr>(ext));
}

//===----------------------------------------------------------------------===//
// StructOp
//===----------------------------------------------------------------------===//

ParseResult StructOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the struct fields.
  SmallVector<Attribute> names;
  SmallVector<OpAsmParser::UnresolvedOperand, 16> operands;
  std::string nameBuffer;
  auto parseField = [&]() {
    nameBuffer.clear();
    if (parser.parseString(&nameBuffer) || parser.parseColon() ||
        parser.parseOperand(operands.emplace_back()))
      return failure();
    names.push_back(StringAttr::get(parser.getContext(), nameBuffer));
    return success();
  };
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Braces, parseField))
    return failure();

  // Parse the attribute dictionary.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse the field types, if there are any fields.
  SmallVector<Type> types;
  if (!operands.empty()) {
    if (parser.parseColon())
      return failure();
    auto typesLoc = parser.getCurrentLocation();
    if (parser.parseTypeList(types))
      return failure();
    if (types.size() != operands.size())
      return parser.emitError(typesLoc,
                              "number of fields and types must match");
  }

  // Resolve the operands.
  for (auto [operand, type] : llvm::zip(operands, types))
    if (parser.resolveOperand(operand, type, result.operands))
      return failure();

  // Finalize the op.
  result.addAttribute("names", ArrayAttr::get(parser.getContext(), names));
  result.addTypes(StructType::get(parser.getContext()));
  return success();
}

void StructOp::print(OpAsmPrinter &printer) {
  printer << " {";
  llvm::interleaveComma(llvm::zip(getFields(), getNames()), printer.getStream(),
                        [&](auto pair) {
                          auto [field, name] = pair;
                          printer.printAttribute(name);
                          printer << ": ";
                          printer.printOperand(field);
                        });
  printer << '}';
  printer.printOptionalAttrDict(getOperation()->getAttrs(), {"names"});
  if (!getFields().empty()) {
    printer << " : ";
    printer << getFields().getTypes();
  }
}

LogicalResult StructOp::verify() {
  auto flips = getFlips();
  if (!flips)
    return success();

  if (flips->size() != getFields().size())
    return emitOpError("has ")
           << flips->size() << " flips for " << getFields().size() << " fields";

  if (!llvm::is_contained(*flips, true))
    return emitOpError("has no flipped field; leave out 'flips'");

  return success();
}

//===----------------------------------------------------------------------===//
// ArrayOp
//===----------------------------------------------------------------------===//

ParseResult ArrayOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the elements, attributes and types.
  SmallVector<OpAsmParser::UnresolvedOperand, 16> operands;
  if (parser.parseOperandList(operands, AsmParser::Delimiter::Square) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Resolve the operands.
  if (!operands.empty()) {
    Type type;
    if (parser.parseColon() || parser.parseType(type))
      return failure();
    for (auto operand : operands)
      if (parser.resolveOperand(operand, type, result.operands))
        return failure();
  }

  // Finalize the op.
  result.addTypes(ArrayType::get(parser.getContext()));
  return success();
}

void ArrayOp::print(OpAsmPrinter &printer) {
  printer << " [";
  printer << getElements();
  printer << ']';
  printer.printOptionalAttrDict(getOperation()->getAttrs());
  if (!getElements().empty()) {
    printer << " : ";
    printer << getElements()[0].getType();
  }
}

//===----------------------------------------------------------------------===//
// Generated operation code
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "circt/Dialect/Debug/Debug.cpp.inc"

void DebugDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Debug/Debug.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ValueOp
//===----------------------------------------------------------------------===//

LogicalResult ValueOp::verify() {
  // Final IR is expected to have >=1 user (dbg.variable/dbg.struct/dbg.array).
  if (!llvm::all_of(getResult().getUsers(), [](Operation *user) {
        return isa<VariableOp, StructOp, ArrayOp>(user);
      }))
    return emitOpError(
        "must only be used as an operand of dbg.variable, dbg.struct, or "
        "dbg.array");

  auto kind = getKind();
  bool isClockOrReset = kind == ValueKind::Clock || kind == ValueKind::Reset ||
                        kind == ValueKind::AsyncReset;
  auto intType = dyn_cast<IntegerType>(getValue().getType());
  if (isClockOrReset && intType && intType.getWidth() != 1)
    return emitOpError("of kind ")
           << stringifyValueKind(*kind) << " must be i1, got " << intType;

  return verifyExt(*this, getExtAttr());
}

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

LogicalResult VariableOp::verify() {
  auto kind = getKind();
  if (getDir() && kind && *kind != VariableKind::Reg)
    return emitOpError("of kind ")
           << stringifyVariableKind(*kind) << " cannot have a direction";

  return success();
}

//===----------------------------------------------------------------------===//
// ScopeOp
//===----------------------------------------------------------------------===//

LogicalResult ScopeOp::verify() { return verifyExt(*this, getExtAttr()); }

//===----------------------------------------------------------------------===//
// EnumOp
//===----------------------------------------------------------------------===//

LogicalResult EnumOp::verify() {
  if (getVariantsMap().empty())
    return emitOpError("variantsMap must not be empty");

  llvm::SmallDenseSet<int64_t> seenValues{};
  for (auto namedAttr : getVariantsMap()) {
    auto intAttr = dyn_cast<IntegerAttr>(namedAttr.getValue());
    if (!intAttr)
      return emitOpError("variantsMap entry '")
             << namedAttr.getName().getValue()
             << "' must be an IntegerAttr, got " << namedAttr.getValue();
    if (!intAttr.getType().isSignlessInteger())
      return emitOpError() << "variant '" << namedAttr.getName().getValue()
                           << "' must have a signless integer value";
    auto value = intAttr.getInt();
    if (!seenValues.insert(value).second)
      return emitOpError("duplicate enum value ") << value;
  }

  return verifyExt(*this, getExtAttr());
}
