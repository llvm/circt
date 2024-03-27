//===- HWOpInterfaces.cpp - Implement the HW op interfaces ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the HW operation interfaces.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypeInterfaces.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"

using namespace circt;

hw::InnerSymAttr hw::PortInfo::getSym() const {
  if (attrs)
    return attrs.getAs<::circt::hw::InnerSymAttr>(
        hw::HWModuleLike::getPortSymbolAttrName());
  return {};
}

void hw::PortInfo::setSym(InnerSymAttr sym, MLIRContext *ctx) {
  auto portSymAttr =
      StringAttr::get(ctx, hw::HWModuleLike::getPortSymbolAttrName());
  NamedAttrList pattr(attrs);
  Attribute oldValue;
  if (!sym)
    oldValue = pattr.erase(portSymAttr);
  else
    oldValue = pattr.set(portSymAttr, sym);
  if (oldValue != sym) {
    attrs = pattr.getDictionary(ctx);
  }
}

StringRef hw::PortInfo::getVerilogName() const {
  if (attrs)
    if (auto updatedName = attrs.get("hw.verilogName"))
      return updatedName.cast<StringAttr>().getValue();
  return name.getValue();
}

LogicalResult hw::verifyInnerSymAttr(InnerSymbolOpInterface op) {
  auto innerSym = op.getInnerSymAttr();
  // If does not have any inner sym then ignore.
  if (!innerSym)
    return success();

  if (innerSym.empty())
    return op->emitOpError("has empty list of inner symbols");

  if (!op.supportsPerFieldSymbols()) {
    // The inner sym can only be specified on fieldID=0.
    if (innerSym.size() > 1 || !innerSym.getSymName()) {
      op->emitOpError("does not support per-field inner symbols");
      return failure();
    }
    return success();
  }

  auto result = op.getTargetResult();
  // If op supports per-field symbols, but does not have a target result,
  // its up to the operation to verify itself.
  // (there are no uses for this presently, but be open to this anyway.)
  if (!result)
    return success();
  auto resultType = result.getType();
  auto maxFields = FieldIdImpl::getMaxFieldID(resultType);
  llvm::SmallBitVector indices(maxFields + 1);
  llvm::SmallPtrSet<Attribute, 8> symNames;
  // Ensure fieldID and symbol names are unique.
  auto uniqSyms = [&](InnerSymPropertiesAttr p) {
    if (maxFields < p.getFieldID()) {
      op->emitOpError("field id:'" + Twine(p.getFieldID()) +
                      "' is greater than the maximum field id:'" +
                      Twine(maxFields) + "'");
      return false;
    }
    if (indices.test(p.getFieldID())) {
      op->emitOpError("cannot assign multiple symbol names to the field id:'" +
                      Twine(p.getFieldID()) + "'");
      return false;
    }
    indices.set(p.getFieldID());
    auto it = symNames.insert(p.getName());
    if (!it.second) {
      op->emitOpError("cannot reuse symbol name:'" + p.getName().getValue() +
                      "'");
      return false;
    }
    return true;
  };

  if (!llvm::all_of(innerSym.getProps(), uniqSyms))
    return failure();

  return success();
}

raw_ostream &circt::hw::operator<<(raw_ostream &printer, PortInfo port) {
  StringRef dirstr;
  switch (port.dir) {
  case ModulePort::Direction::Input:
    dirstr = "input";
    break;
  case ModulePort::Direction::Output:
    dirstr = "output";
    break;
  case ModulePort::Direction::InOut:
    dirstr = "inout";
    break;
  }
  printer << dirstr << " " << port.name << " : " << port.type << " (argnum "
          << port.argNum << ", sym " << port.getSym() << ", loc " << port.loc
          << ", args " << port.attrs << ")";
  return printer;
}

#include "circt/Dialect/HW/HWOpInterfaces.cpp.inc"
