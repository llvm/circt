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
      return cast<StringAttr>(updatedName).getValue();
  return name.getValue();
}

LogicalResult
hw::verifyInnerSymAttr(InnerSymAttr innerSym, Type type,
                       llvm::function_ref<InFlightDiagnostic()> emitError) {
  assert(innerSym && "null inner symbol attribute provided");
  // Tentatively accept empty inner symbol attributes.
  // In the future we may prefer these over NULL.
  if (innerSym.empty())
    return success();

  if (!type) {
    // No "type" provided means must have zero or one inner symbols.
    // If symbol is present, must have fieldID == 0.
    // Scan and reject any with fieldID != 0.
    for (auto prop : innerSym) {
      if (prop.getFieldID() != 0)
        return emitError() << "does not support per-field inner symbols, but "
                              "has inner symbol '"
                           << prop.getName().getValue()
                           << "' with non-zero field id " << prop.getFieldID();
    }
    // If scan passed but there are multiple inner symbols for this target.
    if (innerSym.size() > 1) {
      // Multiple with fieldID == 0.
      auto err = emitError() << "has more than one symbol defined: ";
      llvm::interleaveComma(innerSym, err, [&](auto prop) {
        err << "'" << prop.getName().getValue() << "'";
      });
      return err;
    }
    return success();
  }

  auto maxFields = FieldIdImpl::getMaxFieldID(type);
  llvm::SmallBitVector indices(maxFields + 1);
  llvm::SmallPtrSet<Attribute, 8> symNames;
  // Ensure fieldID and symbol names are unique.
  auto uniqSyms = [&](InnerSymPropertiesAttr p) {
    if (maxFields < p.getFieldID()) {
      emitError() << "field id " << p.getFieldID()
                  << " is greater than the maximum field id " << maxFields;
      return false;
    }
    if (indices.test(p.getFieldID())) {
      emitError() << "cannot assign multiple symbol names to the field id "
                  << p.getFieldID();
      return false;
    }
    indices.set(p.getFieldID());
    auto it = symNames.insert(p.getName());
    if (!it.second) {
      emitError() << "cannot reuse symbol name '" << p.getName().getValue()
                  << "'";
      return false;
    }
    return true;
  };

  if (!llvm::all_of(innerSym.getProps(), uniqSyms))
    return failure();

  return success();
}

LogicalResult hw::verifyInnerSymOp(InnerSymbolOpInterface op) {
  auto innerSym = op.getInnerSymAttr();
  // If does not have any inner sym then ignore.
  if (!innerSym)
    return success();

  if (innerSym.empty())
    return op->emitOpError("has empty list of inner symbols");

  if (!op.supportsPerFieldSymbols())
    return verifyInnerSymAttr(innerSym, [op]() { return op->emitOpError(); });

  // If op supports per-field symbols, but does not have a target result,
  // its up to the operation to verify itself.
  // (there are no uses for this presently, but be open to this anyway.)
  auto result = op.getTargetResult();
  if (!result)
    return success();

  return verifyInnerSymAttr(innerSym, result.getType(),
                            [op]() { return op->emitOpError(); });
}

LogicalResult hw::verifyPortInnerSymsIfPortList(Operation *op) {
  assert(op->hasTrait<OpTrait::InnerSymbolTable>());
  PortList opWithPorts = dyn_cast<hw::PortList>(op);
  // Skip if not a PortList.
  if (!opWithPorts)
    return success();

  auto ports = opWithPorts.getPortList();
  for (auto const &indexAndPort : llvm::enumerate(ports)) {
    auto &pi = indexAndPort.value();
    auto idx = indexAndPort.index();
    Location loc = pi.loc ? Location(pi.loc) : opWithPorts.getLoc();
    if (auto sym = pi.getSym())
      if (failed(hw::verifyInnerSymAttr(sym, pi.type, [&]() {
            return mlir::emitError(loc)
                   << "verification of inner symbol"
                   << (sym.size() > 1 ? "s" : "") << " failed on port " << idx
                   << " with name " << pi.name << ": ";
          })))
        return failure();
  }

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
