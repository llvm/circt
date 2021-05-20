//===- FieldRef.cpp - FIRRTL Field Refs  ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines FieldRef and helpers for them.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FieldRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace firrtl;

Operation *FieldRef::getDefiningOp() const {
  if (auto *op = value.getDefiningOp())
    return op;
  return value.cast<BlockArgument>().getOwner()->getParentOp();
}

FieldRef::FieldRef(Value val) : value(val) {
  // This code walks upwards from the subfield and calculates the field ID at
  // each level. At each stage, it must take the current id, and re-index it as
  // a nested bundle under the parent field.. This is accomplished by using the
  // parent field's ID as a base, and adding the field ID of the child.
  while (val) {
    Operation *op = val.getDefiningOp();

    // If this is a block argument, we are done.
    if (!op)
      return;

    // TODO: implement subindex op.
    TypeSwitch<Operation *>(op)
        .Case<SubfieldOp>([&](auto op) {
          auto bundleType = op.input().getType().template cast<BundleType>();
          auto index = bundleType.getElementIndex(op.fieldname()).getValue();
          // Rebase the current index on the parent field's index.
          id = bundleType.getFieldID(index) + id;
          val = op.input();
        })
        .Default([&](auto op) {
          // We are done walking the chain.
          value = val;
          val = nullptr;
        });
  }
}

/// Get the string name of a value which is a direct child of a declaration op.
static void getDeclName(Value value, SmallString<64> &string) {
  if (auto arg = value.dyn_cast<BlockArgument>()) {
    // Get the module ports and get the name.
    auto module = cast<FModuleOp>(arg.getOwner()->getParentOp());
    SmallVector<ModulePortInfo> ports = module.getPorts();
    string += ports[arg.getArgNumber()].name.getValue();
    return;
  }

  auto *op = value.getDefiningOp();
  TypeSwitch<Operation *>(op)
      .Case<InstanceOp, MemOp>([&](auto op) {
        string += op.name();
        string += ".";
        string +=
            op.getPortName(value.cast<OpResult>().getResultNumber()).getValue();
      })
      .Case<WireOp, RegOp, RegResetOp>([&](auto op) { string += op.name(); });
}

std::string FieldRef::getFieldName() const {

  SmallString<64> name;
  getDeclName(value, name);

  auto type = value.getType();
  auto localID = id;
  while (localID) {
    // Strip off the flip type if there is one.
    if (auto flipType = type.dyn_cast<FlipType>())
      type = flipType.getElementType();
    // TODO: support vector types.
    auto bundleType = type.cast<BundleType>();
    auto index = bundleType.getIndexForFieldID(localID);
    // Add the current field string, and recurse into a subfield.
    auto &element = bundleType.getElements()[index];
    name += ".";
    name += element.name.getValue();
    type = element.type;
    // Get a field localID for the nested bundle.
    localID = localID - bundleType.getFieldID(index);
  }

  return name.str().str();
}

void FieldRef::print(raw_ostream &os) const { os << getFieldName(); }

void FieldRef::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}
