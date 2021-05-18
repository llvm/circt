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
  //
  // a (0) { b (1) c (2) }
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
  // This will find the index of the bundle element with a field ID which
  // contains the searched for ID.  This means that it returns the index of the
  // greatest less-than field ID.
  auto findIndexForFieldID = [](BundleType bundleType, unsigned id) {
    assert(bundleType.getElements().size() && "Bundle must have >0 fields");
    // Find the field corresponding to this element using a binary search.
    unsigned l = 0;
    unsigned r = bundleType.getElements().size() - 1;
    while (l < r) {
      auto m = l + (r - l + 1) / 2;
      if (id < bundleType.getFieldID(m)) {
        r = m - 1;
      } else {
        l = m;
      }
    }
    return l;
  };

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
    auto index = findIndexForFieldID(bundleType, localID);
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
