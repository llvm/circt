//===- SinkLocation.cpp - FIRRTL Sink Locations -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines SinkLocation and helpers for them.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/SinkLocation.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace firrtl;

Operation *SinkLocation::getDefiningOp() const {
  if (auto *op = sink.getDefiningOp())
    return op;
  return sink.cast<BlockArgument>().getOwner()->getParentOp();
}

SinkLocation::SinkLocation(Value val) : sink(val) {
  while (val) {
    Operation *op = val.getDefiningOp();

    // If this is a block argument, we are done.
    if (!op)
      return;

    // TODO: resolve SubindexOp.
    TypeSwitch<Operation *>(op)
        .Case<SubfieldOp>([&](auto op) {
          auto bundleType = op.input().getType().template cast<BundleType>();
          auto index = bundleType.getElementIndex(op.fieldname()).getValue();
          path.push_back(index);
          val = op.input();
        })
        .Default([&](auto op) {
          // We are done walking the chain.
          sink = val;
          val = nullptr;
        });
  }

  // Path is constructed by pushing to the back as an optimization, and must be
  // reversed before completing.
  std::reverse(path.begin(), path.end());
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

std::string SinkLocation::getFieldName() const {
  SmallString<64> name;
  getDeclName(sink, name);
  auto type = sink.getType();
  // Walk the index path and construct an element name.
  for (auto index : path) {
    // Strip the flip off the type if there is one. TODO: when there are no more
    // "outer" flips for outputs, we will only need to do this when it is a
    // bundle type.
    if (auto flipType = type.dyn_cast<FlipType>())
      type = flipType.getElementType();
    // TODO: resolve SubindexOp.
    TypeSwitch<Type>(type)
        .Case<BundleType>([&](auto bundle) {
          auto element = bundle.getElements()[index];
          name += ".";
          name += element.name.getValue();
          type = element.type;
        })
        .Default(
            [](auto unknown) { llvm_unreachable("unhandled aggregate type"); });
  }
  return name.str().str();
}

void SinkLocation::print(raw_ostream &os) const { os << getFieldName(); }

void SinkLocation::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}
