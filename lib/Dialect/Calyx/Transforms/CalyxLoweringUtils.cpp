//===- CalyxLoweringUtils.cpp - Calyx lowering utility methods --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Various lowering utility methods converting to and from Calyx programs.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxLoweringUtils.h"
#include "circt/Dialect/Calyx/CalyxHelpers.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"

#include <variant>

namespace circt {
namespace calyx {

MemoryInterface::MemoryInterface() {}
MemoryInterface::MemoryInterface(const MemoryPortsImpl &ports) : impl(ports) {}
MemoryInterface::MemoryInterface(calyx::MemoryOp memOp) : impl(memOp) {}

Value MemoryInterface::readData() {
  if (auto *memOp = std::get_if<calyx::MemoryOp>(&impl); memOp) {
    return memOp->readData();
  }
  return std::get<MemoryPortsImpl>(impl).readData;
}

Value MemoryInterface::done() {
  if (auto *memOp = std::get_if<calyx::MemoryOp>(&impl); memOp) {
    return memOp->done();
  }
  return std::get<MemoryPortsImpl>(impl).done;
}

Value MemoryInterface::writeData() {
  if (auto *memOp = std::get_if<calyx::MemoryOp>(&impl); memOp) {
    return memOp->writeData();
  }
  return std::get<MemoryPortsImpl>(impl).writeData;
}

Value MemoryInterface::writeEn() {
  if (auto *memOp = std::get_if<calyx::MemoryOp>(&impl); memOp) {
    return memOp->writeEn();
  }
  return std::get<MemoryPortsImpl>(impl).writeEn;
}

ValueRange MemoryInterface::addrPorts() {
  if (auto *memOp = std::get_if<calyx::MemoryOp>(&impl); memOp) {
    return memOp->addrPorts();
  }
  return std::get<MemoryPortsImpl>(impl).addrPorts;
}

WalkResult
getCiderSourceLocationMetadata(calyx::ComponentOp component,
                               SmallVectorImpl<Attribute> &sourceLocations) {
  Builder builder(component->getContext());
  return component.getControlOp().walk([&](Operation *op) {
    if (!calyx::isControlLeafNode(op))
      return WalkResult::advance();

    std::string sourceLocation;
    llvm::raw_string_ostream os(sourceLocation);
    op->getLoc()->print(os);
    int64_t position = sourceLocations.size();
    sourceLocations.push_back(
        StringAttr::get(op->getContext(), sourceLocation));

    op->setAttr("pos", builder.getI64IntegerAttr(position));
    return WalkResult::advance();
  });
}

bool matchConstantOp(Operation *op, APInt &value) {
  return mlir::detail::constant_int_op_binder(&value).match(op);
}

bool singleLoadFromMemory(Value memoryReference) {
  return llvm::count_if(memoryReference.getUses(), [](OpOperand &user) {
           return isa<mlir::memref::LoadOp>(user.getOwner());
         }) <= 1;
}

bool noStoresToMemory(Value memoryReference) {
  return llvm::none_of(memoryReference.getUses(), [](OpOperand &user) {
    return isa<mlir::memref::StoreOp>(user.getOwner());
  });
}

Value getComponentOutput(calyx::ComponentOp compOp, unsigned outPortIdx) {
  size_t index = compOp.getInputPortInfo().size() + outPortIdx;
  assert(index < compOp.getNumArguments() &&
         "Exceeded number of arguments in the Component");
  return compOp.getArgument(index);
}

Type convIndexType(PatternRewriter &rewriter, Type type) {
  if (type.isIndex())
    return rewriter.getI32Type();
  return type;
}

} // namespace calyx
} // namespace circt
