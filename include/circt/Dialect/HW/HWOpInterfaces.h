//===- HWOpInterfaces.h - Declare HW op interfaces --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation interfaces for the HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWOPINTERFACES_H
#define CIRCT_DIALECT_HW_HWOPINTERFACES_H

#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

namespace circt {
namespace hw {

/// This holds the name, type, direction of a module's ports
struct PortInfo : public ModulePort {
  /// This is the argument index or the result index depending on the direction.
  /// "0" for an output means the first output, "0" for a in/inout means the
  /// first argument.
  size_t argNum = ~0U;

  /// The optional symbol for this port.
  InnerSymAttr sym = {};
  LocationAttr loc = {};

  StringRef getName() const { return name.getValue(); }
  bool isInput() const { return dir == ModulePort::Direction::Input; }
  bool isOutput() const { return dir == ModulePort::Direction::Output; }
  bool isInOut() const { return dir == ModulePort::Direction::InOut; }

  /// Return a unique numeric identifier for this port.
  ssize_t getId() const { return isOutput() ? argNum : (-1 - argNum); };
};

/// This holds a decoded list of input/inout and output ports for a module or
/// instance.
struct ModulePortInfo {
  explicit ModulePortInfo(ArrayRef<PortInfo> inputs,
                          ArrayRef<PortInfo> outputs) {
    ports.insert(ports.end(), inputs.begin(), inputs.end());
    numInputs = ports.size();
    ports.insert(ports.end(), outputs.begin(), outputs.end());
  }

  explicit ModulePortInfo(ArrayRef<PortInfo> mergedPorts)
      : ports(mergedPorts.begin(), mergedPorts.end()), numInputs(0) {
    std::stable_sort(ports.begin(), ports.end(),
                     [](const PortInfo &lhs, const PortInfo &rhs) {
                       return !lhs.isOutput() && rhs.isOutput();
                     });
    while (numInputs < ports.size() && !ports[numInputs].isOutput())
      ++numInputs;
  }

  using iterator = SmallVector<PortInfo>::iterator;
  using const_iterator = SmallVector<PortInfo>::const_iterator;
  iterator begin_input() { return ports.begin(); }
  iterator end_input() { return ports.begin() + numInputs; }
  const_iterator begin_input() const { return ports.begin(); }
  const_iterator end_input() const { return ports.begin() + numInputs; }
  iterator begin_output() { return end_input(); }
  iterator end_output() { return ports.end(); }
  const_iterator begin_output() const { return end_input(); }
  const_iterator end_output() const { return ports.end(); }
  iterator begin() { return ports.begin(); }
  iterator end() { return ports.end(); }
  const_iterator begin() const { return ports.begin(); }
  const_iterator end() const { return ports.end(); }

  llvm::iterator_range<iterator> all() { return {begin(), end()}; }
  llvm::iterator_range<const_iterator> all() const { return {begin(), end()}; }
  llvm::iterator_range<iterator> inputs() {
    return {begin_input(), end_input()};
  }
  llvm::iterator_range<const_iterator> inputs() const {
    return {begin_input(), end_input()};
  }
  llvm::iterator_range<iterator> outputs() {
    return {begin_output(), end_output()};
  }
  llvm::iterator_range<const_iterator> outputs() const {
    return {begin_output(), end_output()};
  }

  size_t size() const { return ports.size(); }
  size_t sizeInputs() const { return numInputs; }
  size_t sizeOutputs() const { return ports.size() - numInputs; }

  PortInfo &at(size_t idx) { return ports[idx]; }
  PortInfo &atInput(size_t idx) { return ports[idx]; }
  PortInfo &atOutput(size_t idx) { return ports[idx + numInputs]; }

  const PortInfo &at(size_t idx) const { return ports[idx]; }
  const PortInfo &atInput(size_t idx) const { return ports[idx]; }
  const PortInfo &atOutput(size_t idx) const { return ports[idx + numInputs]; }

  void eraseInput(size_t idx) {
    assert(numInputs);
    assert(idx < numInputs);
    ports.erase(ports.begin() + idx);
    --numInputs;
  }

private:
  /// This contains a list of all ports.  Input first.
  SmallVector<PortInfo> ports;

  /// This is the index of the first output port
  size_t numInputs;
};

// This provides capability for looking up port indices based on port names.
struct ModulePortLookupInfo {
  FailureOr<unsigned>
  lookupPortIndex(const llvm::DenseMap<StringAttr, unsigned> &portMap,
                  StringAttr name) const {
    auto it = portMap.find(name);
    if (it == portMap.end())
      return failure();
    return it->second;
  }

public:
  explicit ModulePortLookupInfo(MLIRContext *ctx,
                                const ModulePortInfo &portInfo)
      : ctx(ctx) {
    for (auto &in : portInfo.inputs())
      inputPortMap[in.name] = in.argNum;

    for (auto &out : portInfo.outputs())
      outputPortMap[out.name] = out.argNum;
  }

  // Return the index of the input port with the specified name.
  FailureOr<unsigned> getInputPortIndex(StringAttr name) const {
    return lookupPortIndex(inputPortMap, name);
  }

  // Return the index of the output port with the specified name.
  FailureOr<unsigned> getOutputPortIndex(StringAttr name) const {
    return lookupPortIndex(outputPortMap, name);
  }

  FailureOr<unsigned> getInputPortIndex(StringRef name) const {
    return getInputPortIndex(StringAttr::get(ctx, name));
  }

  FailureOr<unsigned> getOutputPortIndex(StringRef name) const {
    return getOutputPortIndex(StringAttr::get(ctx, name));
  }

private:
  llvm::DenseMap<StringAttr, unsigned> inputPortMap;
  llvm::DenseMap<StringAttr, unsigned> outputPortMap;
  MLIRContext *ctx;
};

class InnerSymbolOpInterface;
/// Verification hook for verifying InnerSym Attribute.
LogicalResult verifyInnerSymAttr(InnerSymbolOpInterface op);

namespace detail {
LogicalResult verifyInnerRefNamespace(Operation *op);
} // namespace detail

} // namespace hw
} // namespace circt

namespace mlir {
namespace OpTrait {
/// A trait for inner symbol table functionality on an operation.
template <typename ConcreteType>
class InnerSymbolTable : public TraitBase<ConcreteType, InnerSymbolTable> {
public:
  static LogicalResult verifyRegionTrait(Operation *op);
};
} // namespace OpTrait
} // namespace mlir

//===----------------------------------------------------------------------===//
// ODS-generated code.
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOpInterfaces.h.inc"

//===----------------------------------------------------------------------===//
// Code which needs the generated code declarations.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace OpTrait {

template <typename ConcreteType>
LogicalResult InnerSymbolTable<ConcreteType>::verifyRegionTrait(Operation *op) {
  // Insist that ops with InnerSymbolTable's provide a Symbol, this is
  // essential to how InnerRef's work.
  static_assert(
      ConcreteType::template hasTrait<::mlir::SymbolOpInterface::Trait>(),
      "expected operation to define a Symbol");

  // InnerSymbolTable's must be directly nested within an InnerRefNamespace.
  auto *parent = op->getParentOp();
  if (!isa_and_nonnull<::circt::hw::InnerRefNamespaceOpInterface>(parent))
    return op->emitError("InnerSymbolTable must have InnerRefNamespace parent");

  return success();
}

} // namespace OpTrait
} // namespace mlir

#endif // CIRCT_DIALECT_HW_HWOPINTERFACES_H
