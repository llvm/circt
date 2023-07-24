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

#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

namespace circt {
namespace hw {

/// A module port direction.
enum class PortDirection {
  INPUT = 1,
  OUTPUT = 2,
  INOUT = 3,
};

/// This holds the name, type, direction of a module's ports
struct PortInfo {
  StringAttr name;
  PortDirection direction;
  Type type;

  /// This is the argument index or the result index depending on the direction.
  /// "0" for an output means the first output, "0" for a in/inout means the
  /// first argument.
  size_t argNum = ~0U;

  /// The optional symbol for this port.
  InnerSymAttr sym = {};
  LocationAttr loc = {};

  StringRef getName() const { return name.getValue(); }
  bool isInput() const { return direction == PortDirection::INPUT; }
  bool isOutput() const { return direction == PortDirection::OUTPUT; }
  bool isInOut() const { return direction == PortDirection::INOUT; }

  /// Return a unique numeric identifier for this port.
  ssize_t getId() const { return isOutput() ? argNum : (-1 - argNum); };
};

/// This holds a decoded list of input/inout and output ports for a module or
/// instance.
struct ModulePortInfo {
  explicit ModulePortInfo(ArrayRef<PortInfo> inputs, ArrayRef<PortInfo> outputs)
      : inputs(inputs.begin(), inputs.end()),
        outputs(outputs.begin(), outputs.end()) {}

  explicit ModulePortInfo(ArrayRef<PortInfo> mergedPorts) {
    inputs.reserve(mergedPorts.size());
    outputs.reserve(mergedPorts.size());
    for (auto port : mergedPorts) {
      if (port.isOutput())
        outputs.push_back(port);
      else
        inputs.push_back(port);
    }
  }

  /// This contains a list of the input and inout ports.
  SmallVector<PortInfo> inputs;
  /// This is a list of the output ports.
  SmallVector<PortInfo> outputs;
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
    for (auto &in : portInfo.inputs)
      inputPortMap[in.name] = in.argNum;

    for (auto &out : portInfo.outputs)
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

class HWModuleLike;
namespace hwmodulelike_impl {

void setModuleType(HWModuleLike op, ModuleType type);

/// Return all of the attributes for the argument at 'index'.
ArrayRef<NamedAttribute> getArgAttrs(HWModuleLike op, unsigned index);

/// Return all of the attributes for the result at 'index'.
ArrayRef<NamedAttribute> getResultAttrs(HWModuleLike op, unsigned index);


/// Returns the dictionary attribute corresponding to the argument at 'index'.
/// If there are no argument attributes at 'index', a null attribute is
/// returned.
DictionaryAttr getArgAttrDict(HWModuleLike op, unsigned index);

/// Returns the dictionary attribute corresponding to the result at 'index'.
/// If there are no result attributes at 'index', a null attribute is
/// returned.
DictionaryAttr getResultAttrDict(HWModuleLike op, unsigned index);

/// Set the attributes held by the argument at 'index'.
void setArgAttrs(HWModuleLike op, unsigned index,
                 ArrayRef<NamedAttribute> attributes);

/// Set the attributes held by the result at 'index'.
void setResultAttrs(HWModuleLike op, unsigned index,
                    ArrayRef<NamedAttribute> attributes);

/// If the an attribute exists with the specified name, change it to the new
/// value. Otherwise, add a new attribute with the specified name/value.
//template <typename ConcreteType>
void setArgAttr(HWModuleLike op, unsigned index, StringAttr name,
                Attribute value);

/// If the an attribute exists with the specified name, change it to the new
/// value. Otherwise, add a new attribute with the specified name/value.
//template <typename ConcreteType>
void setResultAttr(HWModuleLike op, unsigned index, StringAttr name,
                   Attribute value);

}


} // namespace hw
} // namespace circt

namespace mlir {
namespace OpTrait {

/// This trait is for operations that define a scope for resolving InnerRef's,
/// and provides verification for InnerRef users (via InnerRefUserOpInterface).
template <typename ConcreteType>
class InnerRefNamespace : public TraitBase<ConcreteType, InnerRefNamespace> {
public:
  static LogicalResult verifyRegionTrait(Operation *op) {
    static_assert(
        ConcreteType::template hasTrait<::mlir::OpTrait::SymbolTable>(),
        "expected operation to be a SymbolTable");

    if (op->getNumRegions() != 1)
      return op->emitError("expected operation to have a single region");
    if (!op->getRegion(0).hasOneBlock())
      return op->emitError("expected operation to have a single block");

    // Verify all InnerSymbolTable's and InnerRef users.
    return ::circt::hw::detail::verifyInnerRefNamespace(op);
  }
};

/// A trait for inner symbol table functionality on an operation.
template <typename ConcreteType>
class InnerSymbolTable : public TraitBase<ConcreteType, InnerSymbolTable> {
public:
  static LogicalResult verifyRegionTrait(Operation *op) {
    // Insist that ops with InnerSymbolTable's provide a Symbol, this is
    // essential to how InnerRef's work.
    static_assert(
        ConcreteType::template hasTrait<::mlir::SymbolOpInterface::Trait>(),
        "expected operation to define a Symbol");

    // InnerSymbolTable's must be directly nested within an InnerRefNamespace.
    auto *parent = op->getParentOp();
    if (!parent || !parent->hasTrait<InnerRefNamespace>())
      return op->emitError(
          "InnerSymbolTable must have InnerRefNamespace parent");

    return success();
  }
};
} // namespace OpTrait
} // namespace mlir

#include "circt/Dialect/HW/HWOpInterfaces.h.inc"

#endif // CIRCT_DIALECT_HW_HWOPINTERFACES_H
