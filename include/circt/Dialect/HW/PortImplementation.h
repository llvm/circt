//===- PortImplementation.h - Declare HW op interfaces ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the port structure of HW modules and instances.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_PORTIMPLEMENTATION_H
#define CIRCT_DIALECT_HW_PORTIMPLEMENTATION_H

#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "circt/Support/LLVM.h"

namespace circt {
namespace hw {

/// This holds the name, type, direction of a module's ports
struct PortInfo : public ModulePort {
  /// This is the argument index or the result index depending on the direction.
  /// "0" for an output means the first output, "0" for a in/inout means the
  /// first argument.
  size_t argNum = ~0U;

  /// The optional symbol for this port.
  DictionaryAttr attrs = {};
  LocationAttr loc = {};

  StringRef getName() const { return name.getValue(); }
  bool isInput() const { return dir == ModulePort::Direction::Input; }
  bool isOutput() const { return dir == ModulePort::Direction::Output; }
  bool isInOut() const { return dir == ModulePort::Direction::InOut; }

  /// Return a unique numeric identifier for this port.
  ssize_t getId() const { return isOutput() ? argNum : (-1 - argNum); };

  // Inspect or mutate attributes
  InnerSymAttr getSym() const;
  void setSym(InnerSymAttr sym, MLIRContext *ctx);

  // Return the port's Verilog name. This returns either the port's name, or the
  // `hw.verilogName` attribute if one is present. After `ExportVerilog` has
  // run, this will reflect the exact name of the port as emitted to the Verilog
  // output.
  StringRef getVerilogName() const;
};

raw_ostream &operator<<(raw_ostream &printer, PortInfo port);

/// This holds a decoded list of input/inout and output ports for a module or
/// instance.
struct ModulePortInfo {
  explicit ModulePortInfo(ArrayRef<PortInfo> inputs,
                          ArrayRef<PortInfo> outputs) {
    ports.insert(ports.end(), inputs.begin(), inputs.end());
    ports.insert(ports.end(), outputs.begin(), outputs.end());
    sanitizeInOut();
  }

  explicit ModulePortInfo(ArrayRef<PortInfo> mergedPorts)
      : ports(mergedPorts.begin(), mergedPorts.end()) {
    sanitizeInOut();
  }

  using iterator = SmallVector<PortInfo>::iterator;
  using const_iterator = SmallVector<PortInfo>::const_iterator;

  iterator begin() { return ports.begin(); }
  iterator end() { return ports.end(); }
  const_iterator begin() const { return ports.begin(); }
  const_iterator end() const { return ports.end(); }

  using PortDirectionRange = llvm::iterator_range<
      llvm::filter_iterator<iterator, std::function<bool(const PortInfo &)>>>;

  using ConstPortDirectionRange = llvm::iterator_range<llvm::filter_iterator<
      const_iterator, std::function<bool(const PortInfo &)>>>;

  PortDirectionRange getPortsOfDirection(bool input) {
    std::function<bool(const PortInfo &)> predicateFn;
    if (input) {
      predicateFn = [](const PortInfo &port) -> bool {
        return port.dir == ModulePort::Direction::Input ||
               port.dir == ModulePort::Direction::InOut;
      };
    } else {
      predicateFn = [](const PortInfo &port) -> bool {
        return port.dir == ModulePort::Direction::Output;
      };
    }
    return llvm::make_filter_range(ports, predicateFn);
  }

  ConstPortDirectionRange getPortsOfDirection(bool input) const {
    std::function<bool(const PortInfo &)> predicateFn;
    if (input) {
      predicateFn = [](const PortInfo &port) -> bool {
        return port.dir == ModulePort::Direction::Input ||
               port.dir == ModulePort::Direction::InOut;
      };
    } else {
      predicateFn = [](const PortInfo &port) -> bool {
        return port.dir == ModulePort::Direction::Output;
      };
    }
    return llvm::make_filter_range(ports, predicateFn);
  }

  PortDirectionRange getInputs() { return getPortsOfDirection(true); }

  PortDirectionRange getOutputs() { return getPortsOfDirection(false); }

  ConstPortDirectionRange getInputs() const {
    return getPortsOfDirection(true);
  }

  ConstPortDirectionRange getOutputs() const {
    return getPortsOfDirection(false);
  }

  size_t size() const { return ports.size(); }
  size_t sizeInputs() const {
    auto r = getInputs();
    return std::distance(r.begin(), r.end());
  }
  size_t sizeOutputs() const {
    auto r = getOutputs();
    return std::distance(r.begin(), r.end());
  }

  size_t portNumForInput(size_t idx) const {
    size_t port = 0;
    while (idx || ports[port].isOutput()) {
      if (!ports[port].isOutput())
        --idx;
      ++port;
    }
    return port;
  }

  size_t portNumForOutput(size_t idx) const {
    size_t port = 0;
    while (idx || !ports[port].isOutput()) {
      if (ports[port].isOutput())
        --idx;
      ++port;
    }
    return port;
  }

  PortInfo &at(size_t idx) { return ports[idx]; }
  PortInfo &atInput(size_t idx) { return ports[portNumForInput(idx)]; }
  PortInfo &atOutput(size_t idx) { return ports[portNumForOutput(idx)]; }

  const PortInfo &at(size_t idx) const { return ports[idx]; }
  const PortInfo &atInput(size_t idx) const {
    return ports[portNumForInput(idx)];
  }
  const PortInfo &atOutput(size_t idx) const {
    return ports[portNumForOutput(idx)];
  }

  void eraseInput(size_t idx) {
    assert(idx < sizeInputs());
    ports.erase(ports.begin() + portNumForInput(idx));
  }

private:
  // convert input inout<type> -> inout type
  void sanitizeInOut() {
    for (auto &p : ports)
      if (auto inout = dyn_cast<hw::InOutType>(p.type)) {
        p.type = inout.getElementType();
        p.dir = ModulePort::Direction::InOut;
      }
  }

  /// This contains a list of all ports.  Input first.
  SmallVector<PortInfo> ports;
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
    for (auto &in : portInfo.getInputs())
      inputPortMap[in.name] = in.argNum;

    for (auto &out : portInfo.getOutputs())
      outputPortMap[out.name] = out.argNum;
  }

  explicit ModulePortLookupInfo(MLIRContext *ctx,
                                const SmallVector<PortInfo> &portInfo)
      : ModulePortLookupInfo(ctx, ModulePortInfo(portInfo)) {}

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

} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_HW_PORTIMPLEMENTATION_H
