//===- ESIDialect.cpp - ESI dialect code defs -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dialect definitions. Should be relatively standard boilerplate.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/RTL/RTLOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/ImplicitLocOpBuilder.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"

using namespace circt;
using namespace circt::esi;

ESIDialect::ESIDialect(MLIRContext *context)
    : Dialect("esi", context, TypeID::get<ESIDialect>()) {

  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/ESI/ESI.cpp.inc"
      >();
}

/// Find all the port triples on a module which fit the
/// <name>/<name>_valid/<name>_ready pattern. Ready must be the opposite
/// direction of the other two.
void circt::esi::findValidReadySignals(Operation *modOp,
                                       SmallVectorImpl<ESIPortMapping> &names) {
  SmallVector<rtl::ModulePortInfo, 64> ports;
  rtl::getModulePortInfo(modOp, ports);

  llvm::StringMap<rtl::ModulePortInfo> nameMap(ports.size());
  for (auto port : ports)
    nameMap[port.getName()] = port;

  for (auto port : ports) {
    if (port.direction == rtl::PortDirection::INOUT)
      continue;

    // Try to find a corresponding 'valid' port.
    SmallString<64> portName = port.getName();
    portName.append("_valid");
    auto valid = nameMap.find(portName);
    if (valid == nameMap.end() || valid->second.direction != port.direction ||
        !valid->second.type.isSignlessInteger(1))
      continue;

    // Try to find a corresponding 'ready' port.
    portName = port.getName();
    portName.append("_ready");
    rtl::PortDirection readyDir = port.direction == rtl::PortDirection::INPUT
                                      ? rtl::PortDirection::OUTPUT
                                      : rtl::PortDirection::INPUT;
    auto ready = nameMap.find(portName);
    if (ready == nameMap.end() || ready->second.direction != readyDir ||
        !valid->second.type.isSignlessInteger(1))
      continue;

    // Found one.
    names.push_back(ESIPortMapping{
        .data = port, .valid = valid->second, .ready = ready->second});
  }
}

/// Build an ESI module wrapper, converting the wires with latency-insensitive
/// semantics with ESI channels and passing through the rest.
Operation *circt::esi::buildESIWrapper(OpBuilder &b, Operation *mod,
                                       ArrayRef<ESIPortMapping> esiPortNames) {
  auto *ctxt = b.getContext();
  Location loc = mod->getLoc();
  FunctionType modType = rtl::getModuleType(mod);

  SmallVector<rtl::ModulePortInfo, 64> ports;
  rtl::getModulePortInfo(mod, ports);

  // Memoize the list of ready/valid ports to ignore.
  StringSet<> controlPorts;
  // Store a lookup table indexed on the data port name.
  llvm::StringMap<ESIPortMapping> dataPortMap;
  for (const auto &esiPort : esiPortNames) {
    assert(esiPort.data.direction != rtl::PortDirection::INOUT);
    dataPortMap[esiPort.data.name.getValue()] = esiPort;

    assert(esiPort.valid.direction == esiPort.data.direction &&
           "Valid port direction must match data port.");
    controlPorts.insert(esiPort.valid.name.getValue());

    assert(esiPort.ready.direction == (esiPort.data.isOutput()
                                           ? rtl::PortDirection::INPUT
                                           : rtl::PortDirection::OUTPUT) &&
           "Ready must be opposite direction to data.");
    controlPorts.insert(esiPort.ready.name.getValue());
  }

  // Build the list of ports, skipping the valid/ready, and converting the ESI
  // data ports to the ESI channel port type.
  SmallVector<rtl::ModulePortInfo, 64> wrapperPorts;
  // Map the new operand to the old port.
  SmallVector<rtl::ModulePortInfo, 64> inputPortMap;
  // Map the new result to the old port.
  SmallVector<rtl::ModulePortInfo, 64> outputPortMap;

  for (const auto port : ports) {
    if (controlPorts.contains(port.name.getValue()))
      continue;

    rtl::ModulePortInfo newPort = port;
    if (dataPortMap.find(port.name.getValue()) != dataPortMap.end())
      newPort.type = esi::ChannelPort::get(ctxt, port.type);

    if (port.isOutput()) {
      newPort.argNum = outputPortMap.size();
      outputPortMap.push_back(port);
    } else {
      newPort.argNum = inputPortMap.size();
      inputPortMap.push_back(port);
    }
    wrapperPorts.push_back(newPort);
  }

  // Create the wrapper module.
  SmallString<64> wrapperNameBuf;
  StringAttr wrapperName = b.getStringAttr(
      (SymbolTable::getSymbolName(mod) + "_esi").toStringRef(wrapperNameBuf));
  auto wrapper = b.create<rtl::RTLModuleOp>(loc, wrapperName, wrapperPorts);
  wrapper.getBodyBlock()->clear(); // Erase the terminator.
  auto modBuilder =
      ImplicitLocOpBuilder::atBlockBegin(loc, wrapper.getBodyBlock());
  BackedgeBuilder bb(modBuilder, modBuilder.getLoc());
  SmallVector<Value, 64> outputs(
      wrapper.getNumResults()); // rtl.output operands.

  // Assemble the inputs for the wrapped module.
  SmallVector<Value, 64> wrappedOperands(modType.getNumInputs());
  // Index the backedges by the wrapped modules result number.
  llvm::DenseMap<size_t, Backedge> backedges;

  // Go through the input ports, either tunneling them through or unwrapping
  // them.
  for (const auto port : wrapperPorts) {
    if (port.isOutput())
      continue;

    Value arg = wrapper.getArgument(port.argNum);
    auto esiPort = dataPortMap.find(port.name.getValue());
    if (esiPort == dataPortMap.end()) {
      // If it's just a regular port, it just gets passed through.
      size_t wrappedOpNum = inputPortMap[port.argNum].argNum;
      wrappedOperands[wrappedOpNum] = arg;
      continue;
    }

    Backedge ready = bb.get(modBuilder.getI1Type());
    backedges.insert(std::make_pair(esiPort->second.ready.argNum, ready));
    auto unwrap = modBuilder.create<UnwrapValidReady>(arg, ready);
    wrappedOperands[esiPort->second.data.argNum] = unwrap.rawOutput();
    wrappedOperands[esiPort->second.valid.argNum] = unwrap.valid();
  }

  // Iterate through the output ports, identify the ESI channels, and build
  // wrappers.
  for (const auto port : wrapperPorts) {
    if (!port.isOutput())
      continue;
    auto esiPort = dataPortMap.find(port.name.getValue());
    if (esiPort == dataPortMap.end())
      continue;

    Backedge data = bb.get(esiPort->second.data.type);
    Backedge valid = bb.get(modBuilder.getI1Type());
    auto wrap = modBuilder.create<WrapValidReady>(data, valid);
    backedges.insert(std::make_pair(esiPort->second.data.argNum, data));
    backedges.insert(std::make_pair(esiPort->second.valid.argNum, valid));
    outputs[port.argNum] = wrap.chanOutput();
    wrappedOperands[esiPort->second.ready.argNum] = wrap.ready();
  }

  // Instantiate the wrapped module.
  auto wrappedInst = modBuilder.create<rtl::InstanceOp>(
      modType.getResults(), "wrapped", SymbolTable::getSymbolName(mod),
      wrappedOperands, DictionaryAttr());

  // Find all the regular outputs and either tunnel them through.
  for (const auto port : wrapperPorts) {
    if (!port.isOutput())
      continue;
    auto esiPort = dataPortMap.find(port.name.getValue());
    if (esiPort != dataPortMap.end())
      continue;
    size_t wrappedResNum = outputPortMap[port.argNum].argNum;
    outputs[port.argNum] = wrappedInst.getResult(wrappedResNum);
  }

  // Hookup all the backedges.
  for (size_t i = 0, e = wrappedInst.getNumResults(); i < e; ++i) {
    auto backedge = backedges.find(i);
    if (backedge != backedges.end())
      backedge->second.setValue(wrappedInst.getResult(i));
  }

  modBuilder.create<rtl::OutputOp>(outputs);
  return wrapper;
}

#include "circt/Dialect/ESI/ESIAttrs.cpp.inc"
