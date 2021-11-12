//===- HandshakeVerilatorWrapper.cpp - Handshake Verilator wrapper --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the HandshakeVerilatorWrapper class,
// an HLT wrapper for wrapping handshake.funcOp based kernels, simulated by
// Verilator.
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/hlt/WrapGen/handshake/HandshakeVerilatorWrapper.h"
#include "circt/Tools/hlt/WrapGen/VerilatorEmitterUtils.h"

using namespace llvm;
using namespace mlir;
using namespace circt;

namespace circt {
namespace hlt {

SmallVector<std::string> HandshakeVerilatorWrapper::getIncludes() {
  SmallVector<std::string> includes;
  includes.push_back(("V" + funcName() + ".h").str());
  includes.push_back("circt/Tools/hlt/Simulator/HandshakeSimInterface.h");
  includes.push_back("circt/Tools/hlt/Simulator/SimDriver.h");
  includes.push_back("cstdint");
  return includes;
}

LogicalResult HandshakeVerilatorWrapper::emitPreamble(Operation *kernelOp) {
  auto handshakeFirMod = dyn_cast<firrtl::FModuleLike>(kernelOp);
  if (!handshakeFirMod)
    return kernelOp->emitOpError() << "Expected a FIRRTL module of the "
                                      "handshake kernel that is to be wrapped";

  if (emitIOTypes(emitVerilatorType).failed())
    return failure();

  // Emit model type.
  osi() << "using TModel = V" << funcName() << ";\n";
  osi() << "using " << funcName()
        << "SimInterface = HandshakeSimInterface<TInput, TOutput, "
           "TModel>;\n\n";

  // Emit simulator.
  emitSimulator(handshakeFirMod);

  // Emit simulator driver type.
  osi() << "using TSim = " << funcName() << "Sim;\n";
  return success();
}

void HandshakeVerilatorWrapper::emitSimulator(
    firrtl::FModuleLike handshakeFirMod) {
  osi() << "class " << funcName() << "Sim : public " << funcName()
        << "SimInterface {\n";
  osi() << "public:\n";
  osi().indent();

  osi() << funcName() << "Sim() : " << funcName() << "SimInterface() {\n";
  osi().indent();

  osi() << "// --- Generic Verilator interface\n";
  osi() << "interface.clock = &dut->clock;\n";
  osi() << "interface.reset = &dut->reset;\n\n";

  auto getInputName = [&](unsigned idx) -> StringRef {
    return handshakeFirMod.getPortName(idx);
  };
  unsigned inCtrlIndex = funcOp.getNumArguments();
  auto getResName = [&](unsigned idx) -> StringRef {
    idx += inCtrlIndex + 1;
    return handshakeFirMod.getPortName(idx);
  };

  auto inCtrlName = getInputName(inCtrlIndex);
  auto outCtrlName = getResName(funcOp.getNumResults());
  osi() << "// --- Handshake interface\n";
  osi() << "inCtrl->readySig = &dut->" << inCtrlName << "_ready;\n";
  osi() << "inCtrl->validSig = &dut->" << inCtrlName << "_valid;\n";
  osi() << "outCtrl->readySig = &dut->" << outCtrlName << "_ready;\n";
  osi() << "outCtrl->validSig = &dut->" << outCtrlName << "_valid;\n\n";

  // We expect equivalence between the order of function arguments and the ports
  // of the FIRRTL module. Additionally, the handshake layer has then added
  // one more argument and return port for the control signals.

  osi() << "// --- Model interface\n";
  auto funcType = funcOp.getType();
  osi() << "// - Input ports\n";
  for (auto &input : enumerate(funcType.getInputs())) {
    auto arg = getInputName(input.index());
    osi() << "addInputPort<HandshakeDataInPort<TArg" << input.index() << ">>(\""
          << arg << "\", ";
    osi() << "&dut->" << arg << "_ready, ";
    osi() << "&dut->" << arg << "_valid, ";
    osi() << "&dut->" << arg << "_data);\n";
  }
  osi() << "\n// - Output ports\n";
  for (auto &res : enumerate(funcType.getResults())) {
    auto arg = getResName(res.index());
    osi() << "addOutputPort<HandshakeDataOutPort<TRes" << res.index() << ">>(\""
          << arg << "\", ";
    osi() << "&dut->" << arg << "_ready, ";
    osi() << "&dut->" << arg << "_valid, ";
    osi() << "&dut->" << arg << "_data);\n";
  }
  osi().unindent();
  osi() << "};\n";

  osi().unindent();
  osi() << "};\n\n";
}

} // namespace hlt
} // namespace circt
