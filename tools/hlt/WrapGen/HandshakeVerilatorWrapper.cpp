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

LogicalResult
HandshakeVerilatorWrapper::emitPreamble(Operation * /*kernelOp*/) {
  if (emitIOTypes(emitVerilatorType).failed())
    return failure();

  // Emit model type.
  osi() << "using TModel = V" << funcName() << ";\n";
  osi() << "using " << funcName()
        << "SimInterface = HandshakeSimInterface<TInput, TOutput, "
           "TModel>;\n\n";

  // Emit simulator.
  emitSimulator();

  // Emit simulator driver type.
  osi() << "using TSim = " << funcName() << "Sim;\n";
  return success();
}
void HandshakeVerilatorWrapper::emitSimulator() {
  osi() << "class " << funcName() << "Sim : public " << funcName()
        << "SimInterface {\n";
  osi() << "public:\n";
  osi().indent();

  osi() << funcName() << "Sim() : " << funcName() << "SimInterface() {\n";
  osi().indent();

  osi() << "// --- Generic Verilator interface\n";
  osi() << "interface.clock = &dut->clock;\n";
  osi() << "interface.reset = &dut->reset;\n\n";

  unsigned inCtrlIndex = funcOp.getNumArguments();
  unsigned outCtrlIndex = inCtrlIndex + funcOp.getNumResults() + 1;
  osi() << "// --- Handshake interface\n";
  osi() << "inCtrl->readySig = &dut->arg" << inCtrlIndex << "_ready;\n";
  osi() << "inCtrl->validSig = &dut->arg" << inCtrlIndex << "_valid;\n";
  osi() << "outCtrl->readySig = &dut->arg" << outCtrlIndex << "_ready;\n";
  osi() << "outCtrl->validSig = &dut->arg" << outCtrlIndex << "_valid;\n\n";

  osi() << "// --- Model interface\n";
  auto funcType = funcOp.getType();
  osi() << "// - Input ports\n";
  for (auto &input : enumerate(funcType.getInputs())) {
    auto arg = "arg" + std::to_string(input.index());
    osi() << "addInputPort<HandshakeDataInPort<TArg" << input.index() << ">>(\""
          << arg << "\", ";
    osi() << "&dut->" << arg << "_ready, ";
    osi() << "&dut->" << arg << "_valid, ";
    osi() << "&dut->" << arg << "_data);\n";
  }

  osi() << "\n// - Output ports\n";
  for (auto &res : enumerate(funcType.getResults())) {
    unsigned resIdx = res.index() + inCtrlIndex + 1;
    auto arg = "arg" + std::to_string(resIdx);
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
