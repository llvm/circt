//===- StdWrapper.cpp - Standard wrapper ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the StdWrapper class, an HLT wrapper
// for wrapping builtin.func based kernels.
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/hlt/WrapGen/std/StdWrapper.h"
#include "circt/Tools/hlt/WrapGen/CEmitterUtils.h"

using namespace llvm;
using namespace mlir;
using namespace circt;

namespace circt {
namespace hlt {

SmallVector<std::string> StdWrapper::getIncludes() {
  SmallVector<std::string> includes;
  includes.push_back("circt/Tools/hlt/Simulator/StdSimInterface.h");
  includes.push_back("circt/Tools/hlt/Simulator/SimDriver.h");
  includes.push_back("cstdint");
  return includes;
}

LogicalResult StdWrapper::emitPreamble(Operation * /*kernelOp*/) {
  if (emitIOTypes(emitType).failed())
    return failure();

  // Emit simulator interface type.
  osi() << "using " << funcName()
        << "SimInterface = StdSimInterface<TInput, TOutput>;\n\n";

  // Forward declare kernel function; this is an external symbol that is defined
  // in the lowered LLVMIR version of the kernel.
  std::string retType;
  auto funcType = funcOp.getType();
  if (funcType.getNumResults() > 1)
    retType = "TOutput";
  else if (funcType.getNumResults() == 1) {
    llvm::raw_string_ostream ss(retType);
    if (emitType(ss, funcOp.getLoc(), funcType.getResult(0)).failed())
      return failure();
  } else
    retType = "void";

  osi() << "extern \"C\" " << retType << " " << funcName() << "(";
  bool failed = false;
  interleaveComma(funcOp.getArgumentTypes(), osi(), [&](Type type) {
    failed |= emitType(osi(), funcOp.getLoc(), type).failed();
  });
  if (failed)
    return failure();
  osi() << ");\n\n";

  // Emit simulator.
  osi() << "class " << funcName() << "Sim : public " << funcName()
        << "SimInterface {\n";
  osi() << "protected:\n";
  osi().indent();

  // Emit 'call' function.
  osi() << "TOutput call(const TInput& input) override {\n";
  osi().indent();

  // Unpack arguments
  for (auto type : enumerate(funcOp.getArgumentTypes())) {
    osi() << "auto a" << type.index() << " = std::get<" << type.index()
          << ">(input);\n";
  }

  // Call
  osi() << "return " << funcName() << "(";
  interleaveComma(llvm::iota_range(0U, funcOp.getNumArguments(), false), osi(),
                  [&](unsigned idx) { osi() << "a" << idx; });
  osi() << ");\n";
  osi().unindent();
  osi() << "};\n";
  osi().unindent();
  osi() << "};\n\n";

  // Emit simulator driver type.
  osi() << "using TSim = " << funcName() << "Sim;\n";

  return success();
}

} // namespace hlt
} // namespace circt
