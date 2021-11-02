//===- BaseWrapper.cpp - Simulation wrapper base class --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements BaseWrapper, the base class for all HLT wrappers.
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/hlt/WrapGen/BaseWrapper.h"
#include "circt/Tools/hlt/WrapGen/CEmitterUtils.h"

using namespace llvm;
using namespace mlir;

namespace circt {
namespace hlt {

LogicalResult BaseWrapper::wrap(Operation *refOp, Operation *kernelOp) {
  funcOp = dyn_cast<FuncOp>(refOp);
  if (!funcOp)
    return refOp->emitOpError()
           << "Expected the reference operation to be a builtin.func operation";

  if (createFile(refOp->getLoc(), funcOp.getName() + ".cpp").failed())
    return failure();
  osi() << "// This file is generated. Do not modify!\n";

  // Emit includes
  for (auto include : getIncludes())
    osi() << "#include \"" << include << "\"\n";
  osi() << "\n";

  // Emit namespaces
  for (auto ns : getNamespaces())
    osi() << "using namespace " << ns << ";\n";
  osi() << "\n";

  // Emit preamble;
  if (emitPreamble(kernelOp).failed())
    return failure();

  // Emit simulator driver and instantiation. This is dependent on types TInput,
  // TOutput, TSim that should have been defined in emitPreamble.
  osi() << "using TSimDriver = SimDriver<TInput, TOutput, TSim>;\n";
  osi() << "static TSimDriver *driver = nullptr;\n\n";
  osi() << "void init_sim() {\n";
  osi() << "  assert(driver == nullptr && \"Simulator already initialized "
           "!\");\n";
  osi() << "  driver = new TSimDriver();\n";
  osi() << "}\n\n";

  // Emit async call
  os() << "extern \"C\" void " << funcOp.getName() + "_call"
       << "(";

  // InterleaveComma doesn't accept enumerate(inTypes)
  int i = 0;
  bool failed = false;
  interleaveComma(funcOp.getType().getInputs(), os(), [&](auto inType) {
    failed |= emitType(os(), kernelOp->getLoc(), inType).failed();
    os() << " in" << i++;
  });
  if (failed)
    return failure();
  os() << ") {\n";
  osi().indent();
  emitAsyncCall(kernelOp);
  osi().unindent();
  osi() << "}\n\n";

  // Emit async await
  os() << "extern \"C\" ";
  if (emitTypes(os(), funcOp.getLoc(), funcOp.getType().getResults()).failed())
    return failure();
  os() << " " << funcOp.getName() + "_await"
       << "() {\n";
  osi().indent();
  emitAsyncAwait(kernelOp);

  // End
  osi().unindent();
  osi() << "}\n";

  return success();
}

LogicalResult BaseWrapper::emitIOTypes(const TypeEmitter &emitter) {
  auto funcType = funcOp.getType();

  // Emit in types.
  for (auto &inType : enumerate(funcType.getInputs())) {
    osi() << "using TArg" << inType.index() << " = ";
    if (emitter(osi(), funcOp.getLoc(), inType.value()).failed())
      return failure();
    osi() << ";\n";
  }
  osi() << "using TInput = std::tuple<";
  interleaveComma(llvm::iota_range(0U, funcOp.getNumArguments(), false), osi(),
                  [&](unsigned i) { osi() << "TArg" << i; });
  osi() << ">;\n\n";

  // Emit out types.
  for (auto &outType : enumerate(funcType.getResults())) {
    osi() << "using TRes" << outType.index() << " = ";
    if (emitter(osi(), funcOp.getLoc(), outType.value()).failed())
      return failure();
    osi() << ";\n";
  }
  osi() << "using TOutput = std::tuple<";
  interleaveComma(llvm::iota_range(0U, funcType.getNumResults(), false), osi(),
                  [&](unsigned i) { osi() << "TRes" << i; });
  osi() << ">;\n\n";
  return success();
}

LogicalResult BaseWrapper::createFile(Location loc, Twine fn) {
  assert(!outputFile && "Output file already set");
  std::error_code EC;
  SmallString<128> absFn = outDir;
  sys::path::append(absFn, fn);
  outputFile = std::make_unique<raw_fd_ostream>(absFn, EC, sys::fs::OF_None);
  if (EC)
    return emitError(loc) << "Error while opening file";
  outputFilename = absFn.str().str();
  outputFileIndented = std::make_unique<raw_indented_ostream>(*outputFile);
  return success();
}

void BaseWrapper::emitAsyncCall(Operation * /*kernelOp*/) {
  osi() << "if (driver == nullptr)\n";
  osi() << "  init_sim();\n";

  // Pack arguments
  osi() << "TInput input{";
  interleaveComma(llvm::iota_range(0U, funcOp.getNumArguments(), false), osi(),
                  [&](unsigned i) { osi() << "in" << i; });
  osi() << "};\n";

  // Push to driver
  osi() << "driver->push(input); // non-blocking\n";
}

void BaseWrapper::emitAsyncAwait(Operation * /*kernelOp*/) {
  osi() << "TOutput output = driver->pop(); // blocking\n";
  if (funcOp.getNumResults() == 1)
    osi() << "return std::get<0>(output);\n";
  else
    osi() << "return output";
}

} // namespace hlt
} // namespace circt
