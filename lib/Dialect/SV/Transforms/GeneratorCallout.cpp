//===- GeneratorCallout.cpp - Generator Callout Pass ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Call arbitrary programs and pass them the attributes attached to external
// modules.
//
//===----------------------------------------------------------------------===//

#include "SVPassDetail.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Support/ImplicitLocOpBuilder.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"

using namespace circt;
using namespace sv;
using namespace rtl;

//===----------------------------------------------------------------------===//
// GeneratorCalloutPass
//===----------------------------------------------------------------------===//

namespace {

struct RTLGeneratorCalloutPass
    : public sv::RTLGeneratorCalloutPassBase<RTLGeneratorCalloutPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

// Parse the \p genExec string, to extract the program executable, the path to
// the executable and all the required arguments from \p genExecArgs.
llvm::ErrorOr<std::string> static parseProgramArgs(const std::string genExec) {
  std::string execName, execPath;

  // TODO: Find a platform independent way to parse the path from the executable
  // string. Does LLVM have such a method already ? The executable name can
  // contain one of the following characters.
  auto pathLoc = genExec.find_last_of("/\\");
  if (pathLoc != std::string::npos)
    execPath = genExec.substr(0, pathLoc + 1);
  else
    execPath = "";
  execName = genExec.substr(pathLoc + 1);

  auto genMemExe = llvm::sys::findProgramByName(execName, {execPath});
  // If the executable was not found in the provided path (when the path is
  // empty), then search it in the $PATH.
  if (!genMemExe)
    genMemExe = llvm::sys::findProgramByName(execName);
  return genMemExe;
}

void RTLGeneratorCalloutPass::runOnOperation() {
  ModuleOp root = getOperation();
  std::string genExecutableName, genExecutablePath;
  SmallVector<StringRef> genOptions;
  StringRef genArgs(genExecArgs);
  genArgs.split(genOptions, ';');
  OpBuilder builder(root->getContext());
  auto genMemExe = parseProgramArgs(genExecutable);
  // If cannot find the executable, then nothing to do, return.
  if (!genMemExe) {
    root.emitError("\n Error: Cannot find executable " + genExecutable +
                   " in program path\n");
    return;
  }
  for (auto &op : llvm::make_early_inc_range(root.getBody()->getOperations())) {
    if (auto modGenOp = dyn_cast<RTLModuleGeneratedOp>(op)) {
      // Get the corresponding schema associated with this generated op.
      if (auto genSchema =
              dyn_cast<RTLGeneratorSchemaOp>(modGenOp.getGeneratorKindOp())) {
        if (genSchema.descriptor().str() != schemaName)
          continue;
        SmallVector<std::string> generatorArgs;
        // First argument should be the executable name.
        generatorArgs.push_back(genExecutableName);

        for (auto o : genOptions) {
          generatorArgs.push_back(o.str());
        }
        auto moduleName = modGenOp.getVerilogModuleNameAttr().getValue().str();
        // The moduleName option is not present in the schema, so add it
        // explicitly.
        generatorArgs.push_back("--moduleName");
        generatorArgs.push_back(moduleName);
        // Iterate over all the attributes in the schema.
        // Assumption: All the options required by the generator program must be
        // present in the schema.
        for (auto attr : genSchema.requiredAttrs()) {
          // Get the port name from schema.
          StringRef portName = attr.dyn_cast<StringAttr>().getValue();
          generatorArgs.push_back("--" + portName.str());
          // Get the value for the corresponding port name.
          // Assumption: the value always exists.
          auto v = modGenOp->getAttr(portName);
          if (auto intV = v.dyn_cast<IntegerAttr>())
            generatorArgs.push_back(intV.getValue().toString(10, false));
          else if (auto strV = v.dyn_cast<StringAttr>())
            generatorArgs.push_back(strV.getValue().str());
        }
        SmallVector<StringRef> generatorArgStrRef;
        for (const std::string &a : generatorArgs) {
          generatorArgStrRef.push_back(a);
        }

        std::string errMsg;
        Optional<StringRef> redirects[] = {None, StringRef(genExecOutFileName),
                                           None};
        int result = llvm::sys::ExecuteAndWait(
            *genMemExe, generatorArgStrRef, /*Env=*/None,
            /*Redirects=*/redirects,
            /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg);

        // TODO: Silently ignore if any error occurs?
        if (result == 0) {
          auto bufferRead = llvm::MemoryBuffer::getFile(genExecOutFileName);
          if (!bufferRead || !*bufferRead)
            return;
          // Only extract the first line from the output.
          auto fileContent = (*bufferRead)->getBuffer().split('\n').first.str();
          builder.setInsertionPointAfter(&op);
          RTLModuleExternOp extMod = builder.create<rtl::RTLModuleExternOp>(
              modGenOp.getLoc(), modGenOp.getVerilogModuleNameAttr(),
              modGenOp.getPorts());
          // Attach an attribute to which file the definition of the external
          // module exists in.
          extMod->setAttr("filenames", builder.getStringAttr(fileContent));
          modGenOp.erase();
        }
      }
    }
  }
}

std::unique_ptr<Pass> circt::sv::createRTLGeneratorCalloutPass() {
  return std::make_unique<RTLGeneratorCalloutPass>();
}
