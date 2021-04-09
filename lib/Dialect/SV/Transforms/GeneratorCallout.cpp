//===- RTLLegalizeNames.cpp - RTL Name Legalization Pass ------------------===//
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
#include "circt/Dialect/SV/SVAnalyses.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Support/ImplicitLocOpBuilder.h"
#include "circt/Translation/ExportVerilog.h"
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

private:
  DenseMap<StringRef, RTLGeneratorSchemaOp> generatorSchema;
};
} // end anonymous namespace

void RTLGeneratorCalloutPass::runOnOperation() {
  ModuleOp root = getOperation();

  OpBuilder builder(root->getContext());
  auto pathLoc = genExecutable.find_last_of("/\\");
  auto genExecutablePath = genExecutable.substr(0, pathLoc);
  auto genExecutableName = genExecutable.substr(pathLoc + 1);
  auto genMemExe =
      llvm::sys::findProgramByName(genExecutableName, {genExecutablePath});
  SmallVector<Operation *> opsToRemove;
  for (auto &op : *root.getBody()) {
    if (auto modGenOp = dyn_cast<RTLModuleGeneratedOp>(op)) {
      std::vector<std::string> generatorArgs;
      // First argument should be the executable name.
      generatorArgs.push_back(genExecutableName);
      // This is the option required by duh-mem, TODO: make it an option.
      generatorArgs.push_back("fir");
      // Get the corresponding schema associated with this generated op.
      if (auto genSchema =
              dyn_cast<RTLGeneratorSchemaOp>(modGenOp.getGeneratorKindOp())) {
        // The moduleName option is not present in the schema, so add it
        // explicitly.
        generatorArgs.push_back("--moduleName");
        generatorArgs.push_back(
            modGenOp.getVerilogModuleNameAttr().getValue().str());
        generatorArgs.push_back("--o");
        // The file name for output is ignored and the module name is used
        // instead.
        generatorArgs.push_back("out");
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
        std::vector<StringRef> generatorArgStrRef;
        for (const std::string &a : generatorArgs) {
          generatorArgStrRef.push_back(a);
        }

        std::string errMsg;
        // TODO: Redirects only for debugging and testing purpose, we will
        // remove this.
        Optional<StringRef> redirects[] = {None, StringRef("genOutRedirect2"),
                                           StringRef("genOutRedirect3")};
        int result = llvm::sys::ExecuteAndWait(
            *genMemExe, generatorArgStrRef, /*Env=*/None,
            /*Redirects=*/redirects,
            /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg);

        // TODO: Silently ignores if any error occurs.
        if (result >= 0) {
          builder.setInsertionPointAfter(&op);
          builder.create<rtl::RTLModuleExternOp>(
              modGenOp.getLoc(), modGenOp.getVerilogModuleNameAttr(),
              modGenOp.getPorts());
          opsToRemove.push_back(modGenOp);
          // builder.create<sv::VerbatimOp>(op.getLoc(), "`include " +
          // moduleName + ".v");
        }
      }
    }
  }
  for (auto remOp : opsToRemove)
    remOp->erase();
}

std::unique_ptr<Pass> circt::sv::createRTLGeneratorCalloutPass() {
  return std::make_unique<RTLGeneratorCalloutPass>();
}
