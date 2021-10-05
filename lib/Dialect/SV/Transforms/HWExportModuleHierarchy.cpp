//===- HWExportModuleHierarchy.cpp - Export Module Hierarchy ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Export the module and instance hierarchy information to JSON. This pass looks
// for modules with the firrtl.moduleHierarchyFile attribute and collects the
// hierarchy starting at those modules. The hierarchy information is then
// encoded as JSON in an sv.verbatim op with the output_file attribute set.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Support/Path.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace circt;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct HWExportModuleHierarchyPass
    : public sv::HWExportModuleHierarchyBase<HWExportModuleHierarchyPass> {
  HWExportModuleHierarchyPass(StringRef directory) {
    directoryName = directory.str();
  }
  void runOnOperation() override;
};

/// Recursively print the module hierarchy as serialized as JSON.
static void printHierarchy(hw::InstanceOp &inst, SymbolTable &symbolTable,
                           llvm::json::OStream &J) {
  J.object([&] {
    J.attribute("instance_name", inst.instanceName());
    J.attribute("module_name", inst.moduleName());
    J.attributeArray("instances", [&] {
      auto moduleOp =
          symbolTable.lookup<hw::HWModuleOp>(inst.moduleNameAttr().getValue());

      // Only recurse on module ops, not extern or generated ops, whose internal
      // are opaque.
      if (moduleOp) {
        for (auto op : moduleOp.getOps<hw::InstanceOp>()) {
          printHierarchy(op, symbolTable, J);
        }
      }
    });
  });
}

/// Return the JSON-serialized module hierarchy for the given module as the top
/// of the hierarchy.
static void extractHierarchyFromTop(hw::HWModuleOp op, SymbolTable &symbolTable,
                                    llvm::raw_ostream &os) {
  llvm::json::OStream J(os);

  // As a special case for top-level module, set instance name to module name,
  // since the top-level module is not instantiated.
  J.object([&] {
    J.attribute("instance_name", op.getName());
    J.attribute("module_name", op.getName());
    J.attributeArray("instances", [&] {
      for (auto op : op.getOps<hw::InstanceOp>())
        printHierarchy(op, symbolTable, J);
    });
  });
}

/// Find the modules corresponding to the firrtl mainModule and DesignUnderTest,
/// and if they exist, emit a verbatim op with the module hierarchy for each.
void HWExportModuleHierarchyPass::runOnOperation() {
  mlir::ModuleOp mlirModule = getOperation();
  Optional<SymbolTable> symbolTable = None;
  bool directoryCreated = false;

  for (auto op : mlirModule.getOps<hw::HWModuleOp>()) {
    if (auto attr = op->getAttrOfType<hw::OutputFileAttr>(
            "firrtl.moduleHierarchyFile")) {
      if (!symbolTable)
        symbolTable = SymbolTable(mlirModule);

      if (!directoryCreated) {
        auto error = llvm::sys::fs::create_directory(directoryName);
        if (error) {
          op->emitError("Error creating directory in HWExportModuleHierarchy: ")
              << error.message();
          signalPassFailure();
          return;
        }
        directoryCreated = true;
      }
      SmallString<128> outputPath(directoryName);
      appendPossiblyAbsolutePath(outputPath, attr.getFilename().getValue());
      std::string errorMessage;
      std::unique_ptr<llvm::ToolOutputFile> outputFile(
          mlir::openOutputFile(outputPath, &errorMessage));
      if (!outputFile) {
        op->emitError("Error creating file in HWExportModuleHierarchy: ")
            << errorMessage;
        signalPassFailure();
        return;
      }

      extractHierarchyFromTop(op, symbolTable.getValue(), outputFile->os());

      outputFile->keep();
    }
  }

  markAllAnalysesPreserved();
}

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass>
sv::createHWExportModuleHierarchyPass(StringRef directory) {
  return std::make_unique<HWExportModuleHierarchyPass>(directory);
}
