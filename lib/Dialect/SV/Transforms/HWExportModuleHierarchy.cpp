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
  HWExportModuleHierarchyPass(std::optional<std::string> directory) {
    if (directory)
      directoryName = *directory;
  }
  void runOnOperation() override;
};

/// Recursively print the module hierarchy as serialized as JSON.
static void printHierarchy(hw::InstanceOp &inst, SymbolTable &symbolTable,
                           llvm::json::OStream &j) {
  auto moduleOp = symbolTable.lookup(inst.getModuleNameAttr().getValue());

  j.object([&] {
    j.attribute("instance_name", inst.instanceName());
    j.attribute("module_name", hw::getVerilogModuleName(moduleOp));
    j.attributeArray("instances", [&] {
      // Only recurse on module ops, not extern or generated ops, whose internal
      // are opaque.
      if (auto module = dyn_cast<hw::HWModuleOp>(moduleOp)) {
        for (auto op : module.getOps<hw::InstanceOp>()) {
          printHierarchy(op, symbolTable, j);
        }
      }
    });
  });
}

/// Return the JSON-serialized module hierarchy for the given module as the top
/// of the hierarchy.
static void extractHierarchyFromTop(hw::HWModuleOp op, SymbolTable &symbolTable,
                                    llvm::raw_ostream &os) {
  llvm::json::OStream j(os, 2);

  // As a special case for top-level module, set instance name to module name,
  // since the top-level module is not instantiated.
  j.object([&] {
    j.attribute("instance_name", op.getName());
    j.attribute("module_name", hw::getVerilogModuleName(op));
    j.attributeArray("instances", [&] {
      for (auto op : op.getOps<hw::InstanceOp>())
        printHierarchy(op, symbolTable, j);
    });
  });
}

/// Find the modules corresponding to the firrtl mainModule and DesignUnderTest,
/// and if they exist, emit a verbatim op with the module hierarchy for each.
void HWExportModuleHierarchyPass::runOnOperation() {
  mlir::ModuleOp mlirModule = getOperation();
  std::optional<SymbolTable> symbolTable;
  bool directoryCreated = false;

  for (auto op : mlirModule.getOps<hw::HWModuleOp>()) {
    auto attr = op->getAttrOfType<ArrayAttr>("firrtl.moduleHierarchyFile");
    if (!attr)
      continue;
    for (auto file : attr.getAsRange<hw::OutputFileAttr>()) {
      if (!symbolTable)
        symbolTable = SymbolTable(mlirModule);

      if (!directoryCreated) {
        auto error = llvm::sys::fs::create_directories(directoryName);
        if (error) {
          op->emitError("Error creating directory in HWExportModuleHierarchy: ")
              << error.message();
          signalPassFailure();
          return;
        }
        directoryCreated = true;
      }
      SmallString<128> outputPath(directoryName);
      appendPossiblyAbsolutePath(outputPath, file.getFilename().getValue());
      std::string errorMessage;
      std::unique_ptr<llvm::ToolOutputFile> outputFile(
          mlir::openOutputFile(outputPath, &errorMessage));
      if (!outputFile) {
        op->emitError("Error creating file in HWExportModuleHierarchy: ")
            << errorMessage;
        signalPassFailure();
        return;
      }

      extractHierarchyFromTop(op, *symbolTable, outputFile->os());

      outputFile->keep();
    }
  }

  markAllAnalysesPreserved();
}

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass>
sv::createHWExportModuleHierarchyPass(std::optional<std::string> directory) {
  return std::make_unique<HWExportModuleHierarchyPass>(directory);
}
