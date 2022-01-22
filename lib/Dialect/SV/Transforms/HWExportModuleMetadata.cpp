//===- HWExportModuleMetadata.cpp - Export Module Metadata ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Export module and port information to JSON.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
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

struct HWExportModuleMetadataPass
    : public sv::HWExportModuleMetadataBase<HWExportModuleMetadataPass> {
  HWExportModuleMetadataPass(Optional<std::string> fileName) {
    if (fileName.hasValue())
      this->fileName = fileName.getValue();
  }
  void runOnOperation() override;
};

/// Add metadata for module 'op'. Namely, the module name, list of ports, and
/// total bits of input/output/inout ports.
static void emitMetadataAbout(hw::HWModuleOp op, llvm::json::OStream &J,
                              SmallVectorImpl<Attribute> &symbols) {

  J.object([&] {
    J.attribute("module_name", SymbolTable::getSymbolName(op).getValue());
    uint64_t inputBits = 0;
    uint64_t outputBits = 0;
    uint64_t inoutBits = 0;
    J.attributeArray("ports", [&] {
      for (auto port : op.getAllPorts()) {
        J.object([&] {
          uint64_t bits = hw::getBitWidth(port.type);
          StringRef dir;

          switch (port.direction) {
          case hw::PortDirection::INPUT:
            dir = "input";
            inputBits += bits;
            break;
          case hw::PortDirection::OUTPUT:
            dir = "output";
            outputBits += bits;
            break;
          case hw::PortDirection::INOUT:
            dir = "inout";
            inoutBits += bits;
            break;
          }
          J.attribute("name", port.getName());
          J.attribute("direction", dir);
          J.attribute("bits", bits);
        });
      }
    });
    J.attribute("input_bits", inputBits);
    J.attribute("output_bits", outputBits);
    J.attribute("inout_bits", inoutBits);
  });
}

/// Construct metadata JSON, then stuff it into a VerbatimOp.
void HWExportModuleMetadataPass::runOnOperation() {
  mlir::ModuleOp mlirModule = getOperation();

  SmallVector<Attribute> symbols;
  std::string jsonString;
  llvm::raw_string_ostream jsonTextStream(jsonString);
  llvm::json::OStream J(jsonTextStream, 4);
  J.object([&] {
    J.attributeArray("modules", [&] {
      for (auto op : mlirModule.getOps<hw::HWModuleOp>()) {
        emitMetadataAbout(op, J, symbols);
      }
    });
  });

  auto b = OpBuilder::atBlockEnd(mlirModule.getBody());
  auto verbatim = b.create<sv::VerbatimOp>(
      b.getUnknownLoc(), jsonString, ValueRange{}, b.getArrayAttr(symbols));
  auto outputFileAttr =
      hw::OutputFileAttr::getFromFilename(b.getContext(), fileName);
  verbatim->setAttr("output_file", outputFileAttr);

  markAllAnalysesPreserved();
}

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass>
sv::createHWExportModuleMetadataPass(Optional<std::string> fileName) {
  return std::make_unique<HWExportModuleMetadataPass>(fileName);
}
