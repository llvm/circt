//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CreateDesignConfigPackage pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/FIRRTLInstanceInfo.h"
#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/OM/OMOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_CREATEDESIGNCONFIGPACKAGE
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {

/// Structure to hold information about a design configuration
struct ConfigInfo {
  StringAttr name;
  int64_t defaultValue;
  StringAttr comment;
  Type resultType;
  Location loc;
};

class CreateDesignConfigPackagePass
    : public circt::firrtl::impl::CreateDesignConfigPackageBase<
          CreateDesignConfigPackagePass> {
public:
  void runOnOperation() override;

private:
  /// Generate the SystemVerilog package containing all configurations
  void generateConfigPackage(CircuitOp circuit, ArrayRef<ConfigInfo> configs);

  /// Generate OM class metadata for the configurations
  void generateOMMetadata(CircuitOp circuit, ArrayRef<ConfigInfo> configs);

  /// Schema class for individual configuration parameters
  ClassOp configSchemaClass;

  /// Metadata class containing list of all configurations
  ClassOp configMetadataClass;
};

} // namespace

void CreateDesignConfigPackagePass::generateConfigPackage(
    CircuitOp circuit, ArrayRef<ConfigInfo> configs) {
  if (configs.empty())
    return;

  auto builder = mlir::ImplicitLocOpBuilder::atBlockEnd(circuit.getLoc(),
                                                        circuit.getBodyBlock());

  // Build the package content
  std::string packageContent = "package DesignConfigPackage;\n";

  for (const auto &config : configs) {
    // Add comment if present and non-empty
    if (config.comment && !config.comment.getValue().empty()) {
      // FIXME: Properly indent if a comment contains a new lines.
      packageContent += "  // " + config.comment.getValue().str() + "\n";
    }

    // Generate parameter declaration with default value
    packageContent += "  parameter " + config.name.getValue().str() + " = " +
                      std::to_string(config.defaultValue) + ";\n";
  }

  packageContent += "endpackage\n";

  // Create a verbatim op with the package content
  auto verbatimOp = sv::VerbatimOp::create(builder, packageContent);
  verbatimOp->setAttr("output_file",
                      hw::OutputFileAttr::getFromFilename(
                          &getContext(), "DesignConfigPackage.sv",
                          /*excludeFromFileList=*/false,
                          /*includeReplicatedOps=*/false));
}

void CreateDesignConfigPackagePass::generateOMMetadata(
    CircuitOp circuit, ArrayRef<ConfigInfo> configs) {
  if (configs.empty())
    return;

  auto unknownLoc = mlir::UnknownLoc::get(&getContext());
  auto builder = mlir::ImplicitLocOpBuilder::atBlockEnd(unknownLoc,
                                                        circuit.getBodyBlock());

  // Create the schema class for individual configuration parameters
  // Fields: name, defaultValue, comment, width
  StringRef schemaFieldNames[] = {"name", "defaultValue", "comment", "width"};
  Type schemaFieldTypes[] = {
      StringType::get(&getContext()), FIntegerType::get(&getContext()),
      StringType::get(&getContext()), FIntegerType::get(&getContext())};

  configSchemaClass = ClassOp::create(builder, "DesignConfigSchema",
                                      schemaFieldNames, schemaFieldTypes);

  // Create the metadata class that will hold all configuration objects
  SmallVector<PortInfo> metadataPorts;
  configMetadataClass = ClassOp::create(
      builder, builder.getStringAttr("DesignConfigMetadata"), metadataPorts);

  builder.setInsertionPointToStart(configMetadataClass.getBodyBlock());

  // Create an object for each configuration and assign its fields
  SmallVector<Value> configObjects;
  for (const auto &config : configs) {
    // Create an object instance of the schema class
    auto configObj = ObjectOp::create(builder, configSchemaClass, config.name);

    // Assign field values using ObjectSubfieldOp to access input ports
    // Port indices: 0=name_in, 1=name_out, 2=defaultValue_in,
    // 3=defaultValue_out, etc.

    // Field 0: name (input port index 0)
    auto nameInPort = ObjectSubfieldOp::create(builder, configObj, 0);
    auto nameConst = StringConstantOp::create(builder, config.name);
    PropAssignOp::create(builder, nameInPort, nameConst);

    // Field 1: defaultValue (input port index 2)
    auto defaultValueInPort = ObjectSubfieldOp::create(builder, configObj, 2);
    auto defaultValueAttr = builder.getIntegerAttr(
        builder.getIntegerType(64, /*isSigned=*/true), config.defaultValue);
    auto defaultValueConst =
        FIntegerConstantOp::create(builder, defaultValueAttr);
    PropAssignOp::create(builder, defaultValueInPort, defaultValueConst);

    // Field 2: comment (input port index 4)
    auto commentInPort = ObjectSubfieldOp::create(builder, configObj, 4);
    auto commentConst = StringConstantOp::create(
        builder, config.comment ? config.comment : builder.getStringAttr(""));
    PropAssignOp::create(builder, commentInPort, commentConst);

    // Field 3: width (input port index 6)
    auto widthInPort = ObjectSubfieldOp::create(builder, configObj, 6);
    int32_t width = cast<IntType>(config.resultType).getWidth().value_or(0);
    auto widthAttr = builder.getIntegerAttr(
        builder.getIntegerType(64, /*isSigned=*/true), width);
    auto widthConst = FIntegerConstantOp::create(builder, widthAttr);
    PropAssignOp::create(builder, widthInPort, widthConst);

    configObjects.push_back(configObj);
  }

  // Create a list of all configuration objects
  auto listType = ListType::get(
      &getContext(), cast<PropertyType>(detail::getInstanceTypeForClassLike(
                         configSchemaClass)));
  auto configList = ListCreateOp::create(builder, listType, configObjects);

  // Add the list as an output port of the metadata class
  auto portIndex = configMetadataClass.getNumPorts();
  SmallVector<std::pair<unsigned, PortInfo>> newPorts = {
      {portIndex, PortInfo(builder.getStringAttr("configs"),
                           configList.getType(), Direction::Out)}};
  configMetadataClass.insertPorts(newPorts);
  auto blockarg = configMetadataClass.getBodyBlock()->addArgument(
      configList.getType(), unknownLoc);
  PropAssignOp::create(builder, blockarg, configList);
}

void CreateDesignConfigPackagePass::runOnOperation() {
  auto circuit = getOperation();
  auto &instanceInfo = getAnalysis<InstanceInfo>();

  auto dut = instanceInfo.getEffectiveDut();

  // Walk all modules in the circuit
  SmallVector<ConfigInfo> configs;
  bool hasErrors = false;

  for (auto module : circuit.getOps<FModuleOp>()) {
    // Check if this module is under the DUT
    bool isUnderDUT = dut && instanceInfo.anyInstanceInEffectiveDesign(module);

    module.walk([&](ReadDesignConfigIntIntrinsicOp op) {
      if (!isUnderDUT) {
        // Emit error if config is used outside of DUT
        // PR NOTE: The requirement here is not to generate a package when
        // compiling testbench (since otherwise there are two configuration
        // packages). How SiFiveMetadata pass avoids this problem?
        op.emitError() << "design configuration '" << op.getParamName()
                       << "' can only be used within the effective DUT";
        hasErrors = true;
      } else {
        // Collect configuration info
        ConfigInfo info{
            op.getParamNameAttr(), static_cast<int64_t>(op.getDefaultValue()),
            op.getCommentAttr(), op.getResult().getType(), op.getLoc()};
        configs.push_back(info);
        ++numConfigs;
      }
    });
  }

  if (hasErrors) {
    signalPassFailure();
    return;
  }

  // TODO: Check the uniqueness of parameters.
  //       We want to allow modules to refer to the same parameters across
  //       the design hierarchy, but we need to ensure they are referring to
  //       the same parameters. In addition to the parameter name, the type,
  //       default value, and comment must all be the same.

  // Generate the configuration package
  generateConfigPackage(circuit, configs);

  // Generate OM metadata
  generateOMMetadata(circuit, configs);
}
