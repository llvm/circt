//===-- ESIPasses.td - ESI pass definition file ------------*- tablegen -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the ESI dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_ESIPASSES_TD
#define CIRCT_DIALECT_ESI_ESIPASSES_TD

include "mlir/Pass/PassBase.td"

include "circt/Dialect/ESI/ESIDialect.td"

def ESIConnectServices : Pass<"esi-connect-services", "mlir::ModuleOp"> {
  let summary = "Connect up ESI service requests to service providers";
  let constructor = "circt::esi::createESIConnectServicesPass()";
  let dependentDialects = [
    "circt::hw::HWDialect", "circt::sv::SVDialect", "circt::seq::SeqDialect",
    "circt::comb::CombDialect"];
}

def ESIAppIDHier : Pass<"esi-appid-hier", "mlir::ModuleOp"> {
  let summary = "Build an AppID based hierarchy rooted at top module 'top'";
  let constructor = "circt::esi::createESIAppIDHierPass()";
  let options = [
    Option<"top", "top", "std::string",
           "", "Root module of the instance hierarchy">
  ];
}

def ESIBuildManifest : Pass<"esi-build-manifest", "mlir::ModuleOp"> {
  let summary = "Build a manifest of an ESI system";
  let constructor = "circt::esi::createESIBuildManifestPass()";
  let dependentDialects = ["circt::hw::HWDialect", "circt::sv::SVDialect"];
  let options = [
    Option<"top", "top", "std::string",
           "", "Root module of the instance hierarchy">
  ];
}

def ESICleanMetadata : Pass<"esi-clean-metadata", "mlir::ModuleOp"> {
  let summary = "Clean up ESI service metadata";
  let constructor = "circt::esi::createESICleanMetadataPass()";
  let dependentDialects = ["circt::hw::HWDialect"];
}

def LowerESIToPhysical: Pass<"lower-esi-to-physical", "mlir::ModuleOp"> {
  let summary = "Lower ESI abstract Ops to ESI physical ops.";
  let constructor = "circt::esi::createESIPhysicalLoweringPass()";
  let dependentDialects = ["circt::hw::HWDialect"];
}

def LowerESIBundles: Pass<"lower-esi-bundles", "mlir::ModuleOp"> {
  let summary = "Lower ESI bundles to channels.";
  let constructor = "circt::esi::createESIBundleLoweringPass()";
  let dependentDialects = ["circt::hw::HWDialect"];
}

def LowerESIPorts: Pass<"lower-esi-ports", "mlir::ModuleOp"> {
  let summary = "Lower ESI input and/or output ports.";
  let constructor = "circt::esi::createESIPortLoweringPass()";
  let dependentDialects = ["circt::comb::CombDialect", "circt::sv::SVDialect"];
}

def LowerESITypes: Pass<"lower-esi-types", "mlir::ModuleOp"> {
  let summary = "Lower ESI high level types.";
  let constructor = "circt::esi::createESITypeLoweringPass()";
  let dependentDialects = ["circt::hw::HWDialect"];
}

def LowerESItoHW: Pass<"lower-esi-to-hw", "mlir::ModuleOp"> {
  let summary = "Lower ESI to HW where possible and SV elsewhere.";
  let constructor = "circt::esi::createESItoHWPass()";
  let dependentDialects = ["circt::comb::CombDialect", "circt::hw::HWDialect"];
  let options = [
    Option<"platform", "platform", "std::string",
           "", "Target this platform">
  ];
}

#endif // CIRCT_DIALECT_ESI_ESIPASSES_TD
