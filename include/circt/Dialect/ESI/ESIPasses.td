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

def ESIConnectServices : Pass<"esi-connect-services", "mlir::ModuleOp"> {
  let summary = "Connect up ESI service requests to service providers";
  let constructor = "circt::esi::createESIConnectServicesPass()";
  let dependentDialects = [
    "circt::hw::HWDialect", "circt::sv::SVDialect", "circt::seq::SeqDialect",
    "circt::comb::CombDialect"];
}

def ESIEmitCollateral: Pass<"esi-emit-collateral", "mlir::ModuleOp"> {
  let summary = "Emit all the neccessary collateral";
  let constructor = "circt::esi::createESIEmitCollateralPass()";
  let dependentDialects = ["circt::sv::SVDialect"];
  let options = [
    Option<"schemaFile", "schema-file", "std::string",
           "", "File to output capnp schema into">,
    ListOption<"tops", "tops", "std::string",
               "List of top modules to export Tcl for",
               "llvm::cl::ZeroOrMore,">
  ];
}

def LowerESIToPhysical: Pass<"lower-esi-to-physical", "mlir::ModuleOp"> {
  let summary = "Lower ESI abstract Ops to ESI physical ops.";
  let constructor = "circt::esi::createESIPhysicalLoweringPass()";
}

def LowerESIPorts: Pass<"lower-esi-ports", "mlir::ModuleOp"> {
  let summary = "Lower ESI input and/or output ports.";
  let constructor = "circt::esi::createESIPortLoweringPass()";
  let dependentDialects = ["circt::comb::CombDialect", "circt::sv::SVDialect"];
}

def LowerESItoHW: Pass<"lower-esi-to-hw", "mlir::ModuleOp"> {
  let summary = "Lower ESI to HW where possible and SV elsewhere.";
  let constructor = "circt::esi::createESItoHWPass()";
  let dependentDialects = ["circt::comb::CombDialect", "circt::hw::HWDialect"];
}

def ESIWrapCombinational: Pass<"esi-wrap-combinational", "mlir::ModuleOp"> {
  let summary = "Materialize dynamically scheduled wrapper module around combinational module";
  let description = [{
    Todo: write a good description
  }];
  let constructor = "circt::esi::createESIWrapCombinationalPass()";
  let dependentDialects = ["circt::comb::CombDialect", "circt::hw::HWDialect"];
}

def ESIWrapPipeline: Pass<"esi-wrap-pipeline", "mlir::ModuleOp"> {
  let summary = "Materialize dynamically scheduled wrapper module around statically scheduled pipeline module";
  let description = [{
    Todo: write a good description
  }];
  let constructor = "circt::esi::createESIWrapPipelinePass()";
  let dependentDialects = ["circt::comb::CombDialect", "circt::hw::HWDialect", "circt::seq::SeqDialect"];
}

def ESIWrapCalyxPipeline: Pass<"esi-wrap-calyx-pipeline", "mlir::ModuleOp"> {
  let summary = "Materialize a wrapper module that wraps a calyx pipeline interface into a general pipeline interface";
  let description = [{
    Todo: write a good description
  }];
  let constructor = "circt::esi::createESIWrapCalyxPipelinePass()";
  let dependentDialects = ["circt::comb::CombDialect", "circt::hw::HWDialect"];
}


#endif // CIRCT_DIALECT_ESI_ESIPASSES_TD
