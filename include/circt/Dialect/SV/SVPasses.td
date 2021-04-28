//===-- SVPasses.td - SV pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the SV dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_SVPASSES
#define CIRCT_DIALECT_SV_SVPASSES

include "mlir/Pass/PassBase.td"

def RTLCleanup : Pass<"rtl-cleanup", "rtl::RTLModuleOp"> {
  let summary = "Cleanup transformations for operations in rtl.module bodies";
  let description = [{
      This pass merges sv.alwaysff operations with the same condition, sv.ifdef
      nodes with the same condition, and perform other cleanups for the IR.
      This is a good thing to run before Verilog emission.
  }];

  let constructor = "circt::sv::createRTLCleanupPass()";
}

def RTLStubExternalModules : Pass<"rtl-stub-external-modules", 
                                  "mlir::ModuleOp"> {
  let summary = "transform external rtl modules to empty rtl modules";
  let description = [{
      This pass creates empty module bodies for external modules.  This is
      useful for linting to eliminate missing file errors.
  }];

  let constructor = "circt::sv::createRTLStubExternalModulesPass()";
  let dependentDialects = ["circt::sv::SVDialect"];
}

def RTLLegalizeNames : Pass<"rtl-legalize-names", "ModuleOp"> {
  let summary = "Legalize module/variable names in rtl.modules and their bodies";
  let description = [{
    This pass modifies the names of modules and names such that they do not
    cause conflicts with SV keywords. This is a good thing to run before
    Verilog emission.
  }];

  let constructor = "circt::sv::createRTLLegalizeNamesPass()";
}

def RTLGeneratorCalloutPass : Pass<"rtl-generator-callout", "ModuleOp"> {
  let summary = "Lower Generator Schema to external module";
  let description = [{
    This pass calls an external program for all the rtl.module.generated nodes.
    The corresponding rtl.generator.schema node is used for the port names, 
    the value for each port name is parsed from the attribute on the rtl.module.generated nodes.
    The module is then declared as an external module, whose definition is present 
    in a file with the module name.
  }];
  let constructor = "circt::sv::createRTLGeneratorCalloutPass()";

  let options = [
    Option<"schemaName", "schema-name", "std::string",
                "", "Name of the schema to process">,
    Option<"genExecutable", "generator-executable", "std::string",
                "", "Generator program executable with optional full path name">,
    Option<"genExecArgs", "generator-executable-arguments", "std::string",
                "", "Generator program arguments separated by ;">,
    Option<"genExecOutFileName", "generator-executable-output-filename", "std::string",
                "", "Name of the output file to be used by the generator program to dump the filenames in which the module definition is generated.">
   ];
}

def RTLMemSimImpl : Pass<"rtl-memory-sim", "ModuleOp"> {
  let summary = "Implement FIRRTMMem memories nodes with simulation model";
  let description = [{
    This pass replaces generated module nodes of type FIRRTLMem with a model
    suitable for simulation.
  }];

  let constructor = "circt::sv::createRTLMemSimImplPass()";
  let dependentDialects = ["circt::sv::SVDialect"];
}


#endif // CIRCT_DIALECT_SV_SVPASSES
