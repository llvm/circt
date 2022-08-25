//===-- Passes.td - Transforms pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes within the Transforms/ directory.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TRANSFORMS_PASSES
#define CIRCT_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def FlattenMemRef : Pass<"flatten-memref", "::mlir::ModuleOp"> {
  let summary = "Flatten memrefs";
  let description = [{
    Flattens multidimensional memories and accesses to them into
    single-dimensional memories.}];
  let constructor = "circt::createFlattenMemRefPass()";
  let dependentDialects = ["mlir::memref::MemRefDialect"];
}

def FlattenMemRefCalls : Pass<"flatten-memref-calls", "::mlir::ModuleOp"> {
  let summary = "Flatten memref calls";
  let description = [{
    Flattens calls to functions which have multidimensional memrefs as arguments.
    This is done by casting memref arguments through memref.subview operations.
    Any called functions which had their type signatures changes will be replaced
    by a private function definition with the new signature.
    It is up to users of this pass to define how these rewritten functions are
    to be implemented.}];
  let constructor = "circt::createFlattenMemRefCallsPass()";
  let dependentDialects = ["mlir::memref::MemRefDialect"];
}

def StripDebugInfoWithPred : Pass<"strip-debuginfo-with-pred", "::mlir::ModuleOp"> {
  let summary = "Selectively strip debug info from all operations";

  let description = [{
    This pass extends mlir::StrinpDebugInfoPass to selectively strip locations with a
    given predicate.
  }];
  let constructor = "circt::createStripDebugInfoWithPredPass({})";
  let options = [
    Option<"dropSuffix", "drop-suffix", "std::string",
           /*default=*/"",
           "Drop file location info with the specified suffix. This option is"
           "intended to be used for testing."> ];
}

def Feedthrough : Pass<"feedthrough", "::mlir::ModuleOp"> {
  let summary = "Add feedthrough ports to specified modules";
  let description = [{
    Add feedthrough ports to specified modules and instances by starting at a
    specified module and list of ports. These ports will be punched through to
    output ports. These outputs will then be punched through a list of modules
    to become inputs and outputs, fed directly through. Finally, the punched
    through signals will be wired to the specified destination.
  }];
  let constructor = "circt::createFeedthroughPass()";
  let options = [
    Option<"sourceModule", "source-module", "std::string", /*default=*/"",
           "Source module to feed signals from through.">,
    ListOption<"sourcePorts", "source-ports", "std::string",
           "Source signals to feed through.",
	   "llvm::cl::OneOrMore">,
    ListOption<"innerModules", "inner-modules", "std::string",
           "Inner modules to feed signals through.",
	   "llvm::cl::ZeroOrMore">,
    Option<"destModule", "dest-module", "std::string", /*default=*/"",
           "Destination module to feed signals to.">
  ];
}

#endif // CIRCT_TRANSFORMS_PASSES
