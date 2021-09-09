//===- MSFTPasses.td - MSFT dialect passes -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

def RunGenerators: Pass<"run-generators", "mlir::ModuleOp"> {
  let summary = "Run the MSFT design entry generators";
  let description = [{
    Generators allow designers to model their system with high-level constructs
    (which they define), then lower them down into synthesizable code.
    High-level constructs are modeled as MLIR Operations. Generators are
    essentially lowerings of those operations and they are registered along
    with the name of an MLIR operation which they lower. Generators also have a
    name so multiple can be registered per operation then one of them selected
    by some criteria, allowing the designer to choose a particular
    implementation of the same logical feature.
  }];
  let constructor = "circt::msft::createRunGeneratorsPass()";
  let options = [
    ListOption<"generators", "generators", "std::string",
               "List of possible generators to run.",
               "llvm::cl::OneOrMore, llvm::cl::MiscFlags::CommaSeparated">
  ];
}
