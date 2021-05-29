//===- MSFT.td - MSFT dialect definition -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the MSFT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MSFT_TD
#define MSFT_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Pass/PassBase.td"

def MSFTDialect : Dialect {
  let name = "msft";

  let summary = "Microsoft internal support dialect";
  let description = [{
    Umbrella dialect for everything needed to support Microsoft development
    but not thoroughly discussed. Most (if not everything) in this dialect is
    a candidate for generalization and re-homing.
  }];

  let hasConstantMaterializer = 1;
  let cppNamespace = "::circt::msft";
  let extraClassDeclaration = [{
    /// Register all MSFT attributes.
    void registerAttributes();

    void registerGenerator(StringRef opName, StringRef generatorName,
                           GeneratorCallback cb);

    /// Generator details don't need to be exposed.
    llvm::ManagedStatic<detail::Generators> generators;
  }];
}

// Base class for the operation in this dialect.
class MSFTOp<string mnemonic, list<OpTrait> traits = []> :
    Op<MSFTDialect, mnemonic, traits>;

def DeviceType : I32EnumAttr<"DeviceType",
  "Type of device at physical location", [
    I32EnumAttrCase<"M20K", 1>,
    I32EnumAttrCase<"DSP", 2>,
  ]>;

class MSFT_Attr<string name, list<Trait> traits = [],
                string baseCppClass = "::mlir::Attribute">
    : AttrDef<MSFTDialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

def PhysLocation : MSFT_Attr<"PhysLocation"> {
  let summary = "Descibes a physical location on a device";
  let description = [{
    Annotate a particular entity within an op with the location of the device
    on an FPGA to which it should mapped. The coordinates in this attribute
    are absolute locations on the device, so if there are two instances of a
    module with this annotation incorrect results will be generated. How to
    solve this is a more general, open problem.
  }];
  let mnemonic = "physloc";
  let parameters = (ins
    "DeviceTypeAttr":$devType,
    "uint64_t":$x, "uint64_t":$y, "uint64_t":$num);
}

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
}

#endif // MSFT_TD
