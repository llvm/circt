//===- MSFTAttributes.td - MSFT attribute definitions ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// "Forward-declare" these HW attributes rather than including or duplicating
// them here. This lets us to refer to them in ODS, but delegates to HW in C++.
// These are used to represent parameters to MSFTModuleExternOp.
def ParamDeclAttr : Attr<CPred<"$_self.isa<hw::ParamDeclAttr>()">>;
def ParamDeclArrayAttr : TypedArrayAttrBase<ParamDeclAttr, "parameter array">;

def PrimitiveType : I32EnumAttr<"PrimitiveType",
  "Type of device at physical location", [
    I32EnumAttrCase<"M20K", 1>,
    I32EnumAttrCase<"DSP", 2>,
    I32EnumAttrCase<"FF", 3>,
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
    "PrimitiveTypeAttr":$primitiveType,
    "uint64_t":$x, "uint64_t":$y, "uint64_t":$num);
}

def PhysicalBounds : MSFT_Attr<"PhysicalBounds"> {
  let summary = "Describes a rectangle bounding a physical region on a device";
  let description = [{
    Describes a rectangular bound within a device. The lower and upper bounds
    must be specified for both the X and Y axis. The bounds are inclusive.
  }];
  let mnemonic = "physical_bounds";
  let parameters = (ins
    "uint64_t":$xMin,
    "uint64_t":$xMax,
    "uint64_t":$yMin,
    "uint64_t":$yMax
  );
}

def PhysicalBoundsArray : TypedArrayAttrBase<PhysicalBounds,
  "array of PhysicalBounds">;

def PhysicalRegionRef : MSFT_Attr<"PhysicalRegionRef"> {
  let summary = "Describes a reference to a physical region on a device";
  let description = [{
    Annotate a particular entity within an op with the region of the devices
    on an FPGA to which it should mapped. The physicalRegion must refer to a
    PhysicalRegion operation.
  }];
  let mnemonic = "physical_region_ref";
  let parameters = (ins
    "FlatSymbolRefAttr":$physicalRegion
  );
}
