//===- MSFTPDOps.td - MSFT physical design operations ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/HW/HWTypes.td"

def DeclPhysicalRegionOp : MSFTOp<"physical_region",
    [Symbol, HasParent<"mlir::ModuleOp">]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    PhysicalBoundsArray:$bounds
  );

  let assemblyFormat = [{
    $sym_name `,` $bounds attr-dict
  }];
}

def EntityExternOp : MSFTOp<"entity.extern",
    [Symbol, HasParent<"mlir::ModuleOp">]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    AnyAttr:$metadata);
  let assemblyFormat = [{
    $sym_name $metadata attr-dict
  }];
}

def PDPhysLocationOp : MSFTOp<"pd.location",
      [DeclareOpInterfaceMethods<DynInstDataOpInterface>]> {
  let summary = "Specify a location for an instance";
  let description = [{
    Used to specify a specific location on an FPGA to place a dynamic instance.
    Supports specifying the location of a subpath for extern modules and device
    primitives. Intended to live as a child of `instance.dynamic` initially
    without the `ref` field. The dynamic instance lowering will fill in `ref`
    with the symol of the `hw.globalref` op corresponding to the lowered dynamic
    instance.
  }];
  let arguments = (ins PhysLocation:$loc,
                       OptionalAttr<StrAttr>:$subPath,
                       OptionalAttr<FlatSymbolRefAttr>:$ref);
  let assemblyFormat = [{
    ($ref^)? custom<PhysLoc>($loc) (`path` `:` $subPath^)? attr-dict
  }];
}

def PDPhysRegionOp : MSFTOp<"pd.physregion",
      [DeclareOpInterfaceMethods<DynInstDataOpInterface>]> {
  let summary = "Specify a physical region for an instance";
  let description = [{
    Annotate a particular entity within an op with the region of the devices
    on an FPGA to which it should mapped. The physRegionRef must refer to a
    DeclPhysicalRegion operation.
  }];
  let arguments = (ins FlatSymbolRefAttr:$physRegionRef,
                       OptionalAttr<StrAttr>:$subPath,
                       OptionalAttr<FlatSymbolRefAttr>:$ref);
  let assemblyFormat = [{
    ($ref^)? $physRegionRef (`path` `:` $subPath^)? attr-dict
  }];
}

def InstanceHierarchyOp : MSFTOp<"instance.hierarchy",
                                 [HasParent<"mlir::ModuleOp">, NoTerminator]> {
  let summary = "The root of an instance hierarchy";

  let arguments = (ins FlatSymbolRefAttr:$topModuleRef);
  let regions = (region SizedRegion<1>:$instances);

  let assemblyFormat = [{
    $topModuleRef $instances attr-dict
  }];
}

def DynamicInstanceOp : MSFTOp<"instance.dynamic",
                               [ParentOneOf<["circt::msft::InstanceHierarchyOp",
                                             "circt::msft::DynamicInstanceOp"]>,
                                NoTerminator]> {

  let summary = "A module instance in the instance hierarchy";
  let description = [{
    Represents an instance (as in instance in the instance hierarchy) referred
    to henceforth as a dynamic instance. Specified with a path through the
    instance hierarchy (which in the future will be replaced with an AppID).
    Lowers to a `hw.globalref` but unlike a global ref, does not require all of
    the ops participating in the globalref to contain a back pointer attribute.
    Allows users to efficiently add placements to a large number of dynamic
    instances which happen to map to a small number of static instances by
    bulk-adding the necessary `hw.globalref` attributes.

    During the lowering, moves the operations in the body to the top level and
    gives them the symbol of the globalref which was created to replace the
    dynamic instance.
  }];
  let arguments = (ins HWInnerRefAttr:$instanceRef);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    custom<ImplicitInnerRef>($instanceRef) $body attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::ArrayAttr globalRefPath();
  }];
}
