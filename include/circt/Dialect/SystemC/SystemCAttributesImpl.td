//===- SystemCAttributesImpl.td - SystemC attribute defs ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Attribute definitions for the SystemC dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/EnumAttr.td"

class SystemCAttrDef<string name> : AttrDef<SystemCDialect, name> { }

//===----------------------------------------------------------------------===//
// Port direction utilities
//===----------------------------------------------------------------------===//

def PortDirectionInput  : I32EnumAttrCase<"Input", 0, "sc_in">;
def PortDirectionOutput : I32EnumAttrCase<"Output", 1, "sc_out">;
def PortDirectionInOut  : I32EnumAttrCase<"InOut", 2, "sc_inout">;
let cppNamespace = "circt::systemc" in
let genSpecializedAttr = 0 in
def PortDirection : I32EnumAttr<
  "PortDirection",
  "systemc.module port direction",
  [PortDirectionInput, PortDirectionOutput, PortDirectionInOut]>;


def PortDirectionsAttr : SystemCAttrDef<"PortDirections"> {
  let summary = "Port directions attribute";
  let description = [{
    An attribute to store an array of PortDirection.
  }];

  let mnemonic = "port_directions";
  let parameters = (ins "::mlir::IntegerAttr":$storage);

  let builders = [
    AttrBuilderWithInferredContext<(ins
                                   "::mlir::IntegerAttr":$storage), [{
      return Base::get(storage.getContext(), storage);
    }]>,
    AttrBuilder<(ins "llvm::APInt":$storage), [{
      return Base::get($_ctxt,
                       IntegerAttr::get(
                         IntegerType::get($_ctxt, storage.getBitWidth()),
                       storage));
    }]>,
    AttrBuilder<(ins "llvm::ArrayRef<PortDirection>":$directions)>,
  ];

  let hasCustomAssemblyFormat = true;

  let extraClassDeclaration = [{
    bool isInput(unsigned idx) const {
      return getDirection(idx) == PortDirection::Input;
    }
    bool isOutput(unsigned idx) const {
      return getDirection(idx) == PortDirection::Output;
    }
    bool isInOut(unsigned idx) const {
      return getDirection(idx) == PortDirection::InOut;
    }
    PortDirection getDirection(unsigned index) const;

    void getPortDirections(llvm::SmallVectorImpl<PortDirection> &directions) const;
    size_t getNumPorts() const {
      return getStorage().getValue().getBitWidth() >> 1;
    }
  }];
}
