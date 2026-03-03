//===- RTGTestAttributes.td - RTGTest attributes -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the RTGTest attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTGTEST_IR_RTGTESTATTRIBUTES_TD
#define CIRCT_DIALECT_RTGTEST_IR_RTGTESTATTRIBUTES_TD

include "circt/Dialect/RTGTest/IR/RTGTestDialect.td"
include "circt/Dialect/RTG/IR/RTGAttributes.td"
include "circt/Dialect/RTG/IR/RTGInterfaces.td"
include "circt/Dialect/RTG/IR/RTGISAAssemblyInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

class RTGTestAttrDef<string name, list<Trait> traits = []> 
  : AttrDef<RTGTestDialect, name, traits>;

def CPUAttr : RTGTestAttrDef<"CPU", [ContextResourceAttrInterface]> {
  let summary = "this attribute represents a CPU referred to by the core ID";

  let parameters = (ins "size_t":$id);

  let mnemonic = "cpu";
  let assemblyFormat = "`<` $id `>`";

  let extraClassDeclaration = [{
    // TypedAttrInterface
    Type getType() const;
  }];
}

class IntegerRegisterAttrBase<string cppName, string name, int classIndex>
  : RTGTestAttrDef<cppName, [RegisterAttrInterface]> {

  let mnemonic = name;

  let extraClassDeclaration = [{
    unsigned getClassIndex() const {
      return }] # classIndex # [{;
    }

    llvm::StringLiteral getRegisterAssembly() const {
      return "}] # name # [{";
    }

    Type getType() const {
      return IntegerRegisterType::get(getContext());
    }
  }];
}

def RegZeroAttr : IntegerRegisterAttrBase<"RegZero", "zero", 0>;
def RegRaAttr   : IntegerRegisterAttrBase<"RegRa", "ra", 1>;
def RegSpAttr   : IntegerRegisterAttrBase<"RegSp", "sp", 2>;
def RegGpAttr   : IntegerRegisterAttrBase<"RegGp", "gp", 3>;
def RegTpAttr   : IntegerRegisterAttrBase<"RegTp", "tp", 4>;
def RegT0Attr   : IntegerRegisterAttrBase<"RegT0", "t0", 5>;
def RegT1Attr   : IntegerRegisterAttrBase<"RegT1", "t1", 6>;
def RegT2Attr   : IntegerRegisterAttrBase<"RegT2", "t2", 7>;
def RegS0Attr   : IntegerRegisterAttrBase<"RegS0", "s0", 8>;
def RegS1Attr   : IntegerRegisterAttrBase<"RegS1", "s1", 9>;
def RegA0Attr   : IntegerRegisterAttrBase<"RegA0", "a0", 10>;
def RegA1Attr   : IntegerRegisterAttrBase<"RegA1", "a1", 11>;
def RegA2Attr   : IntegerRegisterAttrBase<"RegA2", "a2", 12>;
def RegA3Attr   : IntegerRegisterAttrBase<"RegA3", "a3", 13>;
def RegA4Attr   : IntegerRegisterAttrBase<"RegA4", "a4", 14>;
def RegA5Attr   : IntegerRegisterAttrBase<"RegA5", "a5", 15>;

class FloatRegisterAttrBase<string cppName, string name, int classIndex>
  : RTGTestAttrDef<cppName, [RegisterAttrInterface]> {

  let mnemonic = name;

  let extraClassDeclaration = [{
    unsigned getClassIndex() const {
      return }] # classIndex # [{;
    }

    llvm::StringLiteral getRegisterAssembly() const {
      return "}] # name # [{";
    }

    Type getType() const {
      return FloatRegisterType::get(getContext());
    }
  }];
}

def RegF0Attr   : FloatRegisterAttrBase<"RegF0", "f0", 0>;

#endif // CIRCT_DIALECT_RTGTEST_IR_RTGTESTATTRIBUTES_TD
