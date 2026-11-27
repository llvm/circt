//===- DebugTypes.td - Debug dialect types -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DEBUG_DEBUGTYPES_TD
#define CIRCT_DIALECT_DEBUG_DEBUGTYPES_TD

include "circt/Dialect/Debug/DebugDialect.td"
include "mlir/IR/AttrTypeBase.td"

class DebugTypeDef<string name> : TypeDef<DebugDialect, name> { }
class DebugAttrDef<string name> : AttrDef<DebugDialect, name> { }

def ScopeType : DebugTypeDef<"Scope"> {
  let mnemonic = "scope";
  let summary = "debug scope";
  let description = "The result of a `dbg.scope` operation.";
}

def StructType : DebugTypeDef<"Struct"> {
  let mnemonic = "struct";
  let summary = "debug struct aggregate";
  let description = "The result of a `dbg.struct` operation.";
}

def ArrayType : DebugTypeDef<"Array"> {
  let mnemonic = "array";
  let summary = "debug array aggregate";
  let description = "The result of a `dbg.array` operation.";
}

def ValueType : DebugTypeDef<"Value"> {
  let mnemonic = "value";
  let summary = "debug value annotated with source-language metadata";
  let description = "The result of a `dbg.value` operation.";
}

def EnumType : DebugTypeDef<"Enum"> {
  let mnemonic = "enum";
  let summary = "debug enum value";
  let description = "The result of a `dbg.enum` operation.";
}

def InstanceAttr : DebugAttrDef<"Instance"> {
  let mnemonic = "instance";
  let summary = "source-level description of an instance";
  let description = [{
    Describes the instance it sits on. The `dbg.instance` attribute of an
    instance holds a list of these, usually one. Each gives the source `name`,
    the source-language class `typeName` and `ext`, language data such as
    constructor parameters, as on `dbg.value`. The instance itself gives the
    module, so passes that rename or move it keep the description by keeping
    its attributes. A pass that drops them loses the class, never mislabels it.

    After dedup merged modules, the instance has one entry per merged module.
    `origin` names the module an entry comes from, absent for the module the
    instance is in. `module` names the source module of the instance if
    dedup merged it into the one the instance targets.

    `path` names the inlined instances, outermost first, whose bodies the
    instance now sits in, once the inliner has moved it into a parent module.
    The names are those of the module dedup kept.
    With a `path`, `origin` names the module the instance was in.
  }];
  let parameters = (ins
    "::mlir::StringAttr":$name,
    OptionalParameter<"::mlir::StringAttr">:$module,
    OptionalParameter<"::mlir::StringAttr">:$typeName,
    OptionalParameter<"::mlir::DictionaryAttr">:$ext,
    OptionalParameter<"::mlir::ArrayAttr">:$path,
    OptionalParameter<"::mlir::StringAttr">:$origin
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // CIRCT_DIALECT_DEBUG_DEBUGTYPES_TD
