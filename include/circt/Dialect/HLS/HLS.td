//===- HLS.td - Definition of the HLS dialect -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HLS_TD
#define CIRCT_DIALECT_HLS_TD

include "mlir/IR/OpBase.td"

def HLSDialect : Dialect {
  let name = "hls";
  let summary = "Attributes for the HLS dialect";
  let description = [{
    This dialect defines the `HLS` dialect, which is intended to represent
    typical HLS attributes and transformations.
  }];
  let cppNamespace = "::circt::hls";
}

// Dummy dialect attribute for triggering dialect attribute parser/printer
// declarations in TableGen.
def HLS_DummyAttr : DialectAttr<HLSDialect, CPred<"$_self.isa<HLSAttr>()">,
                              "HLS dummy attribute"> {}

//===----------------------------------------------------------------------===//
// HLS directives
//
// An HLS directive is an attribute used for guiding the HLS compilation flow.
// HLS directives are named structs which may contain mandatory or optional fields.
//
// The attributes defined through HLS.td are used to generate functionality such
// as attribute storage, printing and parsing. These base classes are then
// wrapped to realize a true dialect attribute (see HLSDirective in HLSDialect.h).
//
// Rationale:
// By wrapping attributes into dialect attributes we force the parser
// to delegate parsing to the dialect. Through this, we're able to parse and
// instantiate the actual struct attributes rather than letting the parser 
// default to parsing a dictionary.
//
// Usage:
// format: #hls.$keyword<{...}>
// - with {...} being the struct attribute of the directive.
// - All $keyword's are listed in HLSDialect.h
//
// Restrictions/todo:
// - Currently, the tablegen backend for EnumAttr's cannot automatically parse
//   enum names. Given this, the enum value must be used in textual IR.
//===----------------------------------------------------------------------===//


//===- Pipeline directive -------------------------------------------------===//
// Attempts to pipeline a function or loop. A pipelined function or loop can
// process new inputs every $II cycles (initiation interval).

/// Pipeline style:
/// - stp: Stall pipeline; run when input data is available, else, stall.
/// - frp: Free-running pipeline; always runs, even when input data is not available.
/// - flp: Flushable pipeline.
def PipelineStyleSTP : I64EnumAttrCase<"stp", 0>;
def PipelineStyleFLP : I64EnumAttrCase<"frp", 1>;
def PipelineStyleFRP : I64EnumAttrCase<"flp", 2>;
def PipelineStyleAttr : I64EnumAttr<
  "PipelineStyle", "",
  [ PipelineStyleSTP, PipelineStyleFLP, PipelineStyleFRP ]> {
  let cppNamespace = "::circt::hls";
}

def PipelineDirective : StructAttr<"PipelineDirective", HLSDialect,
  [
    // Target initiation interval (II); 
    StructFieldAttr<"II", OptionalAttr<I64Attr>>,

    // Pipeline style
    StructFieldAttr<"style", OptionalAttr<PipelineStyleAttr>>
  ]>;

//===- Function instantiate directive -------------------------------------===//
// The Function Instantiate directive is used to create a unique implementation
// for each instance of a function, allowing each instance to be optimized
// according to the function call.
def FunctionInstantiateDirective : StructAttr<"FunctionInstantiateDirective", HLSDialect,
  [
    /// variable: The function argument to be used as a constant in the implementation.
    StructFieldAttr<"variable", StrAttr>
  ]>;

#endif // CIRCT_DIALECT_HLS_TD
