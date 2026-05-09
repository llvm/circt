//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_SYNTHATTRIBUTES_TD
#define CIRCT_DIALECT_SYNTH_SYNTHATTRIBUTES_TD


include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"

// A single resolved 2D NLDM lookup table. Values are stored in row-major order
// with dimensions index1.size x index2.size.
def NLDMTableAttr : AttrDef<Synth_Dialect, "NLDMTable"> {
  let mnemonic = "nldm_table";
  let summary = "Resolved 2D NLDM lookup table";
  let parameters = (ins
    "::mlir::DenseF64ArrayAttr":$index1,
    "::mlir::DenseF64ArrayAttr":$index2,
    "::mlir::DenseF64ArrayAttr":$values
  );

  let assemblyFormat = "`<` $index1 `,` $index2 `,` $values `>`";
}

def NLDMTimingArcAttr : AttrDef<Synth_Dialect, "NLDMTimingArc"> {
  let mnemonic = "nldm_timing_arc";
  let summary = "NLDM timing arc with delay and transition lookup tables";
  let parameters = (ins
    "::mlir::StringAttr":$pin,
    "::mlir::StringAttr":$relatedPin,
    "::mlir::StringAttr":$timingSense,
    "NLDMTableAttr":$cellRise,
    "NLDMTableAttr":$cellFall,
    "NLDMTableAttr":$riseTransition,
    "NLDMTableAttr":$fallTransition
  );
  let assemblyFormat =
    "`<` $pin `,` $relatedPin `,` $timingSense `,` "
    "$cellRise `,` $cellFall `,` "
    "$riseTransition `,` $fallTransition `>`";
}


#endif // CIRCT_DIALECT_SYNTH_SYNTHATTRIBUTES_TD
