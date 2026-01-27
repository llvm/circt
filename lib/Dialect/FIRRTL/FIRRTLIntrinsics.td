//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the table of FIRRTL intrinsics supported by CIRCT. This is distinct
// from the MLIR ops representing some of these intrinsics in
// `include/circt/Dialect/FIRRTL/FIRRTLIntrinsics.td`.
//
//===----------------------------------------------------------------------===//

// A FIRRTL intrinsic. Defs of this class generate part of the intrinsic
// lowering code, as well as fragments for the dialect documentation.
class FIRRTLIntrinsic {
  // The mnemonic for the intrinsic, e.g. "circt.sizeof". This should use the
  // form with dots. A second mnemonic with only underscores will be generated
  // automatically.
  string mnemonic;

  // The C++ class name (potentially including template parameters) that
  // converts the generic intrinsic to a concrete op during lowering. Defaults
  // to `<DefName>Converter`.
  code converter = ?;

  // A description of the intrinsic for documentation purposes. Markdown.
  string description;
}

//===----------------------------------------------------------------------===//
// Intrinsics
//===----------------------------------------------------------------------===//

def CirctSizeof : FIRRTLIntrinsic {
  let mnemonic = "circt.sizeof";
  let description = [{
    Returns the size of a type.  The input port is not read from and may be any
    type, including uninferred types.

    | Argument | Type | Description                                |
    | -------- | ---- | ------------------------------------------ |
    | i        | Any  | value whose type's size is to be returned  |

    | Result | Type     | Description       |
    | ------ | -------- | ----------------- |
    | size   | UInt<32> | Size of type of i |
  }];
}

def CirctIsX : FIRRTLIntrinsic {
  let mnemonic = "circt.isX";
  let description = [{
    Tests if the value is a literal `x`.  FIRRTL doesn't have a notion of X
    per-se, but x can come in to the system from external modules and from SV
    constructs. Verification constructs need to explicitly test for X.

    | Argument | Type | Description   |
    | -------- | ---- | ------------- |
    | i        | Any  | value to test |

    | Result | Type    | Description |
    | ------ | ------- | ----------- |
    | found  | UInt<1> | i is `x`    |
  }];
}
