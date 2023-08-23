##===- CosimDpi.capnp - ESI cosim RPC schema ------------------*- CAPNP -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//
##
## The ESI low level cosimulation RPC Cap'nProto schema.
##
##===----------------------------------------------------------------------===//

@0xd9df719e13276339;

interface LLCosimServer @0xe30046acabda2ef9 {
  # Write the the ESI MMIO space. Throws an exception on an error.
  writeMMIO @0 (addr :UInt64, data :UInt64) -> ();

  # Read from the ESI MMIO space at an address. Return data on success or throws
  # an exception on error.
  readMMIO @1 (addr :UInt64) -> (data :UInt64);
}
