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
  test @0 () -> ();
}
