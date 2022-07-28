//===- ESIInterfaces.td - ESI Interfaces -----------------*- tablegen -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the interfaces in the ESI dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

def ChannelOpInterface : OpInterface<"ChannelOpInterface"> {
  let description =
      [{"An interface for operations which carries channel semantics."}];

  let methods = [
    InterfaceMethod<
      [{"Returns the channel type of this operation."}],
      "circt::esi::ChannelType", "channelType"
    >,
    InterfaceMethod<
        "Returns true if this channel carries data.",
        "bool",
        "hasData",
        (ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return $_op.channelType().hasData();
        }]>,
    InterfaceMethod<
        [{"Returns the inner type of this channel. This will be the type of the
           data value of the channel, if the channel carries data semantics. Else,
           return NoneType."}],
        "mlir::Type",
        "innerType",
        (ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          if(!$_op.hasData())
            return mlir::NoneType::get($_op.getContext());
          return $_op.channelType().getInner();
        }]>
  ];
}
