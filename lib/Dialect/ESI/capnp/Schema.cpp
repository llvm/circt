//===- Schema.cpp - ESI Cap'nProto schema utilities -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "ESICapnp.h"

#include "circt/Dialect/ESI/ESITypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt::esi;
using namespace circt::esi::capnp;

TypeSchema::TypeSchema(Type type) {
  ChannelPort chan = type.dyn_cast<ChannelPort>();
  if (chan)
    this->type = chan.getInner();
  else
    this->type = type;
}

// Compute the expected size of the capnp message field in bits. Return -1 on
// non-representable type.
static ssize_t getCapnpMsgFieldSize(Type type) {
  return llvm::TypeSwitch<::mlir::Type, int64_t>(type)
      .Case<IntegerType>([](IntegerType t) {
        auto w = t.getWidth();
        if (w == 1)
          return 8;
        else if (w <= 8)
          return 8;
        else if (w <= 16)
          return 16;
        else if (w <= 32)
          return 32;
        else if (w <= 64)
          return 64;
        return -1;
      })
      .Default([](Type) { return -1; });
}

// Compute the expected size of the capnp message in bits. Return -1 on
// non-representable type. TODO: replace this with a call into the Capnp C++
// library to parse a schema and have it compute sizes and offsets.
ssize_t TypeSchema::size() {
  ssize_t headerSize = 128;
  ssize_t fieldSize = getCapnpMsgFieldSize(type);
  if (fieldSize < 0)
    return fieldSize;
  // Capnp sizes are always multiples of 8 bytes, so round up.
  fieldSize = (fieldSize & ~0x3f) + (fieldSize & 0x3f ? 0x40 : 0);
  return headerSize + fieldSize;
}
