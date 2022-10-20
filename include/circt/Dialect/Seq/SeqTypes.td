//===- SeqTypes.td - Seq data type definitions -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Data types for the Seq dialect.
//
//===----------------------------------------------------------------------===//

class SeqType<string name> : TypeDef<SeqDialect, name> { }

class MemoryPortType<string name, string _mnemonic> : TypeDef<SeqDialect, name # "Port"> {
  let parameters = (ins "hw::ArrayType":$memoryType);
  let assemblyFormat = [{
      `<` $memoryType `>`
  }];
  let mnemonic = _mnemonic;
}

def ReadPortType : MemoryPortType<"Read", "read_port"> {
  let summary = "Read port type";
  let description = [{
      This type represents a reference to a read port of a memory. The type
      serves as a link between the owner of the port (the instantiated memory)
      and the user of the port (some `seq` read port operation).
  }];
}

def WritePortType : MemoryPortType<"Write", "write_port"> {
  let summary = "Write port type";
  let description = [{
      This type represents a reference to a write port of a memory. The type
      serves as a link between the owner of the port (the instantiated memory)
      and the user of the port (some `seq` write port operation).
  }];
}
