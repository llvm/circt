//===- SystemCStatements.td - SystemC statements ops -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for SystemC statements.
//
//===----------------------------------------------------------------------===//

def SignalWriteOp : SystemCOp<"signal.write", [
    TypesMatchWith<"dest has to be the base type of the src",
                   "dest", "src", "systemc::getBaseType($_self)">]> {
  let summary = "Declare two SSA values equivalent.";
  let description = [{
    This operation does not model a specific construct in the SystemC spec, but
    is required to 'connect' an SSA value to a module output SSA value. It can,
    however, also be used to declare any two SSA values to be aliases which
    means that ExportSystemC will generate at most one variable name for all
    SSA values in the alias set.
  }];

  let arguments = (ins AnyTypeOf<[OutputType, InOutType, SignalType]>:$dest, AnyType:$src);
  let assemblyFormat = "$dest `,` $src attr-dict `:` type($dest)";
}

def SignalOp : SystemCOp<"signal", [HasCustomSSAName, SystemCNameDeclOpInterface,
                                    HasParent<"SCModuleOp">]> {
  let summary = "Declares a SystemC `sc_signal<T>`.";
  let description = [{
    Represents the `sc_signal` template as described in IEEE 1666-2011 §6.4.
  }];

  let arguments = (ins StrAttr:$name);
  let results = (outs SignalType:$signal);

  let assemblyFormat = [{
    custom<ImplicitSSAName>($name) attr-dict `:` type($signal)
  }];
}

def MethodOp : SystemCOp<"method", []> {
  let summary = "Represents the SystemC SC_METHOD macro.";
  let description = [{
    Represents the SC_METHOD macro as described in IEEE 1666-2011 §5.2.9.
  }];

  let arguments = (ins FuncHandleType:$funcHandle);
  let assemblyFormat = "$funcHandle attr-dict";
}

def ThreadOp : SystemCOp<"thread", []> {
  let summary = "Represents the SystemC SC_THREAD macro.";
  let description = [{
    Represents the SC_THREAD macro as described in IEEE 1666-2011 §5.2.9.
  }];

  let arguments = (ins FuncHandleType:$funcHandle);
  let assemblyFormat = "$funcHandle attr-dict";
}
