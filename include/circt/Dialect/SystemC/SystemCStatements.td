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
                   "dest", "src", "systemc::getSignalBaseType($_self)">]> {
  let summary = "Writes a value to a signal or port.";
  let description = [{
    Represents the member function `void write(const T&);` and several variants
    of the operator `operator=` of class `sc_signal` as described in
    IEEE 1666-2011 §6.4.8., of class `sc_inout` as described in §6.10.3., and of
    class `sc_out` as decribed in §6.12.3. It shall modify the value of the
    signal/port such that it appears to have the new value (as observed using
    the `sytemc.signal.read` operation) in the next delta cycle but not before
    then.
  }];

  let arguments = (ins AnyTypeOf<[OutputType, InOutType, SignalType]>:$dest,
                       AnyType:$src);
  let assemblyFormat = "$dest `,` $src attr-dict `:` type($dest)";
}

def SignalOp : SystemCOp<"signal", [HasCustomSSAName,
                                    SystemCNameDeclOpInterface,
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

  let arguments = (ins NullaryVoidFunctionType:$funcHandle);
  let assemblyFormat = "$funcHandle attr-dict";
}

def ThreadOp : SystemCOp<"thread", []> {
  let summary = "Represents the SystemC SC_THREAD macro.";
  let description = [{
    Represents the SC_THREAD macro as described in IEEE 1666-2011 §5.2.9.
  }];

  let arguments = (ins NullaryVoidFunctionType:$funcHandle);
  let assemblyFormat = "$funcHandle attr-dict";
}
