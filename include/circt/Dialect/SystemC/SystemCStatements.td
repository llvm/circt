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

def InstanceDeclOp : SystemCOp<"instance.decl", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<HWInstanceLike>,
  HasCustomSSAName,
  SystemCNameDeclOpInterface,
  HasParent<"SCModuleOp">
]> {
  let summary = "Declares a SystemC module instance.";
  let description = [{
    Declares an instantiation of a SystemC module inside another SystemC module
    by value. The instance handle returned by this operation can then be used
    to initialize it in the constructor and to access its fields.
    More information can be found in IEEE 1666-2011 §4.1.1.
  }];

  let arguments = (ins StrAttr:$name, FlatSymbolRefAttr:$moduleName);
  let results = (outs ModuleType:$instanceHandle);

  let assemblyFormat = [{
    custom<ImplicitSSAName>($name) $moduleName attr-dict
    `:` type($instanceHandle)
  }];

  let builders = [
    OpBuilder<(ins "StringAttr":$instanceName,
                   "FlatSymbolRefAttr":$moduleRef,
                   "ArrayRef<ModuleType::PortInfo>":$ports), [{
      build($_builder, $_state,
            ModuleType::get($_builder.getContext(), moduleRef.getAttr(), ports),
            instanceName, moduleRef);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Get the systemc::ModuleType that represents the instantiated module.
    systemc::ModuleType getInstanceType() {
      return getInstanceHandle().getType().cast<systemc::ModuleType>();
    }

    StringRef getInstanceName() { return getName(); }
    StringAttr getInstanceNameAttr() { return getNameAttr(); }

    /// Lookup the module for the symbol. This returns null on invalid IR.
    Operation *getReferencedModule(const hw::HWSymbolCache *cache);

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    /// An InstanceDeclOp may not optionally define a symbol.
    bool isOptionalSymbol() { return false; }
  }];
}

def BindPortOp : SystemCOp<"instance.bind_port", [HasParent<"CtorOp">]> {
  let summary = "Binds a port of a module instance to a channel.";
  let description = [{
    The ports of a submodule have to be bound to channels in a module further
    up in the instance hierarchy (as opposed to `sc_export` where the channel
    has to reside in the same module or a submodule). Therefore, a port can be
    bound to either a signal declared by the `systemc.signal` operation or to
    a port with matching direction (and thus bound to a channel further up in
    the hierarchy).
    More information on ports can be found in IEEE 1666-2011 §5.12.,
    in particular IEEE 1666-2011 §5.12.7. is about port binding.
    More information on predefined channels can be found in IEEE 1666-2011 §6.
  }];

  let arguments = (ins ModuleType:$instance,
                       IndexAttr:$portId,
                       ChannelType:$channel);

  let extraClassDeclaration = [{
    /// Get the name of the port to be bound.
    StringRef getPortName();
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operations that model C++-level features.
//===----------------------------------------------------------------------===//

def DeleteOp : SystemCOp<"cpp.delete", []> {
  let summary = "A C++ delete expression.";
  let description = [{
    Destroys objects previously allocated by the new expression and releases
    the allocated memory.
  }];

  let arguments = (ins EmitC_PointerType:$pointer);
  let assemblyFormat = "$pointer attr-dict `:` qualified(type($pointer))";
}
