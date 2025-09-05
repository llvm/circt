//===- FIRRTLOpInterfaces.td - Operation Interfaces --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the FIRRTL operation interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLOPINTERFACES_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTLOPINTERFACES_TD

include "mlir/IR/OpBase.td"
include "circt/Dialect/HW/HWOpInterfaces.td"

def FModuleLike : OpInterface<"FModuleLike", [Symbol, PortList, InstanceGraphModuleOpInterface]> {
  let cppNamespace = "circt::firrtl";
  let description = "Provide common module information.";
  let methods = [

    //===------------------------------------------------------------------===//
    // Parameters
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the parameters", "ArrayAttr", "getParameters">,

    //===------------------------------------------------------------------===//
    // Instantiation Convention
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the module's instantiation convention",
      "ConventionAttr", "getConventionAttr">,

    InterfaceMethod<"Get the module's instantiation convention",
      "Convention", "getConvention">,

    //===------------------------------------------------------------------===//
    // Enabled (AKA Required) Layers
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the module's enabled layers",
      "ArrayAttr", "getLayersAttr">,

    InterfaceMethod<"Get the module's enabled layers.",
      "ArrayRef<Attribute>", "getLayers">,

    //===------------------------------------------------------------------===//
    // Port Directions
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the port directions attribute",
    "mlir::DenseBoolArrayAttr", "getPortDirectionsAttr">,

    InterfaceMethod<"Get the port directions",
    "ArrayRef<bool>", "getPortDirections">,

    InterfaceMethod<"Get a port direction",
    "Direction", "getPortDirection", (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      return direction::get($_op.getPortDirections()[portIndex]);
    }]>,

    //===------------------------------------------------------------------===//
    // Port Names
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the port names attribute",
    "ArrayAttr", "getPortNamesAttr">,

    InterfaceMethod<"Get the port names",
    "ArrayRef<Attribute>", "getPortNames">,

    InterfaceMethod<"Get a port name",
    "StringAttr", "getPortNameAttr", (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      return llvm::cast<StringAttr>($_op.getPortNames()[portIndex]);
    }]>,

    InterfaceMethod<"Get a port name",
    "StringRef", "getPortName", (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      return $_op.getPortNameAttr(portIndex).getValue();
    }]>,

    //===------------------------------------------------------------------===//
    // Port Types
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the port types attribute",
    "ArrayAttr", "getPortTypesAttr">,

    InterfaceMethod<"Get the port types",
    "ArrayRef<Attribute>", "getPortTypes">,

    InterfaceMethod<"Set the types of all ports", "void",
    "setPortTypesAttr", (ins "::mlir::ArrayAttr":$portTypes)>,

    InterfaceMethod<"Get a port type",
    "TypeAttr", "getPortTypeAttr", (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      return llvm::cast<TypeAttr>($_op.getPortTypes()[portIndex]);
    }]>,

    InterfaceMethod<"Get a port type",
    "Type", "getPortType", (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      return $_op.getPortTypeAttr(portIndex).getValue();
    }]>,

    //===------------------------------------------------------------------===//
    // Port Annotations
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the port annotations attribute",
    "ArrayAttr", "getPortAnnotationsAttr">,
    InterfaceMethod<"Get the port annotations attribute",
    "ArrayRef<Attribute>", "getPortAnnotations">,

    InterfaceMethod<"Set the port annotations attribute",
    "void", "setPortAnnotationsAttr", (ins "::mlir::ArrayAttr":$annotations)>,


    InterfaceMethod<"Get a port's annotations attribute",
    "ArrayAttr", "getAnnotationsAttrForPort", (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      auto annos = $_op.getPortAnnotationsAttr();
      if (annos.empty())
        return annos; // annos already is an empty array
      return llvm::cast<ArrayAttr>(annos[portIndex]);
    }]>,

    InterfaceMethod<"Get a port's annotations",
    "ArrayRef<Attribute>", "getAnnotationsForPort",
    (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      return $_op.getAnnotationsAttrForPort(portIndex).getValue();
    }]>,

    //===------------------------------------------------------------------===//
    // Port Symbols
    //===------------------------------------------------------------------===//
    // Symbol conventions:
    // 1. Every element of the ArrayAttr(portSymbols) must be NON-NULL.
    // 2. An empty array represents no symbol.
    // 3. The Getters return null if a port sym is invalid.

    // Getters
    InterfaceMethod<"Get the port symbols attribute",
    "ArrayAttr", "getPortSymbolsAttr">,

    InterfaceMethod<"Get the port symbols",
    "ArrayRef<Attribute>", "getPortSymbols">,

    InterfaceMethod<"Get the port symbol attribute",
    "circt::hw::InnerSymAttr", "getPortSymbolAttr", (ins "size_t":$portIndex), [{}], [{
      auto syms = $_op.getPortSymbols();
      if (syms.empty() ||
          llvm::cast<hw::InnerSymAttr>(syms[portIndex]).empty())
        return hw::InnerSymAttr();
      return llvm::cast<hw::InnerSymAttr>(syms[portIndex]);
    }]>,

    // Setters
    InterfaceMethod<"Set the symbols of all ports and their fields", "void",
    "setPortSymbolsAttr", (ins "::mlir::ArrayAttr":$symbols)>,

    InterfaceMethod<"Set the symbols of all ports and their fields", "void",
    "setPortSymbols", (ins "::llvm::ArrayRef<Attribute>":$symbols)>,

    InterfaceMethod<"Set the symbols for a port including its fields", "void",
    "setPortSymbolAttr", (ins "size_t":$portIndex, "circt::hw::InnerSymAttr":$symbol), [{}],
    [{
      SmallVector<Attribute> symbols($_op.getPortSymbols().begin(),
                                     $_op.getPortSymbols().end());
      if (symbols.empty()) {
        if (symbol.empty())
          return;
        symbols.resize($_op.getNumPorts(),
                      hw::InnerSymAttr::get($_op.getContext()));
      }
      assert(symbols.size() == $_op.getNumPorts());
      symbols[portIndex] = symbol;

      FModuleLike::fixupPortSymsArray(symbols, $_op.getContext());
      assert(symbols.empty() || symbols.size() == $_op.getNumPorts());
      $_op.setPortSymbolsAttr(ArrayAttr::get($_op.getContext(), symbols));
    }]>,

    //===------------------------------------------------------------------===//
    // Port Locations
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the port locations attribute",
    "ArrayAttr", "getPortLocationsAttr">,

    InterfaceMethod<"Get the port locations attribute",
    "ArrayRef<Attribute>", "getPortLocations">,

    InterfaceMethod<"Get a port's location attribute",
    "LocationAttr", "getPortLocationAttr", (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      return llvm::cast<LocationAttr>($_op.getPortLocationsAttr()[portIndex]);
    }]>,

    InterfaceMethod<"Get a port's location",
    "Location", "getPortLocation", (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      return $_op.getPortLocationAttr(portIndex);
    }]>,

    //===------------------------------------------------------------------===//
    // Domain Information
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get domain information attribute",
    "ArrayAttr", "getDomainInfoAttr", (ins), /*body=*/[{}],
    /*defaultImplementation=*/[{
      return {};
    }]>,

    InterfaceMethod<
      "Get domain information",
      "ArrayRef<Attribute>",
      "getDomainInfo", (ins), [{
        return $_op.getDomainInfoAttr().getValue();
      }]
    >,

    InterfaceMethod<"Get a port's domain info attribute",
    "Attribute", "getDomainInfoAttrForPort", (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      auto domains = $_op.getDomainInfoAttr();
      if (domains.empty())
        return domains; // domains already is an empty array
      return domains[portIndex];
    }]>,

    InterfaceMethod<"Set the domain info of all ports", "void",
    "setDomainInfoAttr", (ins "::mlir::ArrayAttr":$domainInfo)>,

    //===------------------------------------------------------------------===//
    // All Port Information
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get information about all ports",
    "SmallVector<PortInfo>", "getPorts">,

    InterfaceMethod<"Inserts the given ports at the corresponding indices",
    "void", "insertPorts",
    (ins "ArrayRef<std::pair<unsigned, PortInfo>>":$ports)>,

    InterfaceMethod<"Erases the ports that have their corresponding bit set in `portIndices`",
    "void", "erasePorts",
    (ins "const llvm::BitVector&":$portIndices)>
  ];

  let extraClassDeclaration = [{

    /// Replace NULL entries with an empty InnerSymbol. If all elements of the
    /// array are empty, then it will clear the array.
    static void fixupPortSymsArray(SmallVectorImpl<Attribute> &syms,
                                   MLIRContext *context) {
      // The lack of *any* port symbol is represented by an empty `portSymbols`
      // array as a shorthand.
      if (llvm::none_of(syms, [&](auto &sym) {
            return sym && !cast<hw::InnerSymAttr>(sym).empty();
          })) {
        syms.clear();
        return;
      }
      auto empty = hw::InnerSymAttr::get(context);
      for (auto &sym : syms)
        if (!sym)
          sym = empty;
    }
  }];

  let verify = [{
    static_assert(
        ConcreteOp::template hasTrait<::mlir::SymbolOpInterface::Trait>(),
        "expected operation to be a symbol");
    static_assert(
        ConcreteOp::template hasTrait<OpTrait::InnerSymbolTable>(),
        "expected operation to be an inner symbol table");
    return verifyModuleLikeOpInterface(cast<::circt::firrtl::FModuleLike>(op));
  }];
}

def FInstanceLike : OpInterface<"FInstanceLike", [
  InstanceGraphInstanceOpInterface
]> {
  let cppNamespace = "circt::firrtl";
  let description = "Provide common  instance information.";

  let methods = [
    InterfaceMethod<"Get the name of the instantiated module",
    "::llvm::StringRef", "getReferencedModuleName", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.getModuleName(); }]>,

    InterfaceMethod<"Get the name of the instantiated module",
    "::mlir::StringAttr", "getReferencedModuleNameAttr", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.getModuleNameAttr().getAttr(); }]>,

    InterfaceMethod<"Get the referenced module via a symbol table.",
    "::mlir::Operation *", "getReferencedOperation",
    (ins "const SymbolTable&":$symtbl),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      return symtbl.lookup($_op.getReferencedModuleName());
    }]>,
  ];
}

def ClassLike : OpInterface<"ClassLike", [FModuleLike]> {
  let cppNamespace = "circt::firrtl";
  let description = "Provide common class information.";
  let methods = [
    InterfaceMethod<
      "Get the type for instances of this class",
      "ClassType", "getInstanceType", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return detail::getInstanceTypeForClassLike($_op);
      }]
    >,
    InterfaceMethod<
      "Verify that the given type agrees with this class",
      "LogicalResult", "verifyType", (ins
        "::circt::firrtl::ClassType":$type,
        "::llvm::function_ref<::mlir::InFlightDiagnostic()>":$emitError
      ),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return detail::verifyTypeAgainstClassLike($_op, type, emitError);
      }]
    >
  ];
}

def FConnectLike : OpInterface<"FConnectLike"> {
  let cppNamespace = "circt::firrtl";
  let description = "Provide common connection information.";

  let methods = [
    InterfaceMethod<"Return a destination of connection.",
    "Value", "getDest", (ins)>,
    InterfaceMethod<"Return a source of connection.",
    "Value", "getSrc", (ins)>,
    StaticInterfaceMethod<"Return connection behavior kind.",
    "ConnectBehaviorKind", "getConnectBehaviorKind", (ins), [{}],
     /*defaultImplementation=*/[{ return ConnectBehaviorKind::LastConnect; }]>,
    StaticInterfaceMethod<"Returns true if ConnectBehavior is StaticSingleConnect.",
    "bool", "hasStaticSingleConnectBehavior", (ins),
     /*methodBody=*/[{ return getConnectBehaviorKind() == ConnectBehaviorKind::StaticSingleConnect; }]>,
    StaticInterfaceMethod<"Returns true if ConnectBehavior is LastConnect.",
    "bool", "hasLastConnectBehavior", (ins),
     /*methodBody=*/[{ return getConnectBehaviorKind() == ConnectBehaviorKind::LastConnect; }]>
  ];
}

def FNamableOp : OpInterface<"FNamableOp"> {
  let cppNamespace = "::circt::firrtl";
  let description = [{
    This interface provides common methods for namable operations.
  }];

  let methods = [
    InterfaceMethod<"Return the name.", "::mlir::StringAttr", "getNameAttr">,
    InterfaceMethod<"Return the name.", "::llvm::StringRef", "getName">,
    InterfaceMethod<"Set the name.", "void", "setNameAttr",
      (ins "::mlir::StringAttr":$name)>,
    InterfaceMethod<"Set the name.", "void", "setName",
      (ins "::llvm::StringRef":$name)>,

    InterfaceMethod<"Get the namekind", "::circt::firrtl::NameKindEnumAttr",
      "getNameKindAttr">,
    InterfaceMethod<"Get the namekind", "::circt::firrtl::NameKindEnum",
      "getNameKind">,
    InterfaceMethod<"Set a namekind.", "void", "setNameKindAttr",
      (ins "::circt::firrtl::NameKindEnumAttr":$nameKind)>,
    InterfaceMethod<"Set a namekind.", "void", "setNameKind",
      (ins "::circt::firrtl::NameKindEnum":$nameKind)>,

    InterfaceMethod<"Return true if the name is droppable.",
    "bool", "hasDroppableName", (ins), [{}], /*defaultImplementation=*/[{
      return $_op.getNameKind() == NameKindEnum::DroppableName;
    }]>
  ];
}

def Forceable : OpInterface<"Forceable"> {
  let cppNamespace = "circt::firrtl";
  let description = [{
    Interaction with declarations of forceable hardware components,
    and managing references to them.
  }];

  let methods = [
    InterfaceMethod<"Return true if the operation is forceable.",
    "bool", "isForceable", (ins), [{}], /*defaultImplementation=*/[{
      return $_op.getForceable();
    }]>,
    InterfaceMethod<"Return data value that will be targeted.",
    "mlir::TypedValue<FIRRTLBaseType>", "getData", (ins), [{}], /*defaultImplementation=*/[{
      return llvm::cast<mlir::TypedValue<FIRRTLBaseType>>($_op.getDataRaw());
    }]>,
    InterfaceMethod<"Return raw data value that will be targeted.",
    "Value", "getDataRaw", (ins), [{}], /*defaultImplementation=*/[{
      return $_op.getResult();
    }]>,
    InterfaceMethod<"Return data type that will be referenced.",
    "FIRRTLBaseType", "getDataType", (ins), [{}], /*defaultImplementation=*/[{
      return firrtl::type_cast<FIRRTLBaseType>($_op.getDataRaw().getType());
    }]>,
    InterfaceMethod<"Return reference result, or null if not active.",
    "mlir::TypedValue<RefType>", "getDataRef", (ins), [{}], /*defaultImplementation=*/[{
      return llvm::cast_or_null<mlir::TypedValue<RefType>>($_op.getRef());
    }]>,
  ];
  let extraClassDeclaration = [{
    /// Attribute name for 'forceable' tracking.
    static StringRef getForceableAttrName() { return "forceable"; }
  }];
  let verify = [{
    return detail::verifyForceableOp(cast<::circt::firrtl::Forceable>($_op));
  }];
}
#endif // CIRCT_DIALECT_FIRRTL_FIRRTLOPINTERFACES_TD
