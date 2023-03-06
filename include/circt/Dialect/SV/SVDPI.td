//===- SVDPI.td - SV DPI Interface Ops ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for SystemVerilog verification statements and
// declarations.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/FunctionInterfaces.td"

def DPIImportOp : SVOp<"dpi.import", [
        FunctionOpInterface,
        Symbol,
        HasParent<"mlir::ModuleOp">]> {
  let summary = "Import a DPI-C function";
  let description = [{
    Represents a SV DPI-C import statement. Imports a C function for use.

    ```
    import "DPI-C" function void some_function(
      input  bit[31:0] value_in,
      output bit[31:0] value_out
    )
    ```
  }];

  // TODO: Add support for the `pure` attribute.
  // TODO: Add support for return values.

  let arguments = (ins TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       StrArrayAttr:$argNames,
                       StrArrayAttr:$resultNames);

  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

def DPICallOp : SVOp<"dpi.call", [
        ProceduralOp,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Call a DPI-C function";
  let description = [{
    Invokes a DPI-C function from a procedural block.

    ```
    test(counter + 1, result)
    ```
  }];

  // TODO: turn this into an arbitrary SV expression and support return types.

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>);

  let assemblyFormat = [{
    $callee `(` $args `)` attr-dict`:` functional-type($args, results)
  }];
}
