include "mlir/IR/OpBase.td"

/// Op Interface for cells.
def CellOpInterface : OpInterface<"CellInterface"> {
  let cppNamespace = "::circt::calyx";

  let description = [{
    This is an op interface for Calyx Cells. Cells consist
    of primitives and instances of components.
  }];

  let methods = [
    InterfaceMethod<
      "This returns the port names associated with the cell.",
      "SmallVector<StringRef>", "portNames"
    >,
    InterfaceMethod<
      "This returns the port directions associated with the cell.",
      "SmallVector<circt::calyx::Direction>", "portDirections"
    >
  ];

  let verify = [{
    return verifyCell(op);
  }];
}
