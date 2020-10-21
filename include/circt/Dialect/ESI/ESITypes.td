//===- ESITypes.td - ESI data type definitions -------------*- tablegen -*-===//
//
// Data types for the ESI dialect.
//
//===----------------------------------------------------------------------===//



// -------------
// ESI port specifications
//
// ESI ports are the primary mechanism on which the ESI dialect operates. They
// represent an ESI communication flows of various kinds. At first, ESI will
// only support 'channels' -- unidirectional, point-to-point,
// latency-insensitive streams.
// -------------
class ESI_Port<string name> : TypeDef<ESI_Dialect, name> {
  let cppClassName = name # "Port";
}

def Channel : ESI_Port<"Channel"> {
  let summary = "An ESI-compatible channel port";
  let description = [{
    An ESI port kind which models a latency-insensitive, unidirectional,
    point-to-point data stream. Channels are typed (like all of ESI). Said
    type can be any MLIR type, but must be lowered to something a backend
    knows how to output (i.e. something emitVerilog knows about).
  }];

  let mnemonic = "channel";
  let parameters = (
    ins
    "Type":$inner
  );
}
