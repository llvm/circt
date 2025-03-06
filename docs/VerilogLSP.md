# CIRCT Verilog LSP
This document describes the custom commands for the Verilog LSP server.

## Custom Commands

### `verilog/putInlayHintsOnObjects`

This is a custom protocol that allows clients to provide inlay hints for specific objects in Verilog source files. These hints will be displayed inline in the editor.

Example Verilog code where hints can be applied:
```verilog
module test();
    wire foo;  // hint will appear here
endmodule
```

The command accepts the following interface for hint specifications:
```typescript
interface VerilogUserProvidedInlayHint {
  // Required: Path to the object (e.g., signal name)
  path: string;
  
  // Required: The hint text to display
  value: string;
  
  // Optional: Root module name containing the object.
  root?: string;
  
  // Optional: Group name for organizing hints. Hints with the same group
  // can be updated together - new hints will replace old ones in the same group
  group?: string;
}

interface VerilogUserProvidedInlayHintParams {
  hints: VerilogUserProvidedInlayHint[];
}
```

Example JSON-RPC request:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "verilog/putInlayHintsOnObjects",
  "params": {
    "hints": [
      {
        "path": "foo",
        "root": "test",
        "value": "hint for foo",
        "group": "simulation"
      }
    ]
  }
}
```

When processed, this will display "hint for foo" as an inlay hint next to the `foo` wire declaration in the `test` module.

The `group` field allows for dynamic updating of hints. When new hints are sent with the same group name, they will replace any existing hints in that group. This is useful for updating hints based on changing conditions, such as simulation values or analysis results. Hints without a group are treated as permanent and won't be replaced by subsequent hint updates.

For example, if the client sends the following hints, the hint for `foo` will be updated to "new hint for foo".
```json
{
  "hints": [
    { "path": "foo", "value": "new hint for foo", "group": "simulation" }
  ]
}
```
