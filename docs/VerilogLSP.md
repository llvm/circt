# CIRCT Verilog LSP

## Custom Commands

### `verilog/putInlayHintsOnObjects`

This is a custom protocl that allows clients to provide inlay hints on specific objects.
```verilog
module test();
    wire foo;
endmodule
```
```
interface VerilogUserProvidedInlayHint {
  path: string;
  string: value;
  string?: root;
  string?: group;
}
```
```json


interface VerilogUserProvidedInlayHintParams {
  array<VerilogUserProvidedInlayHint> hints;
}

{"jsonrpc":"2.0","id":1,"method":"verilog/putInlayHintsOnObjects","params": {
  "hints": [
    {
      "path": "foo",
      "root": "test",
      "value": "hint for foo"
    }
  ]
  }
}
```

It's unimplemented yet but this