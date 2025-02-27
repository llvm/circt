# CIRCT-Verilog-LSP

The CIRCT-Verilog-LSP provides language IDE features for Verilog and SystemVerilog.
CIRCT-Verilog-LSP is built on top of the [Slang](https://github.com/MikePopoloski/slang) and MLIR LSP library.

## `.verilog` - (System) Verilog:

### Features

- Syntax highlighting for `.v` and `.sv` files and markdown blocks
- go-to-definition and cross references
- Cross reference CIRCT emitted locations 

### Configuration

* For the basic features, only the `circt-verilog-lsp.verilog_server_path` setting is required. Set this to the path of the `circt-verilog-lsp-server` executable.

* To cross reference CIRCT emitted locations, the `circt-verilog-lsp.verilog_source_location_root_directories` setting is required. For example, chisel user could set this to the path of the directory containing the chisel sources.

* The language server automatically includes files in the same directory as the file being edited. To manually include additional RTL directories, use the `circt-verilog-lsp.verilog_include_directories` setting.

#### Diagnostics

The language server runs diagnostics ran by slang.

##### Find definition

Jump to the definition of symbol under the cursor. For variables it jumps to the declaration. For module instantiation, it jumps to the module definition.

#### Cross-references to CIRCT emitted locations

CIRCT emits location information `@[loc]` in the SV. CIRCT-Verilog-LSP can navigate to these locations. It's necessary to configure the `circt-verilog-lsp.verilog_source_locaGtion_root_directories` setting to the root directory of the design.

## Development

To build the extension, run `vsce package`.
