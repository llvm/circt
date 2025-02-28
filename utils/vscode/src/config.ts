import * as vscode from 'vscode';

/**
 *  Gets the config value `circt-verilog-lsp.<key>`, with an optional workspace
 * folder.
 */
export function get<T>(key: string,
                       workspaceFolder: vscode.WorkspaceFolder = null,
                       defaultValue: T = undefined): T {
  return vscode.workspace.getConfiguration('circt-verilog-lsp', workspaceFolder)
      .get<T>(key, defaultValue);
}

/**
 *  Sets the config value `circt-verilog-lsp.<key>`.
 */
export function update<T>(key: string, value: T,
                          target?: vscode.ConfigurationTarget) {
  return vscode.workspace.getConfiguration('circt-verilog-lsp')
      .update(key, value, target);
}
