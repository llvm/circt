import * as chokidar from 'chokidar';
import * as vscode from 'vscode';

import {CIRCTContext} from './circtContext';
import * as config from './config';

/**
 *  Prompt the user to see if we should restart the server.
 */
async function promptRestart(settingName: string, promptMessage: string) {
  switch (config.get<string>(settingName)) {
  case 'restart':
    vscode.commands.executeCommand('circt-verilog-lsp.restart');
    break;
  case 'ignore':
    break;
  case 'prompt':
  default:
    switch (await vscode.window.showInformationMessage(
        promptMessage, 'Yes', 'Yes, always', 'No, never')) {
    case 'Yes':
      vscode.commands.executeCommand('circt-verilog-lsp.restart');
      break;
    case 'Yes, always':
      vscode.commands.executeCommand('circt-verilog-lsp.restart');
      config.update<string>(settingName, 'restart',
                            vscode.ConfigurationTarget.Global);
      break;
    case 'No, never':
      config.update<string>(settingName, 'ignore',
                            vscode.ConfigurationTarget.Global);
      break;
    default:
      break;
    }
    break;
  }
}

/**
 *  Activate watchers that track configuration changes for the given workspace
 *  folder, or null if the workspace is top-level.
 */
export async function activate(
    circtContext: CIRCTContext, workspaceFolder: vscode.WorkspaceFolder,
    serverSettings: string[], serverPaths: string[]) {
  // When a configuration change happens, check to see if we should restart the
  // server.
  circtContext.subscriptions.push(vscode.workspace.onDidChangeConfiguration(event => {
    for (const serverSetting of serverSettings) {
      const expandedSetting = `circt-verilog-lsp.${serverSetting}`;
      if (event.affectsConfiguration(expandedSetting, workspaceFolder)) {
        promptRestart(
            'onSettingsChanged',
            `setting '${
                expandedSetting}' has changed. Do you want to reload the server?`);
      }
    }
  }));

  // Setup watchers for the provided server paths.
  const fileWatcherConfig = {
    disableGlobbing : true,
    followSymlinks : true,
    ignoreInitial : true,
    awaitWriteFinish : true,
  };
  for (const serverPath of serverPaths) {
    if (serverPath === '') {
      return;
    }

    // If the server path actually exists, track it in case it changes.
    const fileWatcher = chokidar.watch(serverPath, fileWatcherConfig);
    fileWatcher.on('all', (event, _filename, _details) => {
      if (event != 'unlink') {
        promptRestart(
            'onSettingsChanged',
            'circt-verilog-lsp-server file has changed. Do you want to reload the server?');
      }
    });
    circtContext.subscriptions.push(
        new vscode.Disposable(() => { fileWatcher.close(); }));
  }
}
