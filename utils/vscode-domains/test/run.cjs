const {downloadAndUnzipVSCode, runTests} = require('@vscode/test-electron');
const {existsSync} = require('node:fs');
const path = require('node:path');

const options = {
  extensionDevelopmentPath: path.resolve(__dirname, '..'),
  extensionTestsPath: path.resolve(__dirname, 'suite', 'index.cjs'),
  launchArgs: ['--disable-extensions']
};

async function main() {
  const downloaded = await downloadAndUnzipVSCode(options);
  // Newer macOS archives name the executable Code. Older test-electron
  // releases still return a path ending in Electron.
  options.vscodeExecutablePath = process.platform === 'darwin' &&
    !existsSync(downloaded) ? path.join(path.dirname(downloaded), 'Code') :
    downloaded;
  // An extension host can pass these variables to this runner. They make a
  // new Electron process behave like Node instead of starting VS Code.
  delete process.env.ELECTRON_RUN_AS_NODE;
  for (const key of Object.keys(process.env))
    if (key.startsWith('VSCODE_'))
      delete process.env[key];
  await runTests(options);
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
