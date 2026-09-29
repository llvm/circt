import {copyFileSync, existsSync, mkdirSync, rmSync, chmodSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {basename, join, resolve} from 'node:path';

const args = process.argv.slice(2);
const option = (name) => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : undefined;
};
const target = option('--target');
const helper = option('--helper');
const targets = new Set(['darwin-arm64', 'darwin-x64', 'linux-x64']);
if (!target || !targets.has(target) || !helper || !existsSync(helper)) {
  process.stderr.write('usage: npm run package -- --target darwin-arm64|darwin-x64|linux-x64 --helper /path/to/circt-domain-report-server\n');
  process.exit(2);
}

const executable = resolve(helper);
const stage = join('bin', target);
const destination = join(stage, 'circt-domain-report-server');
mkdirSync(stage, {recursive: true});
copyFileSync(executable, destination);
chmodSync(destination, 0o755);
const license = 'LICENSE';
const stagedLicense = !existsSync(license);
if (stagedLicense) copyFileSync(join('..', '..', 'LICENSE'), license);
try {
  const vsce = resolve('node_modules', '.bin', 'vsce');
  const output = `circt-firrtl-domains-${target}.vsix`;
  const result = spawnSync(vsce, ['package', '--no-dependencies', '--target', target,
                                  '--out', output], {stdio: 'inherit'});
  if (result.error) throw result.error;
  if (result.status !== 0) process.exitCode = result.status ?? 1;
  else process.stdout.write(`Created ${basename(output)}\n`);
} finally {
  rmSync(stage, {recursive: true, force: true});
  if (stagedLicense) rmSync(license);
}
