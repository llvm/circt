const {test} = require('node:test');
const assert = require('node:assert/strict');
const {mkdtempSync, rmSync, writeFileSync} = require('node:fs');
const {tmpdir} = require('node:os');
const {join} = require('node:path');
const {Helper} = require('../out/helper.js');

function fakeProcess(source) {
  const directory = mkdtempSync(join(tmpdir(), 'circt-domain-helper-'));
  const script = join(directory, 'fake.js');
  writeFileSync(script, source);
  return {script, cleanup: () => rmSync(directory, {recursive: true, force: true})};
}

test('helper reassembles lines and returns paged responses', async () => {
  const fake = fakeProcess(`
    process.stdout.write('{"event":"pro');
    process.stdout.write('gress","bytes":5,"total":10}\\n');
    process.stdout.write('{"event":"ready","summary":{"complete":true,"values":2}}\\n');
    let buffer = '';
    process.stdin.on('data', chunk => {
      buffer += chunk;
      while (buffer.includes('\\n')) {
        const end = buffer.indexOf('\\n');
        const request = JSON.parse(buffer.slice(0, end));
        buffer = buffer.slice(end + 1);
        process.stdout.write(JSON.stringify({id: request.id,
          result: {items: [{name: 'A'}], total: 2}}) + '\\n');
      }
    });
  `);
  const progress = [];
  const helper = new Helper(process.execPath, fake.script,
    (bytes, total) => progress.push([bytes, total]), () => {});
  try {
    const summary = await helper.waitReady();
    assert.equal(summary.complete, true);
    const page = await helper.request('listModules', {offset: 0, limit: 1});
    assert.equal(page.total, 2);
    assert.deepEqual(page.items.map(item => item.name), ['A']);
    assert.deepEqual(progress, [[5, 10]]);
  } finally {
    helper.dispose();
    fake.cleanup();
  }
});

test('helper reports load errors', async () => {
  const fake = fakeProcess(`
    process.stdout.write('{"event":"error","message":"unsupported version"}\\n');
  `);
  const helper = new Helper(process.execPath, fake.script, () => {}, () => {});
  try {
    await assert.rejects(helper.waitReady(), /unsupported version/);
  } finally {
    helper.dispose();
    fake.cleanup();
  }
});

test('helper rejects an outstanding request if it exits', async () => {
  const fake = fakeProcess(`
    process.stdout.write('{"event":"ready","summary":{"complete":true}}\\n');
    process.stdin.once('data', () => process.exit(9));
  `);
  const helper = new Helper(process.execPath, fake.script, () => {}, () => {});
  try {
    await helper.waitReady();
    await assert.rejects(helper.request('listModules'), /exited \(9\)/);
  } finally {
    helper.dispose();
    fake.cleanup();
  }
});
