const assert = require('node:assert/strict');
const {chmodSync, mkdtempSync, rmSync, writeFileSync} = require('node:fs');
const {tmpdir} = require('node:os');
const {join} = require('node:path');
const vscode = require('vscode');
const {DomainsView, Explorer, openSource} = require('../../out/extension.js');

async function testPaging() {
  const explorer = new Explorer();
  explorer.summary = {complete: true};
  explorer.report = vscode.Uri.file('/tmp/report.json');
  explorer.helper = {
    request: async (method, params) => {
      assert.equal(method, 'listModules');
      const offset = params.offset;
      const count = Math.min(params.limit, 203 - offset);
      return {total: 203, items: Array.from({length: count}, (_, i) => ({
        id: String(offset + i), name: `Module${offset + i}`, kind: 'module',
        ports: 0, values: 0, instances: 0
      }))};
    }
  };
  const roots = await explorer.getChildren();
  const modules = roots.find(node => node.kind === 'modules');
  const first = await explorer.getChildren(modules);
  assert.equal(first.length, 201);
  assert.equal(first.at(-1).kind, 'page');
  const second = await explorer.getChildren(first.at(-1));
  assert.equal(second.length, 3);
  assert.equal(second[0].data.name, 'Module200');
}

async function testSourceNavigation() {
  const directory = mkdtempSync(join(tmpdir(), 'circt-domain-source-'));
  const filename = join(directory, 'design.scala');
  try {
    writeFileSync(filename, 'first\nsecond line\n');
    await openSource({display: 'test range', sources: [{
      file: 'design.scala', line: 2, column: 2, endLine: 2, endColumn: 5,
      paths: [filename]
    }]});
    const editor = vscode.window.activeTextEditor;
    assert.equal(editor.document.uri.fsPath, filename);
    assert.equal(editor.selection.start.line, 1);
    assert.equal(editor.selection.start.character, 1);
    assert.equal(editor.selection.end.character, 4);
  } finally {
    rmSync(directory, {recursive: true, force: true});
  }
}

async function testDomainsView() {
  const domains = new DomainsView();
  domains.summary = {complete: true};
  const edge = {
    index: 0, kind: 'association', lhs: 'portA', rhs: 'dom',
    lhsModule: 'M', rhsModule: 'M', inferred: false, summarized: false,
    location: {display: 'test', sources: []}
  };
  domains.helper = {request: async (method, params) => {
    if (method === 'listDomains')
      return {total: 1, items: [{id: '42', name: 'D', nodes: 2,
        connections: 1, explicitAssociations: 1, components: 1,
        location: {display: 'test', sources: []}}]};
    if (method === 'listDomainComponents')
      return {total: 1, items: [{index: 0, representativeValueId: '1',
        name: 'portA', module: 'M', nodes: 2, connections: 1,
        explicitAssociations: 1, illegalCrossings: 0}]};
    if (method === 'domainItems') {
      assert.equal(params.domainTypeId, '42');
      if (params.category === 'connections')
        return {total: 1, items: [edge]};
      if (params.category === 'nodes')
        return {total: 2, items: [{id: '1', name: 'portA', module: 'M',
          kind: 'hardware', type: '!firrtl.uint<1>',
          location: {display: 'test', sources: []}}]};
      return {total: 1, items: [{name: 'portA', module: 'M',
        domainValue: 'dom', kind: 'hardware', portIndex: 0,
        location: {display: 'test', sources: []}}]};
    }
    if (method === 'minCut') {
      assert.equal(params.componentIndex, 0);
      return {available: true, mode: 'component', componentIndex: 0, count: 1,
        sourceSideSize: 1, targetSideSize: 1, edges: [edge]};
    }
    throw new Error(`unexpected method: ${method}`);
  }};
  const [domain] = await domains.getChildren();
  assert.equal(domain.kind, 'domainDetail');
  assert.equal(domains.getTreeItem(domain).id, 'domain:42');
  const [component] = await domains.getChildren(domain);
  assert.equal(component.kind, 'domainComponent');
  const groups = await domains.getChildren(component);
  assert.deepEqual(groups.map(group => group.kind),
    ['domainGroup', 'domainGroup', 'domainGroup', 'minCutGroup',
      'crossingGroup']);
  assert.equal(domains.getTreeItem(groups[3]).id,
    'domain:42:component:0:minimumCut');
  const associations = await domains.getChildren(groups[0]);
  assert.equal(associations[0].kind, 'domainAssociation');
  const connections = await domains.getChildren(groups[1]);
  assert.equal(connections[0].data.kind, 'association');
  const [action, result] = await domains.getChildren(groups[3]);
  assert.equal(result.data.count, 1);
  assert.equal(action.kind, 'cutAction');
  const cutEdges = await domains.getChildren(result);
  assert.equal(cutEdges[0].kind, 'cutEdge');
  domains.setBetweenCut('42', 0,
    {available: true, mode: 'between', count: 0,
    sourceSideSize: 1, targetSideSize: 1, edges: []}, 'M.portA', 'M.dom');
  const afterChoice = await domains.getChildren(groups[3]);
  assert.equal(afterChoice[1].data.count, 0);
  domains.summary = {complete: false};
  assert.match((await domains.getChildren())[0].label, /Partial report/);
}

async function testLoadCommandAndHover() {
  const directory = mkdtempSync(join(tmpdir(), 'circt-domain-load-'));
  const helper = join(directory, 'helper');
  const report = join(directory, 'report.json');
  const source = join(directory, 'design.scala');
  const settings = vscode.workspace.getConfiguration('circtDomains');
  const previous = settings.get('helperPath');
  try {
    writeFileSync(report, '{}');
    writeFileSync(source, 'marker\n');
    writeFileSync(helper, `#!/usr/bin/env node
      process.stdout.write(JSON.stringify({event:'ready', summary:{
        complete:true, domains:1, modules:1, values:1, instances:0,
        provenanceEdges:0, skippedEdges:0}})+'\\n');
      let buffer='';
      process.stdin.on('data', chunk => {
        buffer += chunk;
        while (buffer.includes('\\n')) {
          const end=buffer.indexOf('\\n');
          const request=JSON.parse(buffer.slice(0,end));
          buffer=buffer.slice(end+1);
          const result=request.method==='sourceMatches' ? {total:1,items:[{
            id:'1', name:'clock', kind:'hardware', module:'Top', moduleId:'0',
            type:'!firrtl.clock', location:{display:'test',sources:[]},
            assignments:[{domainTypeId:'0',domain:'D',inferred:true,
                          domainValue:'root'}]}]} : {ok:true};
          process.stdout.write(JSON.stringify({id:request.id,result})+'\\n');
        }
      });
    `);
    chmodSync(helper, 0o755);
    await settings.update('helperPath', helper, vscode.ConfigurationTarget.Global);
    await vscode.commands.executeCommand('circtDomains.loadReport', vscode.Uri.file(report));
    const document = await vscode.workspace.openTextDocument(vscode.Uri.file(source));
    const hovers = await vscode.commands.executeCommand('vscode.executeHoverProvider',
      document.uri, new vscode.Position(0, 0));
    assert.ok(hovers.some(hover => hover.contents.some(content =>
      String(content.value ?? content).includes('clock'))));
  } finally {
    await settings.update('helperPath', previous, vscode.ConfigurationTarget.Global);
    rmSync(directory, {recursive: true, force: true});
  }
}

exports.run = async () => {
  await testPaging();
  await testDomainsView();
  await testSourceNavigation();
  await testLoadCommandAndHover();
};
