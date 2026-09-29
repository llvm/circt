const {test} = require('node:test');
const assert = require('node:assert/strict');
const Module = require('node:module');

class TreeItem {
  constructor(label, collapsibleState) {
    this.label = label;
    this.collapsibleState = collapsibleState;
  }
}
class EventEmitter {
  event = () => {};
  fire() {}
}
class ThemeColor {
  constructor(id) { this.id = id; }
}
class ThemeIcon {
  constructor(id, color) {
    this.id = id;
    this.color = color;
  }
}
class FileDecoration {
  constructor(badge, tooltip, color) {
    this.badge = badge;
    this.tooltip = tooltip;
    this.color = color;
  }
}
const quickPicks = [];
class QuickPick {
  items = [];
  selectedItems = [];
  value = '';
  onDidChangeValue(callback) { this.change = callback; }
  onDidAccept(callback) { this.accepted = callback; }
  onDidHide(callback) { this.hidden = callback; }
  show() {}
  hide() { this.hidden?.(); }
  dispose() {}
  setValue(value) {
    this.value = value;
    this.change(value);
  }
  accept(index) {
    this.selectedItems = [this.items[index]];
    this.accepted();
  }
}
const vscode = {
  TreeItem,
  EventEmitter,
  ThemeColor,
  ThemeIcon,
  FileDecoration,
  Uri: {from: value => value},
  TreeItemCollapsibleState: {None: 0, Collapsed: 1},
  ProgressLocation: {Window: 1},
  window: {
    withProgress: async (_options, task) => task(),
    createQuickPick: () => {
      const picker = new QuickPick();
      quickPicks.push(picker);
      return picker;
    }
  }
};
const originalLoad = Module._load;
Module._load = function(request, parent, isMain) {
  return request === 'vscode' ? vscode : originalLoad(request, parent, isMain);
};
const {DomainsView, CrossingComponentDecorations, pickDomainNode} =
  require('../out/extension.js');
Module._load = originalLoad;

test('Domains view lists components and calculates their cuts lazily', async () => {
  const requests = [];
  const cutRequests = [];
  let suggestionRequests = 0;
  const view = new DomainsView();
  view.summary = {complete: true, hasIllegalCrossings: true};
  view.helper = {request: async (method, params) => {
    requests.push(method);
    if (method === 'listDomains')
      return {total: 1, items: [{id: '42', name: 'D', nodes: 5,
        connections: 3, explicitAssociations: 2, components: 2,
        illegalCrossings: 2,
        location: {display: 'test', sources: []}}]};
    if (method === 'listIllegalCrossings') {
      assert.ok(params.componentIndex === 0 || params.componentIndex === 1);
      assert.ok(params.limit === 1 || params.limit === 200);
      if (params.limit === 1)
        ++suggestionRequests;
      const first = {index: 0, domainTypeId: '42',
        ownerModule: 'M', operation: 'firrtl.matchingconnect',
        lhsSource: 'A', lhsSourceId: '1', lhsSourceModule: 'M',
        lhsValue: 'a', lhsValueId: '3', lhsValueModule: 'M',
        rhsValue: 'b', rhsValueId: '4', rhsValueModule: 'M',
        rhsSource: 'B', rhsSourceId: '2', rhsSourceModule: 'M',
        location: {display: 'crossing', sources: []}};
      const second = {...first, index: 1};
      return {total: 2, items: params.limit === 1 ? [first] :
        [first, second]};
    }
    if (method === 'crossingPath') {
      assert.equal(params.crossingIndex, 0);
      return {found: true, complete: false, steps: [
        {kind: 'association', lhs: 'A', rhs: 'a', lhsId: '1', rhsId: '3',
          lhsModule: 'M', rhsModule: 'M', fromId: '1', toId: '3',
          location: {display: 'association', sources: []}},
        {kind: 'failed_constraint', lhs: 'a', rhs: 'b', lhsId: '3',
          rhsId: '4', lhsModule: 'M', rhsModule: 'M', fromId: '3', toId: '4',
          location: {display: 'crossing', sources: []}},
        {kind: 'association', lhs: 'b', rhs: 'B', lhsId: '4', rhsId: '2',
          lhsModule: 'M', rhsModule: 'M', fromId: '4', toId: '2',
          location: {display: 'association', sources: []}}
      ]};
    }
    if (method === 'instanceAssociation') {
      assert.equal(params.edgeIndex, 7);
      return {instance: 'child', targets: [{
        moduleId: '9', module: 'Child', port: 'clk', portIndex: 0,
        domainPort: 'clockDomain', domainPortIndex: 1, inferred: true,
        location: {display: 'target hardware port', sources: []},
        domainPortLocation: {display: 'target domain port', sources: []}
      }]};
    }
    if (method === 'neighbors') {
      assert.equal(params.domainTypeId, '42');
      assert.deepEqual(params.kinds, ['constraint', 'domain_alias']);
      if (params.valueId === '11')
        return {total: 0, items: []};
      assert.equal(params.valueId, '10');
      return {total: 1, items: [{
        index: 8, kind: 'constraint', operation: 'firrtl.matchingconnect',
        lhsId: '10', rhsId: '12', lhsModule: 'M', rhsModule: 'Peer',
        otherValue: 'clockOut', inferred: false, summarized: false,
        location: {display: 'parent connection', sources: []}
      }]};
    }
    if (method === 'componentForValue')
      return {index: params.valueId === '1' ? 0 : 1};
    if (method === 'listDomainComponents')
      return {total: 3, items: [
        {index: 0, representativeValueId: '1', name: 'portA', module: 'M',
          nodes: 3, connections: 2, explicitAssociations: 1,
          illegalCrossings: 2},
        {index: 1, representativeValueId: '4', name: 'portB', module: 'N',
          nodes: 2, connections: 1, explicitAssociations: 0,
          illegalCrossings: 2},
        {index: -1, unmapped: true, nodes: 0, connections: 0,
          explicitAssociations: 1, illegalCrossings: 0}
      ]};
    if (method === 'domainItems') {
      if (params.unmapped) {
        assert.equal(params.category, 'associations');
        return {total: 1, items: [{name: 'extPort', module: 'Ext',
          kind: 'external port', domainValue: 'extDom',
          location: {display: 'test', sources: []}}]};
      }
      assert.equal(params.componentIndex, 0);
      if (params.category === 'connections')
        return {total: 2, items: [3, 4].map(index => ({
          index, kind: 'constraint', lhs: 'portA', rhs: 'clock',
          lhsModule: 'M', rhsModule: 'M', inferred: false,
          summarized: false, location: {display: 'test', sources: []}
        }))};
      assert.equal(params.category, 'associations');
      return {total: 1, items: [{name: 'portA', module: 'M', kind: 'hardware',
        domainValue: 'clock', location: {display: 'test', sources: []}}]};
    }
    if (method === 'minCut') {
      assert.ok(params.componentIndex === 0 || params.componentIndex === 1);
      cutRequests.push(params);
      return {available: true, mode: 'component',
        componentIndex: params.componentIndex,
        associationOnly: params.associationOnly,
        count: 1, sourceSideSize: 1, targetSideSize: 2, edges: [{
          index: 3, kind: 'association', lhs: 'portA', rhs: 'clock',
          lhsModule: 'M', rhsModule: 'M', inferred: false, summarized: false,
          location: {display: 'test', sources: []}
        }]};
    }
    throw new Error(`unexpected request: ${method}`);
  }};

  const [domain] = await view.getChildren();
  assert.equal(view.getTreeItem(domain).id, 'domain:42');
  const [first, second, unmapped] = await view.getChildren(domain);
  assert.equal(first.kind, 'domainComponent');
  assert.equal(second.kind, 'domainComponent');
  assert.equal(unmapped.kind, 'domainComponent');
  assert.equal(view.getTreeItem(first).id, 'domain:42:component:0');
  assert.equal(view.getTreeItem(unmapped).label, 'Unmapped records');
  assert.deepEqual(requests, ['listDomains', 'listDomainComponents']);

  const errorItem = view.getTreeItem(first);
  assert.equal(errorItem.label, 'Component 1');
  assert.equal(errorItem.resourceUri.scheme, 'circt-domain-component');
  assert.equal(errorItem.iconPath.color.id, 'list.errorForeground');
  const decoration = new CrossingComponentDecorations()
    .provideFileDecoration(errorItem.resourceUri);
  assert.equal(decoration.badge, '!');
  assert.equal(decoration.color.id, 'list.errorForeground');
  assert.equal(view.getTreeItem(second).resourceUri.scheme,
    'circt-domain-component');
  assert.equal(view.getTreeItem(unmapped).resourceUri, undefined);

  const groups = await view.getChildren(first);
  assert.deepEqual(groups.map(node => node.kind),
    ['domainGroup', 'domainGroup', 'domainGroup', 'minCutGroup',
      'crossingGroup']);
  assert.equal(groups[0].count, 1);
  assert.equal(groups[1].count, 2);
  assert.equal(groups[2].count, 3);
  assert.equal(groups[4].count, 2);
  assert.equal(view.getTreeItem(groups[0]).id,
    'domain:42:component:0:group:Explicit associations');
  const [association] = await view.getChildren(groups[0]);
  assert.equal(association.kind, 'domainAssociation');
  assert.equal(association.data.name, 'portA');
  const connections = await view.getChildren(groups[1]);
  assert.equal(view.getTreeItem(connections[0]).label,
    view.getTreeItem(connections[1]).label);
  assert.notEqual(view.getTreeItem(connections[0]).id,
    view.getTreeItem(connections[1]).id);

  const [crossing] = await view.getChildren(groups[4]);
  assert.equal(view.getTreeItem(crossing).label, 'M.A ↔ M.B');
  assert.equal(view.getTreeItem(crossing).id,
    'domain:42:component:0:crossing:0');
  const steps = await view.getChildren(crossing);
  assert.equal(steps.length, 4);
  assert.equal(steps[0].label, 'Shortest path (3 edges)');
  assert.equal(view.getTreeItem(steps[2]).label,
    'Failed connection: M.a → M.b');
  await view.getChildren(crossing);
  assert.equal(requests.filter(method => method === 'crossingPath').length, 1);

  const [filterRow, sourceRow, targetRow, action, cut] =
    await view.getChildren(groups[3]);
  assert.equal(view.getTreeItem(filterRow).label, 'Cut edges: All edge kinds');
  assert.equal(view.getTreeItem(filterRow).command.command,
    'circtDomains.selectCutFilter');
  assert.equal(view.getTreeItem(sourceRow).label, 'Source: M.A');
  assert.equal(view.getTreeItem(targetRow).label, 'Target: M.a');
  assert.equal(view.getTreeItem(sourceRow).description, 'ID 1');
  assert.equal(view.getTreeItem(sourceRow).command.command,
    'circtDomains.selectCutEndpoint');
  assert.equal(action.kind, 'cutAction');
  assert.equal(action.componentIndex, 0);
  assert.equal(view.getTreeItem(action).label,
    'Calculate cut between selected nodes');
  assert.equal(view.getTreeItem(action).command.command,
    'circtDomains.calculateCut');
  assert.equal(cut.data.count, 1);
  assert.equal((await view.getChildren(cut))[0].kind, 'cutEdge');
  await view.getChildren(groups[3]);
  assert.equal(requests.filter(method => method === 'minCut').length, 1);
  assert.equal(cutRequests[0].associationOnly, false);
  assert.equal(suggestionRequests, 1);

  view.setAssociationOnlyCut('42', 0, true);
  const associationRows = await view.getChildren(groups[3]);
  assert.equal(view.getTreeItem(associationRows[0]).label,
    'Cut edges: Associations only');
  assert.equal(view.getTreeItem(associationRows[4]).label,
    'Minimum: 1 association');
  assert.equal(cutRequests.at(-1).associationOnly, true);
  assert.equal(view.getTreeItem(associationRows[1]).label, 'Source: M.A');
  view.setAssociationOnlyCut('42', 0, false);
  await view.getChildren(groups[3]);
  assert.equal(cutRequests.at(-1).associationOnly, false);

  view.setBetweenCut('42', 0,
    {available: true, mode: 'between', count: 0, edges: []},
    'M.portA', 'M.portB');
  const withPair = await view.getChildren(groups[3]);
  assert.equal(withPair[4].data.count, 0);
  view.setCutEndpoint('42', 0, 'source',
    {id: '1', name: 'A', module: 'M'});
  assert.equal((await view.getChildren(groups[3]))[4].data.count, 0);

  view.setCutEndpoint('42', 0, 'source',
    {id: '5', name: 'other', module: 'M'});
  const changedPair = await view.getChildren(groups[3]);
  assert.equal(view.getTreeItem(changedPair[1]).label, 'Source: M.other');
  assert.equal(view.getTreeItem(changedPair[2]).label, 'Target: M.a');
  assert.equal(changedPair.length, 5);
  view.setCutEndpoint('42', 0, 'source',
    {id: '3', name: 'a', module: 'M'});
  const missingTarget = await view.getChildren(groups[3]);
  assert.equal(view.getTreeItem(missingTarget[2]).label,
    'Target: Choose node…');
  assert.equal(missingTarget.some(row => row.kind === 'cutAction'), false);

  const otherGroups = await view.getChildren(second);
  const [, rightSource, rightTarget] = await view.getChildren(otherGroups[3]);
  assert.equal(suggestionRequests, 2);
  assert.equal(view.getTreeItem(rightSource).label, 'Source: M.b');
  assert.equal(view.getTreeItem(rightTarget).label, 'Target: M.B');
  const [sameCrossing] = await view.getChildren(otherGroups[4]);
  assert.equal(view.getTreeItem(sameCrossing).id,
    'domain:42:component:1:crossing:0');
  await view.getChildren(sameCrossing);
  assert.equal(requests.filter(method => method === 'crossingPath').length, 1);

  const instanceStep = {kind: 'crossingStep', data: {
    index: 7, kind: 'association', domainTypeId: '42',
    instance: 'child', instanceId: '5',
    lhs: 'child.clk', rhs: 'child.clockDomain', lhsId: '10', rhsId: '11',
    lhsModule: 'M', rhsModule: 'M', fromId: '10', toId: '11',
    inferred: true, summarized: true,
    location: {display: 'instance site', sources: []}
  }};
  const instanceItem = view.getTreeItem(instanceStep);
  assert.equal(instanceItem.label,
    'Instance association: M.child.clk → M.child.clockDomain');
  assert.equal(instanceItem.collapsibleState, 1);
  const [targetAssociation, hardwareConnections, domainConnections] =
    await view.getChildren(instanceStep);
  assert.equal(targetAssociation.kind, 'instanceAssociation');
  const targetItem = view.getTreeItem(targetAssociation);
  assert.equal(targetItem.label, 'Child.clk → clockDomain');
  assert.match(targetItem.description, /inferred in target module/);
  assert.match(targetItem.tooltip, /target domain port/);
  assert.equal(targetItem.command.command, 'circtDomains.openSource');
  assert.equal(view.getTreeItem(hardwareConnections).label,
    'Hardware port connections (1)');
  assert.equal(view.getTreeItem(domainConnections).label,
    'Domain port connections (0)');
  const [connection] = await view.getChildren(hardwareConnections);
  const connectionItem = view.getTreeItem(connection);
  assert.equal(connectionItem.label, 'Peer.clockOut');
  assert.equal(connectionItem.description, 'firrtl.matchingconnect');
  assert.match(connectionItem.tooltip, /parent connection/);
  assert.equal(connectionItem.command.command, 'circtDomains.openSource');

  const unmappedGroups = await view.getChildren(unmapped);
  assert.deepEqual(unmappedGroups.map(node => node.kind),
    ['domainGroup', 'crossingGroup']);
  const [external] = await view.getChildren(unmappedGroups[0]);
  assert.equal(external.data.name, 'extPort');

  const originalHelper = view.helper;
  view.clearCuts();
  view.helper = {request: (method, params) =>
    method === 'crossingPath' ? {found: false, complete: false} :
      originalHelper.request(method, params)};
  const manualRows = await view.getChildren(groups[3]);
  assert.equal(suggestionRequests, 3);
  assert.equal(view.getTreeItem(manualRows[1]).label,
    'Source: Choose node…');
  assert.equal(view.getTreeItem(manualRows[2]).label,
    'Target: Choose node…');
  assert.equal(manualRows.some(row => row.kind === 'cutAction'), false);
});

test('Node picker searches as typed and pages matching nodes', async () => {
  const requests = [];
  const values = [
    {id: '1', name: 'clockIn', module: 'M', kind: 'hardware'},
    {id: '2', name: 'clockMid', module: 'M', kind: 'hardware'},
    {id: '3', name: 'clockOut', module: 'M', kind: 'hardware'}
  ];
  const helper = {request: async (method, params) => {
    assert.equal(method, 'domainItems');
    assert.equal(params.componentIndex, 0);
    requests.push(params);
    if (params.query === 'out')
      return {total: 1, items: [values[2]]};
    return {total: 3, items: params.offset === 0 ? values.slice(0, 2) :
      values.slice(2)};
  }};
  const selected = pickDomainNode(helper, '42', 'target', 0, '1');
  const picker = quickPicks.at(-1);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(picker.items[0].label, 'clockMid');
  assert.equal(picker.items[1].label, 'More results…');
  picker.accept(1);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(picker.items[1].label, 'clockOut');

  picker.setValue('out');
  await new Promise(resolve => setTimeout(resolve, 180));
  assert.equal(requests.at(-1).query, 'out');
  assert.deepEqual(picker.items.map(item => item.label), ['clockOut']);
  picker.accept(0);
  assert.deepEqual(await selected,
    {id: '3', name: 'clockOut', module: 'M'});
});
