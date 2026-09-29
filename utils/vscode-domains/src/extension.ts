import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';

import {Helper, Summary} from './helper';

interface Source {
  file: string;
  line: number;
  column: number;
  endLine?: number;
  endColumn?: number;
  paths: string[];
}
interface Location {
  display: string;
  sources: Source[];
}
interface Assignment {
  domainTypeId: string;
  domain: string;
  domainValueId?: string;
  domainValue?: string;
  inferred: boolean;
}
interface ValueInfo {
  id: string;
  name: string;
  kind: string;
  module: string;
  moduleId: string;
  type: string;
  location: Location;
  assignments?: Assignment[];
  definition?: string;
  direction?: string;
  instanceId?: string;
  portIndex?: number;
  instancePortIndex?: number;
  domainTypeId?: string;
}
interface ModuleInfo {
  id: string;
  name: string;
  kind: string;
  ports: number;
  values: number;
  instances: number;
}
interface DomainInfo {
  id: string;
  name: string;
  location: Location;
  nodes: number;
  connections: number;
  explicitAssociations: number;
  components: number;
  illegalCrossings?: number;
}
interface CrossingInfo {
  index: number;
  domainTypeId: string;
  ownerModule: string;
  operation: string;
  location: Location;
  lhsSource?: string;
  lhsSourceId?: string;
  lhsSourceModule?: string;
  rhsSource?: string;
  rhsSourceId?: string;
  rhsSourceModule?: string;
  lhsValue?: string;
  rhsValue?: string;
}
interface CrossingPath {
  found: boolean;
  complete: boolean;
  message?: string;
  usesSummarizedEdges?: boolean;
  steps?: EdgeInfo[];
}
interface DomainComponentInfo {
  index: number;
  representativeValueId?: string;
  name?: string;
  module?: string;
  nodes: number;
  connections: number;
  explicitAssociations: number;
  illegalCrossings: number;
  unmapped?: boolean;
}
interface CutResult {
  available: boolean;
  mode?: 'global'|'component'|'between';
  message?: string;
  count?: number;
  nodeCount?: number;
  sourceSideSize?: number;
  targetSideSize?: number;
  sourceId?: string;
  targetId?: string;
  componentIndex?: number;
  edges?: EdgeInfo[];
}
interface InstanceInfo {
  id: string;
  name: string;
  targets: string[];
  location: Location;
  bindings?: any[];
}
interface EdgeInfo {
  index?: number;
  kind: string;
  operation?: string;
  lhs?: string;
  rhs?: string;
  lhsId?: string;
  rhsId?: string;
  lhsModule?: string;
  rhsModule?: string;
  fromId?: string;
  toId?: string;
  otherValueId?: string;
  otherValue?: string;
  instanceId?: string;
  instance?: string;
  inferred: boolean;
  summarized: boolean;
  location: Location;
}
interface Page<T> {
  items: T[];
  total: number;
}
interface TraceResult {
  found: boolean;
  complete: boolean;
  contextRequired: boolean;
  usesSummarizedEdges: boolean;
  message?: string;
  endpointValueId?: string;
  boundaryValueId?: string;
  steps: EdgeInfo[];
}

type Kind =
    'message'|'domains'|'modules'|'module'|'group'|'page'|'domain'|'port'|
    'value'|'instance'|'assignment'|'binding'|'traceStep'|'neighbors'|'edge'|
    'domainDetail'|'domainGroup'|'domainNode'|'domainAssociation'|
    'domainConnection'|'minCutGroup'|'domainComponent'|'cutResult'|'cutEdge'|'cutPage'|
    'cutAction'|'crossingGroup'|'illegalCrossing'|'crossingStep'|'crossingStepPage';
interface Node {
  kind: Kind;
  label?: string;
  data?: any;
  category?: string;
  moduleId?: string;
  count?: number;
  method?: string;
  params?: object;
  itemKind?: Kind;
  offset?: number;
  total?: number;
  valueId?: string;
  domainTypeId?: string;
  domainName?: string;
  componentIndex?: number;
  unmapped?: boolean;
  treeId?: string;
  crossingIndex?: number;
}

const componentScheme = 'circt-domain-component';

export class CrossingComponentDecorations implements vscode.FileDecorationProvider {
  provideFileDecoration(uri: vscode.Uri): vscode.FileDecoration|undefined {
    if (uri.scheme !== componentScheme)
      return undefined;
    return new vscode.FileDecoration(
        '!', 'Component involved in an illegal domain crossing',
        new vscode.ThemeColor('list.errorForeground'));
  }
}

const pageSize = 200;
function escapeMarkdown(text: string): string {
  return text.replace(/[\\`*_{}\[\]()#+\-.!|>]/g, '\\$&');
}
function locationText(location?: Location): string {
  if (!location)
    return '';
  return location.sources?.length
             ? location.sources
                   .map((source) =>
                            `${source.file}:${source.line}:${source.column}`)
                   .join(', ')
             : location.display;
}
function itemNode(kind: Kind, data: any): Node { return {kind, data}; }

function treeItem(node: Node): vscode.TreeItem {
  const data = node.data;
  let label = node.label ?? '';
  let state: vscode.TreeItemCollapsibleState =
      vscode.TreeItemCollapsibleState.None;
  if (node.kind === 'domains') {
    label = 'Domain definitions';
    state = vscode.TreeItemCollapsibleState.Collapsed;
  }
  if (node.kind === 'modules') {
    label = 'Modules';
    state = vscode.TreeItemCollapsibleState.Collapsed;
  }
  if (node.kind === 'module') {
    label = `${data.name} (${data.kind})`;
    state = vscode.TreeItemCollapsibleState.Collapsed;
  }
  if (node.kind === 'group') {
    label = `${node.category} (${node.count})`;
    state = vscode.TreeItemCollapsibleState.Collapsed;
  }
  if (node.kind === 'page') {
    label = `More (${(node.offset ?? 0) + 1}–${
        Math.min((node.offset ?? 0) + pageSize,
                 node.total ?? 0)} of ${node.total})`;
    state = vscode.TreeItemCollapsibleState.Collapsed;
  }
  if (node.kind === 'domain')
    label = data.name;
  if (node.kind === 'domainDetail') {
    label = data.name;
    state = vscode.TreeItemCollapsibleState.Collapsed;
  }
  if (node.kind === 'domainGroup') {
    label = `${node.category} (${node.count})`;
    state = vscode.TreeItemCollapsibleState.Collapsed;
  }
  if (node.kind === 'crossingGroup') {
    label = `Illegal crossings (${node.count})`;
    state = vscode.TreeItemCollapsibleState.Collapsed;
  }
  if (node.kind === 'illegalCrossing') {
    const crossing = data as CrossingInfo;
    label = `${crossing.lhsSourceModule ?? '?'}.${crossing.lhsSource ?? '?'} ↔ ${
        crossing.rhsSourceModule ?? '?'}.${crossing.rhsSource ?? '?'}`;
    state = vscode.TreeItemCollapsibleState.Collapsed;
  }
  if (node.kind === 'crossingStep') {
    const forward = data.fromId === data.lhsId;
    const from = forward ? `${data.lhsModule}.${data.lhs}`
                         : `${data.rhsModule}.${data.rhs}`;
    const to = forward ? `${data.rhsModule}.${data.rhs}`
                       : `${data.lhsModule}.${data.lhs}`;
    label = `${data.kind === 'failed_constraint' ? 'Failed connection'
                : data.kind}: ${from} → ${to}`;
  }
  if (node.kind === 'crossingStepPage') {
    label = `More (${(node.offset ?? 0) + 1}–${
        Math.min((node.offset ?? 0) + pageSize, node.total ?? 0)} of ${
        node.total})`;
    state = vscode.TreeItemCollapsibleState.Collapsed;
  }
  if (node.kind === 'minCutGroup') {
    label = 'Minimum cut';
    state = vscode.TreeItemCollapsibleState.Collapsed;
  }
  if (node.kind === 'domainComponent') {
    label = data.unmapped ? 'Unmapped records' : `Component ${data.index + 1}`;
    state = vscode.TreeItemCollapsibleState.Collapsed;
  }
  if (node.kind === 'domainNode')
    label = `${data.module}.${data.name}`;
  if (node.kind === 'domainAssociation')
    label = `${data.module}.${data.name} → ${
        data.domainValue ?? data.domainValueId ?? '(unresolved)'}`;
  if (node.kind === 'domainConnection' || node.kind === 'cutEdge') {
    label = `${data.lhsModule}.${data.lhs} ↔ ${data.rhsModule}.${data.rhs}`;
  }
  if (node.kind === 'cutResult') {
    const cut = data as CutResult;
    label = !cut.available ? cut.message ?? 'No cut available'
            : cut.mode === 'between'
                ? `${node.label}: ${cut.count} edge${cut.count === 1 ? '' : 's'}`
            : cut.mode === 'component'
                ? `Minimum: ${cut.count} edge${cut.count === 1 ? '' : 's'}`
                : `Global minimum: ${cut.count} edge${
                      cut.count === 1 ? '' : 's'}`;
    state = cut.edges?.length ? vscode.TreeItemCollapsibleState.Collapsed
                              : vscode.TreeItemCollapsibleState.None;
  }
  if (node.kind === 'cutAction')
    label = 'Choose two nodes…';
  if (node.kind === 'cutPage') {
    label = `More (${(node.offset ?? 0) + 1}–${
        Math.min((node.offset ?? 0) + pageSize, node.total ?? 0)} of ${
        node.total})`;
    state = vscode.TreeItemCollapsibleState.Collapsed;
  }
  if (node.kind === 'port') {
    label = `${data.direction || '?'} ${data.name}`;
    state = data.assignments?.length ? vscode.TreeItemCollapsibleState.Collapsed
                                     : vscode.TreeItemCollapsibleState.None;
  }
  if (node.kind === 'value') {
    label = data.name;
    state = vscode.TreeItemCollapsibleState.Collapsed;
  }
  if (node.kind === 'instance') {
    label = data.name;
    state = vscode.TreeItemCollapsibleState.Collapsed;
  }
  if (node.kind === 'assignment') {
    label = `${data.domain} → ${
        data.domainValue ?? data.domainValueId ?? '(unresolved)'}`;
  }
  if (node.kind === 'binding') {
    label = `${data.targetModule}[${data.portIndex}] → ${
        data.effectiveDomainValueName || data.effectiveDomainValueId ||
        '(unresolved)'}`;
  }
  if (node.kind === 'traceStep') {
    label = `${data.kind}: ${data.lhs ?? '?'} ↔ ${data.rhs ?? '?'}`;
    state = vscode.TreeItemCollapsibleState.Collapsed;
  }
  if (node.kind === 'neighbors') {
    label = 'Other related edges';
    state = vscode.TreeItemCollapsibleState.Collapsed;
  }
  if (node.kind === 'edge') {
    label = `${data.kind}: ${data.otherValue ?? '?'}${
        data.instance ? ` (${data.instance})` : ''}`;
  }
  const item = new vscode.TreeItem(label, state);
  if (node.treeId)
    item.id = node.treeId;
  if (node.kind === 'domainDetail')
    item.id = `domain:${data.id}`;
  if (node.kind === 'domainGroup')
    item.id = `domain:${node.domainTypeId}:component:${
        node.unmapped ? 'unmapped' : node.componentIndex}:group:${
        node.category}`;
  if (node.kind === 'crossingGroup')
    item.id = `domain:${node.domainTypeId}:component:${
        node.unmapped ? 'unmapped' : node.componentIndex}:illegalCrossings`;
  if (node.kind === 'illegalCrossing')
    item.id = `domain:${node.domainTypeId}:component:${
        node.unmapped ? 'unmapped' : node.componentIndex}:crossing:${data.index}`;
  if (node.kind === 'minCutGroup')
    item.id = `domain:${node.domainTypeId}:component:${
        node.componentIndex}:minimumCut`;
  if (node.kind === 'domainComponent')
    item.id = `domain:${node.domainTypeId}:component:${
        data.unmapped ? 'unmapped' : data.index}`;
  if (node.kind === 'cutResult')
    item.id = `domain:${node.domainTypeId}:component:${
        node.componentIndex}:cut:${
        data.mode ?? 'unavailable'}:${data.componentIndex ?? ''}`;
  if (node.kind === 'module')
    item.description = `${data.ports} ports · ${data.values} values`;
  if (node.kind === 'value' || node.kind === 'port')
    item.description = data.type;
  if (node.kind === 'domainDetail')
    item.description = `${data.nodes} nodes · ${data.components} components`;
  if (node.kind === 'domainComponent') {
    item.description = data.unmapped
                           ? `${data.explicitAssociations} associations · ${
                                 data.illegalCrossings} crossings`
                           : `${data.nodes} nodes · ${data.connections} edges · ${
                                 data.module}.${data.name}`;
    if (data.illegalCrossings > 0) {
      item.resourceUri = vscode.Uri.from({
        scheme : componentScheme,
        path : `/${node.domainTypeId}/${data.unmapped ? 'unmapped'
                                                 : data.index}`
      });
      item.iconPath = new vscode.ThemeIcon(
          'error', new vscode.ThemeColor('list.errorForeground'));
      item.accessibilityInformation = {
        label : `${label}, involved in an illegal domain crossing`
      };
    }
  }
  if (node.kind === 'domainNode')
    item.description = `${data.kind} · ${data.type}`;
  if (node.kind === 'domainAssociation')
    item.description = `${data.kind}${
        data.portIndex !== undefined ? ` · port ${data.portIndex}` : ''}`;
  if (node.kind === 'domainConnection' || node.kind === 'cutEdge')
    item.description = [
      data.kind, data.operation, data.inferred ? 'inferred' : 'explicit',
      data.summarized ? 'summary' : undefined
    ].filter(Boolean).join(' · ');
  if (node.kind === 'illegalCrossing')
    item.description = `${data.ownerModule} · ${data.operation}`;
  if (node.kind === 'crossingStep')
    item.description = [
      data.operation, data.summarized ? 'summary' : undefined
    ].filter(Boolean).join(' · ');
  if (node.kind === 'cutResult' && data.available)
    item.description = data.count === 0 ? 'already separate'
                                          : `${data.sourceSideSize} / ${
                                                data.targetSideSize} nodes`;
  if (node.kind === 'instance')
    item.description = data.targets?.join(', ');
  if (node.kind === 'assignment')
    item.description = data.inferred ? 'inferred' : 'explicit';
  if (node.kind === 'binding')
    item.description = data.domain;
  if (node.kind === 'traceStep' || node.kind === 'edge') {
    item.description = [
      data.operation, data.inferred ? 'inferred' : 'explicit',
      data.summarized ? 'summary' : undefined
    ].filter(Boolean).join(' · ');
  }
  if (data?.location)
    item.tooltip = `${locationText(data.location)}\n${data.type ?? ''}`;
  if (node.kind === 'value')
    item.contextValue = 'domainValue';
  if (node.kind === 'domainDetail')
    item.contextValue = 'domainDefinition';
  if (node.kind === 'domainNode')
    item.contextValue = 'domainValue';
  if (node.kind === 'domainAssociation')
    item.contextValue = 'domainAssociation';
  if (node.kind === 'domainConnection' || node.kind === 'cutEdge')
    item.contextValue = 'domainGraphEdge';
  if (node.kind === 'port' && data.valueId)
    item.contextValue = 'domainPort';
  if (node.kind === 'instance')
    item.contextValue = 'domainInstance';
  if (node.kind === 'traceStep')
    item.contextValue = 'domainTraceStep';
  if (node.kind === 'crossingStep' || node.kind === 'illegalCrossing')
    item.contextValue = 'domainTraceStep';
  if (node.kind === 'edge')
    item.contextValue = 'domainTraceEdge';
  if (node.kind === 'assignment' && node.valueId) {
    item.command = {
      command : 'circtDomains.traceValue',
      title : 'Trace Value',
      arguments : [ node ]
    };
  } else if (node.kind === 'cutAction') {
    item.command = {
      command : 'circtDomains.chooseCutEndpoints',
      title : 'Choose Min-Cut Endpoints',
      arguments : [ node ]
    };
  } else if (data?.location && node.kind !== 'value' &&
             node.kind !== 'instance' && node.kind !== 'domainDetail') {
    item.command = {
      command : 'circtDomains.openSource',
      title : 'Open Source',
      arguments : [ node ]
    };
  }
  return item;
}

export class Explorer implements vscode.TreeDataProvider<Node> {
  private readonly changed = new vscode.EventEmitter<Node|undefined>();
  readonly onDidChangeTreeData = this.changed.event;
  helper?: Helper;
  summary?: Summary;
  report?: vscode.Uri;

  refresh(): void { this.changed.fire(undefined); }
  getTreeItem(node: Node): vscode.TreeItem { return treeItem(node); }

  async getChildren(node?: Node): Promise<Node[]> {
    if (!this.helper || !this.summary) {
      return node ? [] : [
        {kind : 'message', label : 'Run “FIRRTL Domains: Load Domain Report”'}
      ];
    }
    if (!node) {
      const status = `complete: ${this.summary.complete}`;
      return [
        {
          kind : 'message',
          label :
              `${path.basename(this.report?.fsPath ?? 'Report')} · ${status}`
        },
        {kind : 'domains'}, {kind : 'modules'}
      ];
    }
    if (node.kind === 'domains')
      return this.page('listDomains', {}, 'domain', 0);
    if (node.kind === 'modules')
      return this.page('listModules', {}, 'module', 0);
    if (node.kind === 'module') {
      const info = node.data as ModuleInfo;
      return [
        {
          kind : 'group',
          category : 'ports',
          moduleId : info.id,
          count : info.ports
        },
        {
          kind : 'group',
          category : 'values',
          moduleId : info.id,
          count : info.values
        },
        {
          kind : 'group',
          category : 'instances',
          moduleId : info.id,
          count : info.instances
        }
      ];
    }
    if (node.kind === 'group') {
      const category = node.category!;
      const kind = category === 'ports'    ? 'port'
                   : category === 'values' ? 'value'
                                           : 'instance';
      return this.page('moduleItems', {moduleId : node.moduleId, category},
                       kind, 0);
    }
    if (node.kind === 'page')
      return this.page(node.method!, node.params ?? {}, node.itemKind!,
                       node.offset ?? 0);
    if (node.kind === 'port')
      return ((node.data.assignments ?? []) as Assignment[])
          .map<Node>(
              (data) =>
                  ({kind : 'assignment', data, valueId : node.data.valueId}));
    if (node.kind === 'value') {
      const value =
          await this.helper.request<ValueInfo>('getValue', {id : node.data.id});
      return (value.assignments ?? [])
          .map<Node>((data) =>
                         ({kind : 'assignment', data, valueId : value.id}));
    }
    if (node.kind === 'instance') {
      const info = await this.helper.request<InstanceInfo>('getInstance',
                                                           {id : node.data.id});
      return (info.bindings ?? [])
          .map<Node>((data) => ({kind : 'binding', data}));
    }
    return [];
  }

  private async page(method: string, params: object, kind: Kind,
                     offset: number): Promise<Node[]> {
    const result = await this.helper!.request<Page<any>>(
        method, {...params, offset, limit : pageSize});
    const nodes = result.items.map((data) => itemNode(kind, data));
    if (offset + result.items.length < result.total)
      nodes.push({
        kind : 'page',
        method,
        params,
        itemKind : kind,
        offset : offset + result.items.length,
        total : result.total
      });
    return nodes;
  }
}

export class DomainsView implements vscode.TreeDataProvider<Node> {
  private readonly changed = new vscode.EventEmitter<Node|undefined>();
  readonly onDidChangeTreeData = this.changed.event;
  private readonly componentCuts = new Map<string, CutResult>();
  private readonly betweenCuts = new Map<
      string, {result: CutResult; source: string; target: string}>();
  private readonly crossingPaths = new Map<number, CrossingPath>();
  helper?: Helper;
  summary?: Summary;

  refresh(): void { this.changed.fire(undefined); }
  clearCuts(): void {
    this.componentCuts.clear();
    this.betweenCuts.clear();
    this.crossingPaths.clear();
    this.refresh();
  }
  setBetweenCut(domainTypeId: string, componentIndex: number,
                result: CutResult, source: string, target: string): void {
    this.betweenCuts.set(`${domainTypeId}:${componentIndex}`,
                         {result, source, target});
    this.refresh();
  }
  getTreeItem(node: Node): vscode.TreeItem { return treeItem(node); }

  async getChildren(node?: Node): Promise<Node[]> {
    if (!this.helper || !this.summary)
      return node ? [] : [
        {kind : 'message', label : 'Run “FIRRTL Domains: Load Domain Report”'}
      ];
    if (!node) {
      const nodes = await this.page('listDomains', {}, 'domainDetail', 0);
      if (!this.summary.complete)
        nodes.unshift({
          kind : 'message',
          label : 'Partial report · cuts cover recorded edges only'
        });
      return nodes;
    }
    if (node.kind === 'domainDetail') {
      const domain = node.data as DomainInfo;
      return this.page('listDomainComponents',
                       {domainTypeId : domain.id, domainName : domain.name},
                       'domainComponent', 0);
    }
    if (node.kind === 'domainComponent') {
      const component = node.data as DomainComponentInfo;
      const selection = {
        domainTypeId : node.domainTypeId,
        componentIndex : component.unmapped ? undefined : component.index,
        unmapped : component.unmapped
      };
      const groups: Node[] = [
        {
          kind : 'domainGroup',
          category : 'Explicit associations',
          count : component.explicitAssociations,
          ...selection
        }
      ];
      if (!component.unmapped) {
        groups.push(
            {
              kind : 'domainGroup',
              category : 'Connections',
              count : component.connections,
              ...selection
            },
            {
              kind : 'domainGroup',
              category : 'Nodes',
              count : component.nodes,
              ...selection
            },
            {kind : 'minCutGroup', domainTypeId : node.domainTypeId,
              domainName : node.domainName, componentIndex : component.index,
              total : component.nodes});
      }
      groups.push({kind : 'crossingGroup', count : component.illegalCrossings,
                   ...selection});
      return groups;
    }
    if (node.kind === 'crossingGroup') {
      if (!this.summary.hasIllegalCrossings)
        return [{kind : 'message',
                 label : 'Illegal crossings were not recorded in this report'}];
      if (!node.count)
        return [{kind : 'message', label : 'No illegal crossings recorded'}];
      return this.page(
          'listIllegalCrossings',
          {domainTypeId : node.domainTypeId,
            componentIndex : node.componentIndex, unmapped : node.unmapped},
          'illegalCrossing', 0);
    }
    if (node.kind === 'illegalCrossing') {
      const crossing = node.data as CrossingInfo;
      const crossingId = `domain:${node.domainTypeId}:component:${
          node.unmapped ? 'unmapped' : node.componentIndex}:crossing:${
          crossing.index}`;
      let path = this.crossingPaths.get(crossing.index);
      if (!path) {
        const requestHelper = this.helper;
        path = await requestHelper.request<CrossingPath>(
            'crossingPath', {crossingIndex : crossing.index});
        if (this.helper !== requestHelper)
          return [];
        this.crossingPaths.set(crossing.index, path);
      }
      if (!path.found)
        return [{kind : 'message', label : path.message ?? 'No path recorded'}];
      const steps = path.steps ?? [];
      const children: Node[] = [
        {kind : 'message', label : `Shortest path (${steps.length} edges)`}
      ];
      if (path.usesSummarizedEdges)
        children.push({kind : 'message',
                       label : 'Path includes summarized inference edges'});
      children.push(...steps.slice(0, pageSize).map(
          (data, index) => ({
            kind : 'crossingStep' as Kind,
            data,
            treeId : `${crossingId}:step:${index}`
          })));
      if (steps.length > pageSize)
        children.push({kind : 'crossingStepPage', data : path,
                       domainTypeId : node.domainTypeId,
                       componentIndex : node.componentIndex,
                       unmapped : node.unmapped,
                       crossingIndex : crossing.index,
                       treeId : `${crossingId}:page:${pageSize}`,
                       offset : pageSize, total : steps.length});
      return children;
    }
    if (node.kind === 'crossingStepPage') {
      const path = node.data as CrossingPath;
      const steps = path.steps ?? [];
      const offset = node.offset ?? 0;
      const crossingId = `domain:${node.domainTypeId}:component:${
          node.unmapped ? 'unmapped' : node.componentIndex}:crossing:${
          node.crossingIndex}`;
      const children: Node[] = steps.slice(offset, offset + pageSize)
                                   .map((data, index) => ({
                                          kind : 'crossingStep',
                                          data,
                                          treeId : `${crossingId}:step:${
                                              offset + index}`
                                        }));
      if (offset + pageSize < steps.length)
        children.push({kind : 'crossingStepPage', data : path,
                       domainTypeId : node.domainTypeId,
                       componentIndex : node.componentIndex,
                       unmapped : node.unmapped,
                       crossingIndex : node.crossingIndex,
                       treeId : `${crossingId}:page:${offset + pageSize}`,
                       offset : offset + pageSize, total : steps.length});
      return children;
    }
    if (node.kind === 'domainGroup') {
      const category = node.category === 'Explicit associations'
                           ? 'associations'
                       : node.category === 'Connections' ? 'connections'
                                                         : 'nodes';
      const kind = category === 'associations' ? 'domainAssociation'
                   : category === 'connections' ? 'domainConnection'
                                               : 'domainNode';
      return this.page(
          'domainItems',
          {domainTypeId : node.domainTypeId,
            componentIndex : node.componentIndex, unmapped : node.unmapped,
            category},
          kind, 0);
    }
    if (node.kind === 'page')
      return this.page(node.method!, node.params ?? {}, node.itemKind!,
                       node.offset ?? 0);
    if (node.kind === 'minCutGroup') {
      const domainTypeId = node.domainTypeId!;
      const componentIndex = node.componentIndex!;
      const key = `${domainTypeId}:${componentIndex}`;
      const children: Node[] = [];
      if ((node.total ?? 0) >= 2)
        children.push({kind : 'cutAction', domainTypeId,
                       domainName : node.domainName, componentIndex});
      const between = this.betweenCuts.get(key);
      if (between)
        children.push({
          kind : 'cutResult',
          data : between.result,
          domainTypeId,
          componentIndex,
          label : `${between.source} ↔ ${between.target}`
        });
      let cut = this.componentCuts.get(key);
      if (!cut) {
        const requestHelper = this.helper;
        cut = await vscode.window.withProgress(
            {
              location : vscode.ProgressLocation.Window,
              title : `Calculating minimum cut for component ${
                  componentIndex + 1}`
            },
            () => requestHelper.request<CutResult>(
                'minCut', {domainTypeId, componentIndex}));
        if (this.helper !== requestHelper)
          return [];
        this.componentCuts.set(key, cut);
      }
      children.push({kind : 'cutResult', data : cut, domainTypeId,
                     componentIndex});
      return children;
    }
    if (node.kind === 'cutResult' || node.kind === 'cutPage') {
      const cut = node.data as CutResult;
      const edges = cut.edges ?? [];
      const offset = node.offset ?? 0;
      const children: Node[] = edges.slice(offset, offset + pageSize)
                                   .map((data, index) => ({
                                          kind : 'cutEdge',
                                          data,
                                          treeId : `domain:${node.domainTypeId}:component:${
                                              node.componentIndex}:cut:${
                                              cut.mode}:edge:${offset + index}`
                                        }));
      if (offset + pageSize < edges.length)
        children.push({kind : 'cutPage', data : cut,
                       domainTypeId : node.domainTypeId,
                       componentIndex : node.componentIndex,
                       offset : offset + pageSize, total : edges.length});
      return children;
    }
    return [];
  }

  private async page(method: string, params: object, kind: Kind,
                     offset: number): Promise<Node[]> {
    const result = await this.helper!.request<Page<any>>(
        method, {...params, offset, limit : pageSize});
    const selection = params as {
      domainTypeId?: string;
      domainName?: string;
      componentIndex?: number;
      unmapped?: boolean;
      category?: string;
    };
    const itemPrefix = method === 'domainItems'
                           ? `domain:${selection.domainTypeId}:component:${
                                 selection.unmapped ? 'unmapped'
                                                    : selection.componentIndex}:items:${
                                 selection.category}`
                           : undefined;
    const nodes: Node[] = result.items.map((data, index) => ({
      kind,
      data,
      treeId : itemPrefix ? `${itemPrefix}:${offset + index}` : undefined,
      domainTypeId : data.id && kind === 'domainDetail' ? data.id
                                                        : selection.domainTypeId,
      domainName : selection.domainName,
      componentIndex : selection.componentIndex,
      unmapped : selection.unmapped
    }));
    if (offset + result.items.length < result.total)
      nodes.push({
        kind : 'page',
        method,
        params,
        itemKind : kind,
        offset : offset + result.items.length,
        total : result.total
      });
    return nodes;
  }
}

export class TraceView implements vscode.TreeDataProvider<Node> {
  private readonly changed = new vscode.EventEmitter<Node|undefined>();
  readonly onDidChangeTreeData = this.changed.event;
  helper?: Helper;
  trace?: TraceResult;
  valueId?: string;
  domainTypeId?: string;
  instanceId?: string;

  refresh(): void { this.changed.fire(undefined); }
  getTreeItem(node: Node): vscode.TreeItem { return treeItem(node); }
  async getChildren(node?: Node): Promise<Node[]> {
    if (!this.helper || !this.trace || !this.valueId || !this.domainTypeId)
      return node ? [] : [
        {kind : 'message', label : 'Select “Trace Value” in the report'}
      ];
    if (!node) {
      const message = this.trace.found
                          ? `Path to ${this.trace.endpointValueId} (${
                                this.trace.steps.length} steps)`
                      : this.trace.boundaryValueId
                          ? `Trace reaches instance boundary at ${
                                this.trace.boundaryValueId}`
                          : this.trace.message ?? 'No recorded path';
      const nodes: Node[] = [ {kind : 'message', label : message} ];
      if (this.instanceId)
        nodes.push({kind : 'message',
                    label : `Instance context: ${this.instanceId}`});
      if (this.trace.usesSummarizedEdges)
        nodes.push({
          kind : 'message',
          label : 'Shortest route includes summarized edges'
        });
      if (!this.trace.found && this.trace.contextRequired)
        nodes.push({
          kind : 'message',
          label : 'Choose an instance edge to continue the trace'
        });
      nodes.push(...this.trace.steps.map(
          (data) => ({kind : 'traceStep' as Kind, data})));
      nodes.push({
        kind : 'neighbors',
        valueId : this.trace.found ? this.valueId
                                   : this.trace.boundaryValueId ?? this.valueId,
        domainTypeId : this.domainTypeId
      });
      return nodes;
    }
    if (node.kind === 'traceStep')
      return [ {
        kind : 'neighbors',
        valueId : node.data.toId,
        domainTypeId : this.domainTypeId
      } ];
    if (node.kind === 'neighbors' || node.kind === 'page') {
      const valueId = node.valueId!;
      const domainTypeId = node.domainTypeId!;
      const offset = node.offset ?? 0;
      const result = await this.helper.request<Page<EdgeInfo>>(
          'neighbors', {valueId, domainTypeId, offset, limit : pageSize});
      const nodes: Node[] = result.items.map<Node>(
          (data) => ({kind : 'edge', data, valueId, domainTypeId}));
      if (offset + result.items.length < result.total)
        nodes.push({
          kind : 'page',
          valueId,
          domainTypeId,
          offset : offset + result.items.length,
          total : result.total
        });
      return nodes;
    }
    return [];
  }
}

function rootsFor(report: vscode.Uri): string[] {
  const folders =
      vscode.workspace.workspaceFolders?.map((folder) => folder.uri.fsPath) ??
      [];
  const configured = vscode.workspace.getConfiguration('circtDomains')
                         .get<string[]>('sourceRoots', []);
  const base = folders[0] ?? path.dirname(report.fsPath);
  const extra = configured.map((root) => path.resolve(base, root));
  return [...new Set([...folders, path.dirname(report.fsPath), ...extra ].map(
      path.normalize)) ];
}

function executableFor(context: vscode.ExtensionContext): string {
  const configured = vscode.workspace.getConfiguration('circtDomains')
                         .get<string>('helperPath', '');
  if (configured) {
    const base = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ??
                 context.extensionPath;
    return path.resolve(base, configured);
  }
  const platform = `${process.platform}-${process.arch}`;
  return context.asAbsolutePath(
      path.join('bin', platform, 'circt-domain-report-server'));
}

export async function openSource(location?: Location): Promise<void> {
  if (!location)
    return;
  const choices = location.sources.flatMap(
      (source) => source.paths.map(
          (filename) => ({
            label : `${source.file}:${source.line}:${source.column}`,
            description : filename,
            source,
            filename
          })));
  if (!choices.length) {
    void vscode.window.showWarningMessage(`Source unavailable: ${
        locationText(location)}. Configure circtDomains.sourceRoots.`);
    return;
  }
  const choice = choices.length === 1
                     ? choices[0]
                     : await vscode.window.showQuickPick(
                           choices, {placeHolder : 'Choose a source location'});
  if (!choice)
    return;
  const start =
      new vscode.Position(choice.source.line - 1, choice.source.column - 1);
  const end = choice.source.endLine !== undefined &&
                      choice.source.endColumn !== undefined
                  ? new vscode.Position(choice.source.endLine - 1,
                                        choice.source.endColumn - 1)
                  : start;
  const range = new vscode.Range(start, end);
  const editor = await vscode.window.showTextDocument(
      vscode.Uri.file(choice.filename), {preview : false, selection : range});
  editor.revealRange(range,
                     vscode.TextEditorRevealType.InCenterIfOutsideViewport);
}

function valueMarkdown(value: ValueInfo): string {
  const lines = [
    `# ${escapeMarkdown(value.name)}`, '',
    `**Module:** ${escapeMarkdown(value.module)}`,
    `**Kind:** ${escapeMarkdown(value.kind)}`,
    `**Type:** \`${value.type.replace(/`/g, '\\`')}\``,
    `**Source:** ${escapeMarkdown(locationText(value.location))}`
  ];
  if (value.definition)
    lines.push(`**Definition:** ${escapeMarkdown(value.definition)}`);
  if (value.direction)
    lines.push(`**Direction:** ${escapeMarkdown(value.direction)}`);
  if (value.portIndex !== undefined)
    lines.push(`**Port index:** ${value.portIndex}`);
  if (value.instanceId)
    lines.push(`**Instance:** ${escapeMarkdown(value.instanceId)}`);
  if (value.instancePortIndex !== undefined)
    lines.push(`**Instance port index:** ${value.instancePortIndex}`);
  if (value.domainTypeId)
    lines.push(`**Domain type ID:** ${escapeMarkdown(value.domainTypeId)}`);
  lines.push('', '## Domain assignments', '');
  if (!value.assignments?.length)
    lines.push('No assignment is recorded.');
  else
    for (const assignment of value.assignments) {
      lines.push(`- ${escapeMarkdown(assignment.domain)} → ${
          escapeMarkdown(assignment.domainValue ?? assignment.domainValueId ??
                         '(unresolved)')} (${
          assignment.inferred ? 'inferred' : 'explicit'})`);
    }
  return lines.join('\n');
}

function traceMarkdown(value: ValueInfo, domainTypeId: string,
                       trace: TraceResult): string {
  const lines = [
    `# Inference trace: ${escapeMarkdown(value.name)}`, '',
    `**Module:** ${escapeMarkdown(value.module)}`,
    `**Domain type ID:** ${escapeMarkdown(domainTypeId)}`, ''
  ];
  if (trace.found)
    lines.push(`Shortest recorded path to value ${
        escapeMarkdown(trace.endpointValueId ?? '?')}.`);
  else
    lines.push(trace.message ?? 'No recorded path.');
  if (!trace.found && trace.contextRequired)
    lines.push('An instance context is needed to continue this trace.');
  if (trace.usesSummarizedEdges)
    lines.push('This path includes summarized inference edges.');
  lines.push('', '## Steps', '');
  if (!trace.steps.length)
    lines.push('No steps were recorded.');
  for (const [index, step] of trace.steps.entries()) {
    const forward = step.fromId === step.lhsId;
    const from = forward ? `${step.lhsModule ?? '?'}.${step.lhs ?? '?'}`
                         : `${step.rhsModule ?? '?'}.${step.rhs ?? '?'}`;
    const to = forward ? `${step.rhsModule ?? '?'}.${step.rhs ?? '?'}`
                       : `${step.lhsModule ?? '?'}.${step.lhs ?? '?'}`;
    lines.push(`${index + 1}. **${escapeMarkdown(step.kind)}:** ${
        escapeMarkdown(from)} → ${escapeMarkdown(to)}`);
    lines.push(`   ${escapeMarkdown(step.operation ?? 'unknown operation')} · ${
        escapeMarkdown(locationText(step.location))}`);
  }
  return lines.join('\n');
}

export function activate(context: vscode.ExtensionContext): void {
  const output = vscode.window.createOutputChannel('FIRRTL Domains');
  const domainsView = new DomainsView();
  let helper: Helper|undefined;
  let loadedSummary: Summary|undefined;
  let currentReport: vscode.Uri|undefined;
  let pendingLoad: Helper|undefined;
  let loadGeneration = 0;
  context.subscriptions.push(
      output,
      vscode.window.registerTreeDataProvider('circtDomains.domains', domainsView),
      vscode.window.registerFileDecorationProvider(
          new CrossingComponentDecorations()));

  async function loadReport(selected?: vscode.Uri): Promise<void> {
    if (!selected) {
      const picked = await vscode.window.showOpenDialog({
        canSelectMany : false,
        openLabel : 'Load Domain Report',
        filters : {'JSON reports' : [ 'json' ]}
      });
      selected = picked?.[0];
    }
    if (!selected)
      return;
    if (selected.scheme !== 'file') {
      void vscode.window.showErrorMessage(
          'The native helper requires a local file report.');
      return;
    }
    const executable = executableFor(context);
    if (!fs.existsSync(executable)) {
      void vscode.window.showErrorMessage(
          `Domain report helper not found: ${executable}`);
      return;
    }
    const generation = ++loadGeneration;
    pendingLoad?.dispose();
    const report = selected;
    try {
      await vscode.window.withProgress(
          {
            location : vscode.ProgressLocation.Notification,
            title : `Loading ${path.basename(report.fsPath)}`,
            cancellable : true
          },
          async (progress, token) => {
            let lastPercentage = 0;
            const next =
                new Helper(executable, report.fsPath, (bytes, total) => {
                  const percentage =
                      total ? Math.min(99, Math.floor(bytes * 100 / total)) : 0;
                  progress.report({
                    increment : percentage - lastPercentage,
                    message : `${Math.round(bytes / 1048576)} MiB read`
                  });
                  lastPercentage = percentage;
                }, (message) => output.append(message));
            pendingLoad = next;
            const cancellation =
                token.onCancellationRequested(() => next.dispose());
            try {
              const summary = await next.waitReady();
              await next.request('setSourceRoots', {roots : rootsFor(report)});
              if (token.isCancellationRequested ||
                  generation !== loadGeneration)
                return;
              helper?.dispose();
              helper = next;
              currentReport = report;
              loadedSummary = summary;
              domainsView.helper = next;
              domainsView.summary = summary;
              domainsView.clearCuts();
              output.appendLine(`Loaded ${report.fsPath}: ${
                  summary.values} values, ${summary.provenanceEdges} edges${
                  summary.complete ? '' : ' (partial)'}`);
              if (summary.skippedEdges)
                output.appendLine(`Skipped ${
                    summary
                        .skippedEdges} edges with missing references in the partial report.`);
            } catch (error) {
              next.dispose();
              if (!token.isCancellationRequested &&
                  generation === loadGeneration)
                throw error;
            } finally {
              cancellation.dispose();
              if (pendingLoad === next)
                pendingLoad = undefined;
            }
          });
    } catch (error) {
      output.appendLine(String(error));
      void vscode.window.showErrorMessage(
          `Could not load domain report: ${String(error)}`);
    }
  }

  async function showValueDetails(id: string): Promise<void> {
    if (!helper)
      return;
    const value = await helper.request<ValueInfo>('getValue', {id});
    const document = await vscode.workspace.openTextDocument(
        {language : 'markdown', content : valueMarkdown(value)});
    await vscode.window.showTextDocument(
        document, {preview : true, viewColumn : vscode.ViewColumn.Beside});
  }

  async function showInstanceDetails(id: string): Promise<void> {
    if (!helper)
      return;
    const instance = await helper.request<InstanceInfo>('getInstance', {id});
    const lines = [
      `# ${escapeMarkdown(instance.name)}`, '',
      `**Targets:** ${escapeMarkdown(instance.targets.join(', '))}`,
      `**Source:** ${escapeMarkdown(locationText(instance.location))}`, '',
      '## Effective domain bindings', ''
    ];
    if (!instance.bindings?.length)
      lines.push('No effective bindings are recorded.');
    else
      for (const binding of instance.bindings) {
        lines.push(`- ${escapeMarkdown(binding.targetModule)}[${
            binding.portIndex}]: ${escapeMarkdown(binding.domain)} → ${
            escapeMarkdown(binding.effectiveDomainValueName ||
                           binding.effectiveDomainValueId || '(unresolved)')}` +
                   ` (domain port ${binding.domainPortIndex}, source ${
                       escapeMarkdown(locationText(binding.location))})`);
      }
    const document = await vscode.workspace.openTextDocument(
        {language : 'markdown', content : lines.join('\n')});
    await vscode.window.showTextDocument(
        document, {preview : true, viewColumn : vscode.ViewColumn.Beside});
  }

  async function traceValue(argument?: Node|string, chosenDomain?: string,
                            instanceId?: string): Promise<void> {
    if (!helper)
      return;
    let id = typeof argument === 'string'
                   ? argument
                   : argument?.valueId ?? argument?.data?.valueId ??
                         argument?.data?.id;
    if (!id) {
      const query = await vscode.window.showInputBox(
          {prompt : 'Search for a value to trace'});
      if (!query)
        return;
      id = (await chooseValue('searchValues', {query}))?.id;
      if (!id)
        return;
    }
    const value = await helper.request<ValueInfo>('getValue', {id});
    let domainTypeId = chosenDomain ?? (argument as Node)?.data?.domainTypeId ??
                       (value.kind === 'domain' ? value.domainTypeId
                                                : undefined);
    if (!domainTypeId) {
      const assignments = value.assignments ?? [];
      if (!assignments.length) {
        void vscode.window.showInformationMessage(
            `${value.name} has no recorded domain assignment.`);
        return;
      }
      if (assignments.length === 1)
        domainTypeId = assignments[0].domainTypeId;
      else {
        const choice = await vscode.window.showQuickPick(
            assignments.map(
                (assignment) => ({
                  label : assignment.domain,
                  description : assignment.domainValue ??
                                    assignment.domainValueId ?? '(unresolved)',
                  id : assignment.domainTypeId
                })),
            {placeHolder : 'Choose a domain assignment'});
        domainTypeId = choice?.id;
      }
    }
    if (!domainTypeId)
      return;
    const trace = await helper.request<TraceResult>(
        'trace',
        {valueId : id, domainTypeId, ...(instanceId ? {instanceId} : {})});
    const document = await vscode.workspace.openTextDocument({
      language : 'markdown',
      content : traceMarkdown(value, domainTypeId, trace)
    });
    await vscode.window.showTextDocument(
        document, {preview : true, viewColumn : vscode.ViewColumn.Beside});
  }

  async function chooseValue(pageMethod: 'searchValues'|'sourceMatches',
                             params: object): Promise<ValueInfo|undefined> {
    if (!helper)
      return;
    let offset = 0;
    while (true) {
      const result = await helper.request<Page<ValueInfo>>(
          pageMethod, {...params, offset, limit : 100});
      const choices =
          result.items.map((value) => ({
                             label : value.name,
                             description : `${value.module} · ${value.type}`,
                             value
                           }));
      if (offset + result.items.length < result.total)
        choices.push({
          label : 'More results…',
          description :
              `${result.total - offset - result.items.length} remaining`,
          value : undefined as unknown as ValueInfo
        });
      const choice = await vscode.window.showQuickPick(
          choices, {placeHolder : `${result.total} matching values`});
      if (!choice)
        return undefined;
      if (!choice.value) {
        offset += result.items.length;
        continue;
      }
      return choice.value;
    }
  }

  async function chooseDomain(): Promise<DomainInfo|undefined> {
    if (!helper)
      return undefined;
    let offset = 0;
    while (true) {
      const result = await helper.request<Page<DomainInfo>>(
          'listDomains', {offset, limit : 100});
      const choices: Array<vscode.QuickPickItem&{domain?: DomainInfo}> =
          result.items.map((domain) => ({
                             label : domain.name,
                             description : `${domain.nodes} nodes · ID ${
                                 domain.id}`,
                             domain
                           }));
      if (offset + result.items.length < result.total)
        choices.push({label : 'More domains…', domain : undefined});
      const choice = await vscode.window.showQuickPick(
          choices, {placeHolder : 'Choose a domain'});
      if (!choice)
        return undefined;
      if (choice.domain)
        return choice.domain;
      offset += result.items.length;
    }
  }

  async function chooseDomainNode(domainTypeId: string,
                                  prompt: string, componentIndex?: number):
      Promise<ValueInfo|undefined> {
    if (!helper)
      return undefined;
    const query = await vscode.window.showInputBox({
      prompt,
      placeHolder : 'Value or port name, module name, or numeric ID'
    });
    if (query === undefined)
      return undefined;
    let offset = 0;
    while (true) {
      const result = await helper.request<Page<ValueInfo>>('domainItems', {
        domainTypeId,
        ...(componentIndex !== undefined ? {componentIndex} : {}),
        category : 'nodes',
        query,
        offset,
        limit : 100
      });
      if (!result.total) {
        void vscode.window.showInformationMessage(
            `No nodes match “${query}” in this domain.`);
        return undefined;
      }
      const choices: Array<vscode.QuickPickItem&{value?: ValueInfo}> =
          result.items.map((value) => ({
                             label : value.name,
                             description : `${value.module} · ${value.kind} · ID ${
                                 value.id}`,
                             value
                           }));
      if (offset + result.items.length < result.total)
        choices.push({
          label : 'More results…',
          description : `${result.total - offset - result.items.length} remaining`
        });
      const choice = await vscode.window.showQuickPick(
          choices, {placeHolder : `${result.total} matching domain nodes`});
      if (!choice)
        return undefined;
      if (choice.value)
        return choice.value;
      offset += result.items.length;
    }
  }

  context.subscriptions.push(
      vscode.commands.registerCommand('circtDomains.loadReport', loadReport),
      vscode.commands.registerCommand('circtDomains.reloadReport',
                                      () => currentReport &&
                                            loadReport(currentReport)),
      vscode.commands.registerCommand(
          'circtDomains.searchValues',
          async () => {
            if (!helper)
              return;
            const query = await vscode.window.showInputBox(
                {prompt : 'Search value names in the loaded report'});
            if (query) {
              const value = await chooseValue('searchValues', {query});
              if (value)
                await showValueDetails(value.id);
            }
          }),
      vscode.commands.registerCommand(
          'circtDomains.showValuesAtCursor',
          async () => {
            const editor = vscode.window.activeTextEditor;
            if (!helper || !editor || editor.document.uri.scheme !== 'file')
              return;
            const position = editor.selection.active;
            const value = await chooseValue('sourceMatches', {
              path : editor.document.uri.fsPath,
              line : position.line + 1,
              column : position.character + 1
            });
            if (value)
              await showValueDetails(value.id);
          }),
      vscode.commands.registerCommand('circtDomains.traceValue', traceValue),
      vscode.commands.registerCommand(
          'circtDomains.chooseCutEndpoints',
          async (node?: Node) => {
            if (!helper)
              return;
            const requestHelper = helper;
            try {
              const domain = node?.domainTypeId
                                 ? {id : node.domainTypeId,
                                    name : node.domainName ??
                                        node.domainTypeId}
                                 : await chooseDomain();
              if (!domain)
                return;
              let componentIndex = node?.componentIndex;
              const source = await chooseDomainNode(
                  domain.id, 'Search for the first min-cut node',
                  componentIndex);
              if (!source)
                return;
              if (componentIndex === undefined) {
                const component = await requestHelper.request<{index: number}>(
                    'componentForValue',
                    {domainTypeId : domain.id, valueId : source.id});
                componentIndex = component.index;
              }
              const target = await chooseDomainNode(
                  domain.id, 'Search for the second min-cut node',
                  componentIndex);
              if (!target)
                return;
              if (helper !== requestHelper)
                return;
              if (source.id === target.id) {
                void vscode.window.showWarningMessage(
                    'Choose two different nodes for the minimum cut.');
                return;
              }
              const result = await vscode.window.withProgress(
                  {
                    location : vscode.ProgressLocation.Notification,
                    title : `Calculating minimum cut between ${source.name} and ${
                        target.name}`
                  },
                  () => requestHelper.request<CutResult>('minCut', {
                    domainTypeId : domain.id,
                    sourceId : source.id,
                    targetId : target.id
                  }));
              if (helper !== requestHelper)
                return;
              domainsView.setBetweenCut(
                  domain.id, componentIndex, result,
                  `${source.module}.${source.name}`,
                  `${target.module}.${target.name}`);
              await vscode.commands.executeCommand('circtDomains.domains.focus');
              void vscode.window.showInformationMessage(
                  `${domain.name} component ${componentIndex + 1}: cut ${result.count} edge${
                      result.count === 1 ? '' : 's'} between ${source.name} and ${
                      target.name}. Expand its Minimum cut to inspect.`);
            } catch (error) {
              output.appendLine(`Minimum cut failed: ${String(error)}`);
              void vscode.window.showErrorMessage(
                  `Could not calculate minimum cut: ${String(error)}`);
            }
          }),
      vscode.commands.registerCommand('circtDomains.openSource',
                                      async (arg?: Node|Location) => {
                                        const location =
                                            (arg as Node)?.data?.location ??
                                            (arg as Location | undefined);
                                        if (location?.sources)
                                          await openSource(location);
                                      }),
      vscode.commands.registerCommand('circtDomains.showDetails',
                                      async (node?: Node) => {
                                        const id = node?.data?.id;
                                        if (id && node?.kind === 'instance')
                                          await showInstanceDetails(id);
                                        else if (id)
                                          await showValueDetails(id);
                                      }),
      vscode.workspace.onDidChangeConfiguration(async (change) => {
        if (change.affectsConfiguration('circtDomains.sourceRoots') && helper &&
            currentReport) {
          try {
            await helper.request('setSourceRoots',
                                 {roots : rootsFor(currentReport)});
            domainsView.clearCuts();
          } catch (error) {
            output.appendLine(
                `Could not update source roots: ${String(error)}`);
          }
        }
      }),
      vscode.languages.registerHoverProvider({scheme : 'file'}, {
        async provideHover(document, position) {
          if (!helper)
            return undefined;
          try {
            const result =
                await helper.request<Page<ValueInfo>>('sourceMatches', {
                  path : document.uri.fsPath,
                  line : position.line + 1,
                  column : position.character + 1,
                  limit : 10
                });
            if (!result.total)
              return undefined;
            const markdown = new vscode.MarkdownString();
            markdown.appendMarkdown('**FIRRTL domain report**\n\n');
            for (const value of result.items) {
              markdown.appendMarkdown(`**${escapeMarkdown(value.name)}** · ${
                  escapeMarkdown(value.module)}  \n`);
              markdown.appendMarkdown(
                  `Type: \`${value.type.replace(/`/g, '\\`')}\`  \n`);
              for (const assignment of value.assignments ?? []) {
                markdown.appendMarkdown(
                    `${escapeMarkdown(assignment.domain)} → ${
                        escapeMarkdown(assignment.domainValue ??
                                       assignment.domainValueId ??
                                       '(unresolved)')} (${
                        assignment.inferred ? 'inferred' : 'explicit'})  \n`);
              }
              markdown.appendMarkdown('\n');
            }
            if (result.total > result.items.length)
              markdown.appendMarkdown(`+${
                  result.total -
                  result.items
                      .length} more. Run “Show Values at Cursor” to choose one.`);
            if (!loadedSummary?.complete)
              markdown.appendMarkdown('\n\n*This report is partial.*');
            return new vscode.Hover(markdown);
          } catch (error) {
            output.appendLine(`Hover lookup failed: ${String(error)}`);
            return undefined;
          }
        }
      }),
      {dispose : () => {
        pendingLoad?.dispose();
        helper?.dispose();
      }});
}
