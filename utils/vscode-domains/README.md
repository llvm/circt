# CIRCT FIRRTL Domains for VS Code

This extension explores `circt-domain-inference` JSON reports with schema
version 3. It reads existing reports; it never runs `firtool` or compiles a
design. A native C++ helper indexes the report outside VS Code's JavaScript
process, so large reports do not enter a VS Code text buffer.

## Use

1. Install the VSIX matching your machine: `darwin-arm64`, `darwin-x64`, or
   `linux-x64`.
2. Run **FIRRTL Domains: Load Domain Report** from the Command Palette and pick
   the JSON file. The command is also available from the **Domains** view.
3. Open **Domains** and expand a domain definition to see its connected
   components. Each component contains its explicit associations, recorded
   connections, nodes, minimum cut, and illegal crossings. Expand **Minimum
   cut** to find the fewest recorded edges whose removal splits that
   component. Choose **Choose two nodes…** there, or run **FIRRTL Domains:
   Choose Min-Cut Endpoints**, to find a cut between nodes in the same
   component. Cut edges can be opened at their source locations. For a failed
   report, components involved in an illegal crossing receive the theme's
   error color and an error badge. Expand **Illegal crossings** under either
   component to see the shortest recorded path between the reported domain
   annotations. The failed connection appears as a separate step.
4. Use **Search Values** or **Show Values at Cursor** to inspect a value's
   assignments. **Trace Value** on a value, or from the Command Palette,
   opens the shortest recorded provenance route in a temporary editor.
5. Hover over a source location. Use **Open Source** on a Domains item to
   navigate to its source file.

The graph for a domain contains report values assigned to that domain, domain
values of its type, and values connected to those anchors by provenance edges.
Unassigned constraint fragments are omitted. Every provenance relation with
two endpoints is an undirected edge of weight one. This
includes association, constraint, alias, and instance-binding edges; separate
rows between the same values count separately. A summarized edge also counts
as one report edge. Self-edges cannot cross a cut and are omitted from the
connection list. A domain type such as `ClockDomain` can have many connected
components because a design can contain many independent clocks. Minimum cuts
are shown within each component; choosing two nodes restricts the second
choice to the first node's component. Cut
results describe recorded relations, not physical wires. Module values are
shared templates rather than separate copies for every instance. Ports on an
extmodule have no value IDs in the report, so their explicit associations are
listed under **Unmapped records** when they cannot be assigned to a component.
Those ports cannot be selected as cut endpoints. A partial report may omit
relations, so its cut counts apply only to the recorded graph.

The Domains view labels reports with `complete: false` as partial. Missing values
or edges in a partial report are inconclusive. A crossing path can be
unavailable when its partial report lacks an annotation or a connecting edge.
Earlier version 3 reports do not contain crossing records. Unsupported
versions and malformed reports are rejected.

Source files are searched beneath open workspace folders, the report's parent
directory, and `circtDomains.sourceRoots`. Relative configured roots use the
first workspace folder, or the report directory when no folder is open.
Unresolved source locations remain visible in the report but cannot be opened.
Reported lines and columns are converted from one-based positions; point
locations match only their exact position. Ranges use the reported endpoint as
VS Code's exclusive end position. Column coordinates are assumed to count
UTF-16 code units because the report does not identify their producer's unit.

## Development and packaging

The extension uses the helper packaged in its `bin/<platform>-<arch>` directory.
Set `circtDomains.helperPath` to a locally built helper to run from a checkout.
Relative helper paths use the first workspace folder.
The C++ target is `circt-domain-report-server`; it links only LLVM Support and
does not require the FIRRTL compiler libraries. Configure CIRCT as described
in the repository's `AGENTS.md`, then build that target when builds are wanted:

```sh
ninja -C build bin/circt-domain-report-server
cd utils/vscode-domains
npm install
npm run compile
npm run package -- --target darwin-arm64 --helper ../../build/bin/circt-domain-report-server
```

The package script copies the supplied binary into a matching VSIX and removes
the staging copy. It does not build the helper. The manual GitHub workflow
`domainReportVSCode.yml` builds and packages macOS arm64, macOS x64, and Linux
x64 variants. No VSIX or native executable is checked into the source tree.

After building the helper and installing npm dependencies, use `npm test` for
the protocol and Domains tree checks. Use `npm run test:vscode` for paging and
source navigation checks in the VS Code extension host.
