import argparse
import readchar
from vcdvcd import VCDVCD
import tempfile
import os
import subprocess
from viewer import HSDbgImageServer
from utils import *
from handshake import *
"""HSDbg
HSDbg is a script which couples together a .dot representation of a CIRCT
handshake program, and a .vcd trace of a simulation on the same program.
This is done by correlating the .vcd hierarchy with the DOT hierarchy, and
correlating the VCD signals with what looks to be the same values in the DOT
file. Since .dot vertices are created on the handshake operation level,
the script relies on additional attributes in the .dot file to disambiguated
which in- and output edges of a node corresponds to which signal in the .vcd.

.dot files are rendered to .svg images and served by a web server. This server
runs a simple web page which will automatically update the shown image.
"""


class DotFile:
  """ A class which handles in-place modification of a dot file.
    This is unfortunately not available in any existing graphviz library.
    This class doesn't parse the vcd grammer. Rather, it looks for edges in the
    input file and registers these edges. It then allows the user to add
    attributes to edges and rewrite the file.
    """

  def __init__(self, filename):
    self.rawDot = []  # List of lines
    self.modifiedDot = []
    self.edgeToLine = {}
    self.lineToEdge = {}
    self.edges = set()
    self.parseDot(filename)
    self.reset()

  def reset(self):
    # Resets the modifiedDot to the rawDot.
    self.modifiedDot = self.rawDot

  def addAttributesToEdge(self, edge, attrs):
    if edge not in self.edges:
      raise Exception(f"Edge not found: {edge}")
    # Edges are frozen to allow for hashing, so we need to make a copy.
    edgeAttrs = set(edge.attributes)
    for attr in attrs:
      edgeAttrs.add(attr)
    self.modifiedDot[edge.line] = str(
        DotEdge(edge.src, edge.dst, edgeAttrs, edge.line,
                edge.customAttributes))

  def parseCustomAttributes(self, attrString):
    """ Parses custom attributes on a handshake graphviz edge."""
    attrList = set()
    for attr in attrString.split(" "):
      if "=" in attr:
        k, v = attr.split("=")
        v = v.replace('"', "")
        attrList.add(((k, v)))
    return frozenset(attrList)

  def parseDotEdge(self, i, line):
    """Parses a graphviz file edge. An edge has the format:
                "from" -> "to" [optional attributes] // custom attributes
        """
    src, rest = line.split("->")[0].strip(), line.split("->")[1].strip()

    rest = rest.split("//")
    customAttributes = frozenset()
    if len(rest) > 1:
      customAttributes = self.parseCustomAttributes(rest[1].strip())
    rest = rest[0].strip()
    dstSplit = rest.split("[")
    dst = dstSplit[0]
    src = src.replace('"', '').strip()
    dst = dst.replace('"', '').strip()
    # split [.* from dst and also return the second part
    attributes = frozenset()
    if len(dstSplit) > 1:
      attrStr = dstSplit[1].replace(']', '')
      # attributes is a list of key value pairs with '=' in between
      # and separated by spaces. Parse it as a list of pairs
      attributes = [x.strip() for x in attrStr.split(" ")]
      attributes = [x.split("=") for x in attributes if len(x) > 0]
      attributes = frozenset({
          (x[0].replace("\"", ""), x[1].replace("\"", "")) for x in attributes
      })
    edge = DotEdge(src, dst, attributes, i, customAttributes)
    self.edgeToLine[edge] = i
    self.lineToEdge[i] = edge
    self.edges.add(edge)

  def parseDot(self, filename):
    with open(filename, 'r') as f:
      self.rawDot = f.readlines()

    # Create edge to line mapping
    for i, line in enumerate(self.rawDot):
      if "->" in line:
        self.parseDotEdge(i, line)

  def dump(self, path):
    """ write the modified dot file to path."""
    with open(path, "w") as f:
      f.write("".join(self.modifiedDot))


class HSDbg():

  def __init__(self, target, dotFile, vcdFile, printBundles):
    # Factory for building signal bundles and modules.
    self.target = target

    # Maintain the source code of the dot file. Since there aren't any good
    # python libraries for in-place modification of
    self.dotFile = DotFile(dotFile)
    self.workingImagePath = self.setupWorkingImage(dotFile)
    # The bundles which have been successfully mapped to a set of
    # VCD signals and the corresponding dot edges.
    self.bundles = []

    self.vcd = None
    self.parseVCD(vcdFile)
    self.cycle = self.beginTime()
    self.resolveDotToVCD()

    if printBundles:
      # Print bundles and exit.
      self.printBundles()
      return

    # Create the graph.
    self.updateCurrentCycle(self.cycle)

  def printBundles(self):
    for b in self.bundles:
      b.printResolvedEdge()

  def resolveDotToVCD(self):
    """ This function uses the edges gathered from the .dot file to try and
            create an association with handshake bundles in the vcd file. Be aware
            that quite a few assumptions are placed on the format of both the vcd
            file and the dot file.
        """
    # Infer top module name
    hier = resolveVCDHierarchy(self.vcd)
    if len(hier) != 1:
      raise Exception("Expected one top-level in vcd hierarchy")
    top = list(hier.keys())[0]
    topHier = hier[top]
    modules = {}

    def buildModule(self, *argv):
      return self.target.moduleType(*argv)

    def indexModule(vcdHierName, dotHierName):
      # Find all items under this module
      items = [s for s in self.vcd.signals if s.startswith(vcdHierName)]
      # Filder those which are in submodules. For this, we just check if
      # there any any more '.' characters in the name.
      signals = [
          s for s in items if s.replace(f"{vcdHierName}.", "").count(".") == 0
      ]
      modules[vcdHierName] = buildModule(self, signals, dotHierName,
                                         vcdHierName)

    def recurseModuleHierarchy(vcdHierName, dotHierName, subHier):
      if subHier == {}:
        indexModule(vcdHierName, dotHierName)
      else:
        for module in subHier:
          indexModule(vcdHierName, dotHierName)
          recurseModuleHierarchy(joinHierName(vcdHierName, module),
                                 joinHierName(dotHierName, module),
                                 subHier[module])

    recurseModuleHierarchy("TOP", "TOP", topHier)

    # We've now indexed all the modules. Now we need to resolve the graph edges.
    for module in modules.values():
      module.resolveToDot(self.dotFile.edges)

    # Finally, we iterate through the modules, gathering up the handshake
    # bundles which successfully resolved.
    for module in modules.values():
      for bundle in module.bundles:
        if bundle.isResolved():
          self.bundles.append(bundle)

    return

  def currentCycle(self):
    return self.cycle

  def setupWorkingImage(self, dotFile):
    temp_dir = tempfile.gettempdir()
    dotFileName = dotFile.split("/")[-1]
    temp_path = os.path.join(temp_dir, dotFileName + ".svg")
    return temp_path

  def beginTime(self):
    return self.vcd.begintime

  def endTime(self):
    return self.vcd.endtime

  def parseVCD(self, vcdFile):
    self.vcd = VCDVCD(vcdFile)

  def updateCurrentCycle(self, cycle):
    try:
      cycle = int(cycle)
    except ValueError:
      print("Invalid cycle value: " + str(cycle))
      return

    if cycle < self.beginTime():
      print(f"Capping at minimum time ({self.beginTime()})")
      cycle = self.beginTime()
    elif cycle > self.endTime():
      print(f"Capping at maximum time ({self.endTime()})")
      cycle = self.endTime()
    self.cycle = cycle

    self.updateGraph()

  def updateGraph(self):
    # Update edge state of all bundle edges.
    for bundle in self.bundles:
      self.dotFile.addAttributesToEdge(
          bundle.edge, bundle.getEdgeStateAttrs(self.vcd, self.cycle))

    # Dump the modified dot file to a temporary file.
    workingDotfile = self.workingImagePath + ".dot"
    self.dotFile.dump(workingDotfile)

    # Run 'dot' on the file to produce an svg file and place it in the
    # working image path, updating any viewers that have the file opened.
    subprocess.call(
        ["dot", workingDotfile, "-Tsvg", "-o", self.workingImagePath])

    # Reset modifications to original dot file after each update.
    self.dotFile.reset()

  # A callback for handling left arrow presses.
  def left(self):
    self.updateCurrentCycle(self.cycle - 1)

    # A callback for handling right arrow presses.
  def right(self):
    self.updateCurrentCycle(self.cycle + 1)

    # A callback for handling "goto" commands.
  def goto(self, cycle):
    self.updateCurrentCycle(cycle)


def start_interactive_mode(port, hsdbg):
  msg = f"""=== Handshake interactive simulation viewer ===

Usage:
    Open "http://localhost:{port}" in a browser to view the simulation.

    right arrow: step forward 1 timestep in vcd time
    left arrow: step backwards 1 timestep in vcd time
    g <cycle>: step to a specific cycle in vcd time

"Entering interactive mode. Type 'q' to quit."
"""
  print(flush=True)
  print(msg)
  while True:
    print(f"\r> [{hsdbg.currentCycle()}] ", end="")
    command = readchar.readkey()
    if command == "q":
      print("Exiting...")
      break
    elif command == '\x1b[D':
      hsdbg.left()
    elif command == '\x1b[C':
      hsdbg.right()
    elif command == "g":
      address = input("Goto cycle: ")
      hsdbg.goto(address)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description=
      """A command line tool for visualizing handshake execution traces.
        
Compatible .dot files may be generated by using circt-opt:
    \"circt-opt --handshake-print-dot ${handshake IR file}\"
    
Legend:
    Red edge    : !(valid | ready)
    Yellow edge : valid | ready [hollow dot: ready, filled dot: valid]
    Green edge  : valid & ready
    Dashed edge : control signal
    """,
      formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument("dotfile", help="The dotfile to use.", type=str)
  parser.add_argument("vcdfile", help="The vcdfile to use.", type=str)
  parser.add_argument(
      "-f",
      "--format",
      help="The value format of data signals. options are 'hex, bin, dec'.",
      type=str,
      default="dec")
  parser.add_argument(
      "-p",
      "--port",
      help="The port to run the image server on. default='8080'",
      type=int,
      default=8080)
  parser.add_argument(
      "--print-bundles-only",
      help="Print the bundles that were successfully resolved, and exit.",
      action="store_true")
  args = parser.parse_args()

  VALUE_FORMAT = args.format

  # For now, only support handshake targets
  target = HandshakeTarget()

  hsdbg = HSDbg(target, args.dotfile, args.vcdfile, args.print_bundles_only)
  if not args.print_bundles_only:
    viewer = HSDbgImageServer(hsdbg, args.port)
    viewer.start()
    start_interactive_mode(args.port, hsdbg)
