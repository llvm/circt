import argparse
import readchar
from vcdvcd import VCDVCD
import tempfile
import os
import subprocess
from dataclasses import dataclass
from collections import defaultdict
import re
from flask import Flask, render_template, send_file, jsonify
import threading
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
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

# Identifier for deciding how to format values from the VCD file in the .dot file.
VALUE_FORMAT = None

# Handle for the HSDbg instance.
hsdbg = None


def formatBitString(bitstring):
  """ Formats a bit string based on the set VALUE_FORMAT."""
  if VALUE_FORMAT == "hex":
    return f"0x{int(bitstring, 2):x}"
  elif VALUE_FORMAT == "bin":
    return bitstring
  elif VALUE_FORMAT == "dec":
    # Assume that length of bitstring is actual bitwidth of signal
    w = len(bitstring)
    res = int(bitstring, 2)
    if res >= 2**(w - 1):
      res -= 2**w
    return res
  else:
    raise ValueError(f"Unknown value format '{VALUE_FORMAT}'")


def joinHierName(lhs, rhs):
  """Joins two hierarhical names together. This is identical for both the
       vcd and the dot file."""
  return f"{lhs}.{rhs}"


def splitOnTrailingNumber(s):
  m = re.search(r'\d+$', s)
  if m:
    grp = m.group()
    lhs = s.rsplit(grp, 1)
    return lhs, grp
  return s, None


def resolveVCDHierarchy(vcd):
  """ Generates a module from the source vcd object. This is needed due to the
    function missing an implementation in VCDVCD."""
  hier = {}
  for sig in vcd.signals:
    curHier = hier
    for module in sig.split(".")[:-1]:
      if module not in curHier:
        curHier[module] = {}
      curHier = curHier[module]
  return hier


# Possible custom attributes; these are the attributes which may appear
# in the comment section after an edge definition in the .dot file.

# Which output port of the source node this edge is connected to.
CA_OUTPUT = "output"
# Which input port of the sink node this edge is connected to.
CA_INPUT = "input"


@dataclass(frozen=True, order=True)
class DotEdge:
  src: str  # name of source node
  dst: str  # name of destination node
  attributes: set  # keep as a list to make hashable
  line: int  # line number in the original file
  # custom attributes are attributes that are not part of the standard dot language.
  customAttributes: set

  def getCustomAttr(self, name):
    for attr in self.customAttributes:
      if attr[0] == name:
        return attr[1]
    return None

  def getAttr(self, name):
    for attr in self.attributes:
      if attr[0] == name:
        return attr[1]
    return None

  def __str__(self):
    """String representation of this edge in the DOT graph. This is used for
           generating the modified .dot file."""
    out = ""
    out += (f'\"{self.src}\" -> \"{self.dst}\"')
    if len(self.attributes) > 0:
      out += " ["
      for attr in self.attributes:
        out += f'{attr[0]}=\"{attr[1]}\" '
      out += "]"

    if len(self.customAttributes) > 0:
      out += "// "
      for attr in self.customAttributes:
        out += f'{attr[0]}=\"{attr[1]}\" '
    out += '\n'
    return out

  def isFromInput(self):
    # An input node will not have an output port.
    return self.getAttr(CA_OUTPUT) is None

  def isToOutput(self):
    # An output node will not have an input port.
    return self.getAttr(CA_INPUT) is None


@dataclass
class HandshakeBundle:
  valid: str
  ready: str
  data: str
  vcdParent: str
  dotParent: str

  def isResolved(self):
    return self.edge != None

  # Dot edge which this handshake bundle represents. The edge should be
  # considered immutable once it is resolved; the edge will reference the
  # edge in the (unmodified) source dot file.
  edge: DotEdge = None

  def getEdgeStateAttrs(self, vcd, cycle):
    """ Returns a set of attributes that are to be added to the edge to
        indicate the state of the edge.
        """
    if self.edge == None:
      raise Exception("Handshake bundle not resolved")

    attrs = []

    # Handshake state
    isReady = vcd[self.ready][cycle] == "1"
    isValid = vcd[self.valid][cycle] == "1"
    attrs.append(("penwidth", "3"))

    if isReady and isValid:
      attrs.append(("color", "green"))
    elif isReady or isValid:
      attrs.append(("color", "orange"))
      attrs.append(("arrowhead", "dot" if isValid else "odot"))
    else:
      attrs.append(("color", "red"))

    # Data value
    if self.data != None:
      data = vcd[self.data][cycle]
      if data != "x":
        attrs.append(("label", formatBitString(data)))

    return attrs

  def dotBaseName(self):
    dotName = self.valid.replace(self.vcdParent, self.dotParent)
    dotName = dotName.rsplit("_valid")[0]
    return dotName

  def resolveToDot(self, edges):
    """ Resolves this handshake bundle to a dot edge.
        This is done by finding the dot edge that has the same bundle basename
        as this handshake bundle. In here we have space for a bunch of heuristics...
        whatever gets us closer to mapping all edges!
        """
    # First, locate all of the candidate edges. These are the edges which
    # originate from the source of this handshake bundle.
    basename = self.dotBaseName()

    def checkEdge(edge):
      # Check if there exists a direct name match; this is true for in- and output variables
      # which, due to the single-use requirement of handshake circuits should
      # only ever have 1 edge with the variable as a source or destination.
      if edge.src == basename or edge.dst == basename:
        return True

      # Check if there exists an edge where the edge basename + source port or basename + sinkport
      # matches the handshake bundle basename.
      srcPort = edge.getCustomAttr(CA_OUTPUT)
      if srcPort and joinHierName(edge.src, srcPort) == basename:
        return edge

      dstPort = edge.getCustomAttr(CA_INPUT)
      if dstPort and joinHierName(edge.dst, dstPort) == basename:
        return edge

      return None

    # A bit of course filtering to disregard things which we know cannot be
    # this edge (edges of other parent modules).
    filteredEdges = [e for e in edges if self.dotParent in e.src]
    candidates = [e for e in filteredEdges if checkEdge(e)]

    if len(candidates) == 0:
      return
    elif len(candidates) == 1:
      self.edge = candidates[0]


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


class IndexedModule:

  def __init__(self, signals, dotName, vcdName) -> None:
    self.signals = signals
    self.dotName = dotName
    self.vcdName = vcdName
    self.bundles = []  # list of handshake bundles
    self.resolveVcdBundles()

  def vcdToDotName(self, sig):
    return sig.replace(f".{self.vcdName}.", f".{self.dotName}.")

  def resolveVcdBundles(self):
    # Based on the hierarchical name of this module, resolves any handshake
    # bundles within it.
    def getDimensionless(sig):
      # strip a number surrounded by square brackets from the end of basesig.
      return re.sub(r"(\[\d*(:\d*)?\])", "", sig)

    def getBasename(sig):
      baseSig = getDimensionless(sig)
      # strip occurences of ready valid and data string at the end of sig
      baseSig = baseSig.rsplit("_ready")[0]
      baseSig = baseSig.rsplit("_valid")[0]
      baseSig = baseSig.rsplit("_data")[0]
      return baseSig

    bundles = defaultdict(lambda: [])
    for sig in self.signals:
      bundles[getBasename(sig)].append(sig)

    for basename, bundle in bundles.items():

      def resolveBundleSig(name):
        opt = [x for x in bundle if getDimensionless(x).endswith(f"_{name}")]
        if len(opt) == 1:
          return opt[0]
        return None

      dataSig = resolveBundleSig("data")
      validSig = resolveBundleSig("valid")
      readySig = resolveBundleSig("ready")

      if validSig and readySig:
        self.bundles.append(
            HandshakeBundle(validSig,
                            readySig,
                            dataSig,
                            vcdParent=self.vcdName,
                            dotParent=self.dotName))

  def resolveToDot(self, edges):
    """ Given a set of DotEdge's, tries to resolve the bundles of this module
            to edges in the dot file.
        """
    for bundle in self.bundles:
      bundle.resolveToDot(edges)


class HSDbg():

  def __init__(self, dotFile, vcdFile):
    # Maintain the source code of the dot file. Since there aren't any good
    # python libraries for in-place modification of
    self.dotFile = DotFile(dotFile)
    self.workingImagePath = self.setupWorkingImage(dotFile)
    # The handshake bundles which have been successfully mapped to a set of
    # VCD signals and the corresponding dot edges.
    self.handshakeBundles = []

    self.vcd = None
    self.parseVCD(vcdFile)
    self.cycle = self.beginTime()
    self.resolveDotToVCD()

    # Create the graph.
    self.updateCurrentCycle(self.cycle)

  def resolveDotToVCD(self):
    """ This function uses the edges gathered from the .dot file to try and
            create an association with handshake bundles in the vcd file. Be aware
            that quite a few assumptions are placed on the format of both the vcd
            file and the dot file.
        """
    # Step 1: infer top module name
    top = "TOP"

    hier = resolveVCDHierarchy(self.vcd)
    if len(hier) != 1 or "TOP" not in hier:
      raise Exception("Expected \"TOP\" as top-level in vcd hierarchy")
    topHier = hier["TOP"]
    if len(topHier) != 1:
      raise Exception("Expected a single module under the \"TOP\" module.")
    dutHier = topHier[list(topHier.keys())[0]]
    dutName = list(topHier.keys())[0]

    # Funky heuristic: the CIRCT dot dump will name a top module "foo0" but
    # FIRRTL will name it "foo_0".
    dotDutName = "".join([s for s in dutName.rsplit("_", 1)])
    dotDutHierName = joinHierName(top, dotDutName)

    modules = {}
    afterTopHier = False

    def indexModule(vcdHierName, dotHierName):
      # Find all items under this module
      items = [s for s in self.vcd.signals if s.startswith(vcdHierName)]
      # Filder those which are in submodules. For this, we just check if
      # there any any more '.' characters in the name.
      signals = [
          s for s in items if s.replace(f"{vcdHierName}.", "").count(".") == 0
      ]
      modules[vcdHierName] = IndexedModule(signals, dotHierName, vcdHierName)

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
          self.handshakeBundles.append(bundle)

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
    cycle = int(cycle)
    if cycle < self.beginTime():
      print(f"Capping at minimum time ({self.beginTime()})")
      cycle = self.beginTime()
    elif cycle > self.endTime():
      print(f"Capping at maximum time ({self.endTime()})")
      cycle = self.endTime()
    self.cycle = cycle

    self.updateGraph()

  def updateGraph(self):
    # Update edge state of all handshake bundle edges.
    for bundle in self.handshakeBundles:
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


def start_interactive_mode(port):
  msg = f"""=== Handshake interactive simulation viewer ===

Usage:
    Open "localhost:{port}" in a browser to view the simulation.

    left arrow: step forward 1 step in vcd time
    right arrow: step backward 1 step in vcd time
    goto <cycle>: step to a specific cycle in vcd time

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
      address = input("Cycle: ")
      hsdbg.goto(address)
    else:
      print(f"Unknown command '{command}'")


def runImageServer(port):
  global hsdbg
  # A simple flask application which serves index.html to continuously update
  # the svg file.
  app = Flask(__name__)
  app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

  @app.route("/")
  def index():
    return render_template("index.html")

  @app.route("/cycle")
  def cycle():
    return jsonify(cycle=hsdbg.currentCycle())

  # Serve the svg file on any path request.
  @app.route("/<path:path>")
  def serveFile(path):
    return send_file(hsdbg.workingImagePath, mimetype="image/svg+xml")

  app.run(host="localhost", port=port)


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
  args = parser.parse_args()

  VALUE_FORMAT = args.format

  hsdbg = HSDbg(args.dotfile, args.vcdfile)

  # Start the image server and a browser to monitor the image.
  server = threading.Thread(target=lambda: runImageServer(args.port))
  server.start()

  # Start interactive mode.
  start_interactive_mode(args.port)
