from dataclasses import dataclass
from collections import defaultdict

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
    raise NotImplementedError("Subclass must implement resolveVcdBundles")

  def resolveToDot(self, edges):
    """ Given a set of DotEdge's, tries to resolve the bundles of this module
            to edges in the dot file.
        """
    for bundle in self.bundles:
      bundle.resolveToDot(edges)


@dataclass
class SignalBundle:
  vcdParent: str
  dotParent: str

  # Dot edge which this handshake bundle represents. The edge should be
  # considered immutable once it is resolved; the edge will reference the
  # edge in the (unmodified) source dot file.
  edge: DotEdge = None

  def isResolved(self):
    return self.edge != None

  def printResolvedEdge(self):
    print(str(self.edge).strip())

  def resolveToDot(self, edges):
    """ Resolves this handshake bundle to a dot edge.
        This is done by finding the dot edge that has the same bundle basename
        as this handshake bundle. In here we have space for a bunch of heuristics...
        whatever gets us closer to mapping all edges!
        """
    raise NotImplementedError("Subclass must implement resolveToDot")


@dataclass
class Target:
  # A Target defines the level of abstraction which we're trying to visually
  # simulate. bundleType and moduleType should be assigned to the classes of
  # the specific implementation for the target abstraction.
  bundleType: type
  moduleType: type