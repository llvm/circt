from base import *
from utils import *


class HandshakeBundle(SignalBundle):

  def __init__(self, vcdParent, dotParent, valid, ready, data):
    self.vcdParent = vcdParent
    self.dotParent = dotParent
    self.valid = valid
    self.ready = ready
    self.data = data

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


class IndexedHandshakeModule(IndexedModule):

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
            HandshakeBundle(vcdParent=self.vcdName,
                            dotParent=self.dotName,
                            valid=validSig,
                            ready=readySig,
                            data=dataSig))

  def resolveToDot(self, edges):
    """ Given a set of DotEdge's, tries to resolve the bundles of this module
            to edges in the dot file.
        """
    for bundle in self.bundles:
      bundle.resolveToDot(edges)


class HandshakeTarget(Target):

  def __init__(self):
    super().__init__(bundleType=HandshakeBundle,
                     moduleType=IndexedHandshakeModule)
