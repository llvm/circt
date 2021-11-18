import re


def formatBitString(bitstring, format="hex"):
  if format == "hex":
    return f"0x{int(bitstring, 2):x}"
  elif format == "bin":
    return bitstring
  elif format == "dec":
    # Assume that length of bitstring is actual bitwidth of signal
    w = len(bitstring)
    res = int(bitstring, 2)
    if res >= 2**(w - 1):
      res -= 2**w
    return res
  else:
    raise ValueError(f"Unknown value format '{format}'")


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