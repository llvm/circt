void emitHarrisBody(Value img) {
  auto wndwImg = emitLineBuffer(img, imgDims, kernelDims);
  auto Ix = emitDotProduct2D(wndwImg, kernelX);
  auto Iy = emitDotProduct2D(wndwImg, kernelY);
  auto Ixx = emitMultiplyStreams(Ix, Ix);
  auto Iyy = emitMultiplyStreams(Iy, Iy);
  auto Ixy = emitMultiplyStreams(Ix, Iy);
  auto wndwIxx = emitLineBuffer(Ixx, IDims, kernelDims);
  auto wndwIyy = emitLineBuffer(Iyy, IDims, kernelDims);
  auto wndwIxy = emitLineBuffer(Ixy, IDims, kernelDims);
  auto Sxx = emitReductionSum2D(wndwIxx, kernelDims);
  auto Syy = emitReductionSum2D(wndwIyy, kernelDims);
  auto Sxy = emitReductionSum2D(wndwIxy, kernelDims);
  auto det = emitDet(Sxx, Syy, Sxy);
  auto trace = emitTrace(Sxx, Syy);
  auto harris = emitHarris(det, trace);
}
