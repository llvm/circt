##===- utils/get-grpc.ps1 - Install gRPC (for ESI runtime) ---*- Script -*-===##
# 
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
#
##===----------------------------------------------------------------------===##

$ErrorActionPreference = "Stop"

# Create the 'ext' directory relative to the script location
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$extDir = Join-Path $scriptDir "..\ext"
New-Item -ItemType Directory -Path $extDir -Force | Out-Null

Write-Output $extDir
Set-Location $extDir

$vcpkgRoot = $env:VCPKG_ROOT

# Check if VCPKG_ROOT is set
if (-Not $vcpkgRoot) {
    Write-Output "VCPKG_ROOT is not set..."
    # Install vcpkg if vcpkg executable isn't available in path
    if (Test-Path -Path ".\vcpkg.exe") {
        Write-Output "vcpkg executable found..."
        # set vcpkgRoot from the vcpkg executable path
        $vcpkgRoot = (Get-Item .\vcpkg.exe).Directory.FullName
    }
    else {
        # Is vcpkg available in $extDir?
        if (Test-Path -Path ".\vcpkg") {
            Write-Output "vcpkg found in ext directory..."
            $vcpkgRoot = (Get-Item .\vcpkg).FullName
        }
        else {
            Write-Output "Downloading and installing vcpkg..."
            # Download and install vcpkg
            git clone https://github.com/microsoft/vcpkg
            Set-Location .\vcpkg
            .\bootstrap-vcpkg.bat
            .\vcpkg integrate install
            $vcpkgRoot = (Get-Item .\vcpkg.exe).Directory.FullName
            Set-Location $extDir
        }
    }
}

Write-Output "VCPKG_ROOT: $vcpkgRoot"

# Install zlib, gRPC and protobuf
& "$vcpkgRoot\vcpkg.exe" install zlib grpc protobuf

# And integrate
& "${vcpkgRoot}\vcpkg.exe" integrate install

# vcpkg should now have printet a CMAKE_TOOLCHAIN_FILE path to set when you
# cmake configure circt/mlir.
