function Get-Capnp {
    [CmdletBinding()]
    param(
        [Parameter()]
        [string]$config = "Release",
        [string]$version = "0.10.2"
    )

    $ErrorActionPreference = 'Stop'

    $capnp = "capnproto-c++-win32-$version"
    $capnpZip = "$capnp.zip"
    $capnpUrl = "https://capnproto.org/$capnpZip"

    # Directory of this script
    $rootDir = (get-item $PSScriptRoot ).parent.FullName
    $extDir = [System.IO.Path]::Combine($rootDir, "ext")
    
    # Create if not exists
    if (!(Test-Path $extDir)) {
        New-Item -ItemType Directory -Force -Path $extDir
    }

    # cd to ext dir
    Push-Location $extDir

    # Download capnp
    if (!(Test-Path $capnpZip)) {
        Write-Host "Downloading $capnpUrl"
        Invoke-WebRequest -Uri $capnpUrl -OutFile $capnpZip
    }

    # Unzip capnp
    if (!(Test-Path $capnp)) {
        Write-Host "Unzipping $capnpZip"
        Expand-Archive -Path $capnpZip -DestinationPath $capnp
    }
    
    # Build CapNProto
    Write-Host "Building CapNProto"
    Push-Location "$capnp"
    Push-Location "capnproto-c++-$version"
    # Build without fiber support (requires exceptions, disabled by LLVM).
    cmake . -DCMAKE_CXX_FLAGS="-DKJ_USE_FIBERS=0" -DCMAKE_BUILD_TYPE=$config -DCMAKE_INSTALL_PREFIX="$extDir"
    cmake --build . --config $config
    cmake --install .

    # copy cmake to install dir
    # Copy-Item "cmake" -Destination $extDir -Recurse -Force

    # Pop locations
    Pop-Location
    Pop-Location
    Pop-Location
}

. (Join-Path $PSScriptRoot "find-vs.ps1")

# parse the CMake build configuration from the command line
$buildConfig = $args[0]

# Must be either Debug, Release, RelWithDebInfo, or MinSizeRel
if ($buildConfig -notin @("Debug", "Release", "RelWithDebInfo", "MinSizeRel")) {
    Write-Error "Invalid build configuration: $buildConfig"
    exit 1
}

Get-Capnp $buildConfig
