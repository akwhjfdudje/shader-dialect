# build_runner.ps1 — Compile vulkan_runner.cpp for Windows using MSVC.
#
# Prerequisites:
#   - Visual Studio Build Tools (cl.exe on PATH or in a Dev Shell)
#   - Vulkan SDK 1.4 installed at %VULKAN_SDK% or C:\VulkanSDK\1.4.350.0
#
# Usage (from Developer PowerShell for VS 2022):
#   powershell -File tools/build_runner.ps1
#
# Or run from WSL:
#   powershell.exe -File tools/build_runner.ps1

$ErrorActionPreference = "Stop"

# Locate Vulkan SDK
$VulkanSDK = $env:VULKAN_SDK
if (-not $VulkanSDK) {
    $VulkanSDK = "C:\VulkanSDK\1.4.350.0"
}

$VulkanInclude = "$VulkanSDK\Include"
$VulkanLib = "$VulkanSDK\Lib\vulkan-1.lib"
$SrcFile = "tools\vulkan_runner.cpp"
$OutObj = "build\vulkan_runner.obj"
$OutExe = "build\vulkan_runner.exe"

# Ensure build directory exists
if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Path "build" | Out-Null
}

# Compile
Write-Host "Compiling $SrcFile ..."
Write-Host "  Vulkan SDK: $VulkanSDK"

$clArgs = @(
    "/std:c++17", "/EHsc", "/O2", "/Fo:$OutObj", "/Fe:$OutExe",
    "/I", $VulkanInclude,
    $SrcFile,
    "/link", $VulkanLib, "/SUBSYSTEM:CONSOLE"
)

& "cl.exe" $clArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Compilation failed." -ForegroundColor Red
    exit 1
}

Write-Host "SUCCESS: $OutExe built." -ForegroundColor Green
