@echo off
REM build_runner.bat — Compile vulkan_runner.cpp using MSVC.
REM Called from WSL via: powershell.exe -Command "cmd.exe /c 'C:\...\build_runner.bat'"
REM Also works directly from a Developer Command Prompt for VS 2022.

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if not exist build mkdir build
cl.exe /std:c++17 /EHsc /O2 /I "C:\VulkanSDK\1.4.350.0\Include" tools\vulkan_runner.cpp /Fobuild\vulkan_runner.obj /Fe:build\vulkan_runner.exe /link "C:\VulkanSDK\1.4.350.0\Lib\vulkan-1.lib" /SUBSYSTEM:CONSOLE
