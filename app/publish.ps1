# Build the distributable Kestrel Studio: one self-contained KestrelStudio.exe for
# 64-bit Windows 10/11 (no .NET install needed on the target PC), plus the kestrel CLI.
#
#   powershell -ExecutionPolicy Bypass -File app\publish.ps1
#
# Output: app\dist\KestrelStudio.exe and app\dist\cli\kestrel.exe
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$dist = Join-Path $here 'dist'
if (Test-Path $dist) { Remove-Item $dist -Recurse -Force }

dotnet publish (Join-Path $here 'Kestrel.Studio\Kestrel.Studio.csproj') -c Release -o $dist -nologo
if ($LASTEXITCODE -ne 0) { throw 'Kestrel Studio publish failed' }

dotnet publish (Join-Path $here 'Kestrel.Cli\Kestrel.Cli.csproj') -c Release -r win-x64 --self-contained `
    -p:PublishSingleFile=true -o (Join-Path $dist 'cli') -nologo
if ($LASTEXITCODE -ne 0) { throw 'CLI publish failed' }

Get-ChildItem $dist -Filter *.pdb -Recurse | Remove-Item
Get-ChildItem $dist -Recurse -File | Select-Object FullName, @{n='MB';e={[math]::Round($_.Length/1MB,1)}} | Format-Table -AutoSize
