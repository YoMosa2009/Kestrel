<#
    Set the GPU power ceiling (needs an elevated shell).

    Why: during Phase D the RTX 3060 sat at 84-85 C with the fan already pinned
    at 100% and sw_thermal_slowdown ACTIVE, dropping clocks 1950 -> 1897 MHz.
    The fan curve has nothing left to give, so the only software lever that
    reduces heat is the power ceiling. The card draws 155-166 W against a 170 W
    cap and is NOT power-limited, so trimming the cap trades a little power for
    clocks that thermal throttling is already taking away.

    Safe and reversible. Nothing here touches the running training job: the
    power limit is a driver-level setting applied live, and CUDA work continues
    uninterrupted. To undo, re-run with the default:

        .\set_power_limit.ps1 -Watts 170

    Usage:
        .\set_power_limit.ps1 -Watts 140
        .\set_power_limit.ps1 -Revert
#>
[CmdletBinding()]
param(
    [int]$Watts = 0,
    [switch]$Revert
)

$ErrorActionPreference = 'Stop'

function Get-Field($pattern) {
    $line = (nvidia-smi -q -d POWER | Select-String $pattern | Select-Object -First 1)
    if (-not $line) { return $null }
    if ($line.Line -match '([\d\.]+)\s*W') { return [double]$Matches[1] }
    return $null
}

# --- read the envelope from the driver; never assume it ----------------------
$current = Get-Field 'Current Power Limit'
$default = Get-Field 'Default Power Limit'
$min     = Get-Field 'Min Power Limit'
$max     = Get-Field 'Max Power Limit'

Write-Host "GPU power envelope (reported by driver):"
Write-Host ("  current {0} W | default {1} W | min {2} W | max {3} W" -f $current, $default, $min, $max)

if ($Revert) { $Watts = [int]$default }

if ($Watts -le 0) {
    Write-Host "`nNo -Watts given and -Revert not set. Nothing changed."
    exit 0
}

# --- refuse anything outside what the driver itself allows -------------------
if ($Watts -lt $min -or $Watts -gt $max) {
    Write-Error ("Refusing: {0} W is outside the driver's allowed range {1}-{2} W." -f $Watts, $min, $max)
    exit 1
}

# Record the pre-change value next to the script so the revert value survives
# even if this window is lost.
$stamp = Join-Path $PSScriptRoot 'power_limit_previous.txt'
if (-not (Test-Path $stamp)) {
    "default=$default`nbefore=$current`nsaved=$(Get-Date -Format s)" |
        Out-File $stamp -Encoding utf8
    Write-Host "Recorded pre-change limit to $stamp"
}

Write-Host ("`nSetting power limit: {0} W -> {1} W ..." -f $current, $Watts)
nvidia-smi -pl $Watts
if ($LASTEXITCODE -ne 0) {
    Write-Error "nvidia-smi -pl failed (exit $LASTEXITCODE). Are you elevated?"
    exit $LASTEXITCODE
}

Start-Sleep -Seconds 2
$now = Get-Field 'Current Power Limit'
Write-Host ("`nVerified: current power limit is now {0} W" -f $now)
if ([math]::Abs($now - $Watts) -gt 0.5) {
    Write-Warning "Driver did not accept the requested value. Check for vendor software overriding it."
}
Write-Host ("To undo:  nvidia-smi -pl {0}" -f [int]$default)
