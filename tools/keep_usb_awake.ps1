<#
    Stop Windows from powering down the external USB drive during a long run.

    Why: the Kestrel corpus, checkpoints and repo all live on S:, which is a
    Seagate ST1000LM024 5400rpm HDD attached over USB. A Phase D run reads from
    it continuously for ~8.4 days and writes a 1.47 GB checkpoint every hour.
    If Windows suspends the USB port or spins the disk down, the trainer dies
    with an I/O error. WSD resume means at most one checkpoint interval is lost,
    but only if somebody notices.

    What it changes (AC and DC, current power scheme):
      * USB selective suspend            -> Disabled
      * Turn off hard disk after         -> Never
      * Per-device "allow the computer to turn off this device" on USB hubs
        and the disk itself              -> unchecked, where the device exposes it

    All of it is reversible - the previous values are printed and saved before
    anything is written. Requires an elevated shell.
#>
[CmdletBinding()]
param([switch]$Revert)

$ErrorActionPreference = 'Stop'

# GUIDs are stable across Windows versions.
$SUB_USB   = '2a737441-1930-4402-8d77-b2bebba308a3'
$USB_SUSP  = '48e6b7a6-50f5-4782-a5d4-53bb8f07e226'
$SUB_DISK  = '0012ee47-9041-4b5d-9b77-535fba8b1442'
$DISK_IDLE = '6738e2c4-e8a5-4a42-b16a-e040e769756e'

function Show($label, $sub, $setting) {
    $out = powercfg /query SCHEME_CURRENT $sub $setting 2>$null
    $ac = ($out | Select-String 'Current AC Power Setting Index:\s*(0x[0-9a-f]+)').Matches.Groups[1].Value
    $dc = ($out | Select-String 'Current DC Power Setting Index:\s*(0x[0-9a-f]+)').Matches.Groups[1].Value
    Write-Host ("  {0,-28} AC={1}  DC={2}" -f $label, $ac, $dc)
    return @{ ac = $ac; dc = $dc }
}

Write-Host "BEFORE:"
$b1 = Show 'USB selective suspend' $SUB_USB  $USB_SUSP
$b2 = Show 'Turn off hard disk after' $SUB_DISK $DISK_IDLE

$stamp = Join-Path $PSScriptRoot 'power_settings_previous.txt'
if (-not (Test-Path $stamp)) {
    @(
        "saved=$(Get-Date -Format s)"
        "usb_selective_suspend_ac=$($b1.ac)"
        "usb_selective_suspend_dc=$($b1.dc)"
        "disk_idle_ac=$($b2.ac)"
        "disk_idle_dc=$($b2.dc)"
    ) | Out-File $stamp -Encoding utf8
    Write-Host "`nSaved previous values to $stamp"
}

if ($Revert) {
    Write-Host "`nReverting to Windows defaults (USB suspend on, disk idle 20 min)..."
    $usb, $disk = 1, 1200
} else {
    Write-Host "`nApplying: USB selective suspend OFF, hard disk never sleeps..."
    $usb, $disk = 0, 0
}

powercfg /setacvalueindex SCHEME_CURRENT $SUB_USB  $USB_SUSP  $usb
powercfg /setdcvalueindex SCHEME_CURRENT $SUB_USB  $USB_SUSP  $usb
powercfg /setacvalueindex SCHEME_CURRENT $SUB_DISK $DISK_IDLE $disk
powercfg /setdcvalueindex SCHEME_CURRENT $SUB_DISK $DISK_IDLE $disk
powercfg /setactive SCHEME_CURRENT

Write-Host "`nAFTER:"
Show 'USB selective suspend' $SUB_USB  $USB_SUSP   | Out-Null
Show 'Turn off hard disk after' $SUB_DISK $DISK_IDLE | Out-Null

# --- per-device power management ------------------------------------------
# Not every device exposes this; the ones that do get it cleared. Failures are
# reported, never fatal - this is belt-and-braces on top of the scheme settings.
Write-Host "`nPer-device power management (USB hubs + the S: disk):"
$enable = [bool]$Revert
$targets = Get-CimInstance -ClassName Win32_PnPEntity |
    Where-Object { $_.PNPClass -in @('USB','DiskDrive') -and $_.Status -eq 'OK' }

foreach ($d in $targets) {
    $id = $d.PNPDeviceID
    try {
        $pm = Get-CimInstance -Namespace root\wmi -ClassName MSPower_DeviceEnable `
              -ErrorAction Stop | Where-Object InstanceName -like "$id*"
        if ($pm) {
            Set-CimInstance -InputObject $pm[0] -Property @{ Enable = $enable } -ErrorAction Stop
            Write-Host ("  set Enable={0}  {1}" -f $enable, $d.Name)
        }
    } catch {
        # device does not expose the setting, or is not writable - fine
    }
}

Write-Host "`nDone. To undo:  .\keep_usb_awake.ps1 -Revert"
