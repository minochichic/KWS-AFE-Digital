# board_stats 를 SWD 로 읽는다 (보드에 LED 가 없다).
#
#   powershell -File firmware\stm32_wake\tools\read_stats.ps1            # 한 번
#   powershell -File firmware\stm32_wake\tools\read_stats.ps1 -Watch 2   # 2 초마다 (Ctrl+C)
#
# Stop 2 중에 읽으면 RAM 이 0 으로 읽힌다 (DBG_STOP 을 켜도). BOARD_TEST 빌드는 10 ms 마다
# 3 ms 깨어 있으므로, 그 순간에 걸릴 때까지 다시 읽는다 -- ticks 가 0 이면 버린다.
param(
  [double]$Watch = 0,
  [string]$Elf = "$PSScriptRoot\..\kws_wake\Debug\kws_wake.elf"
)

$cli = Get-ChildItem "C:\ST" -Recurse -Filter STM32_Programmer_CLI.exe -ErrorAction SilentlyContinue |
       Select-Object -First 1 -ExpandProperty FullName
$nm  = Get-ChildItem "C:\ST" -Recurse -Filter arm-none-eabi-nm.exe -ErrorAction SilentlyContinue |
       Select-Object -First 1 -ExpandProperty FullName
$sym = & $nm -S $Elf | Select-String ' board_stats$'
if (-not $sym) { throw "board_stats not found in $Elf" }
$addr = '0x' + ($sym.Line -split '\s+')[0]
$size = '0x' + ($sym.Line -split '\s+')[1]

function Get-Words {
  for ($t = 0; $t -lt 40; $t++) {
    $o = & $cli -c port=SWD mode=HOTPLUG -r32 $addr $size 2>&1 | Select-String '^0x2000'
    $w = @()
    foreach ($l in $o) {
      $w += ($l.Line -split ':')[1].Trim() -split '\s+' | ForEach-Object { [Convert]::ToUInt32($_, 16) }
    }
    if ($w.Count -ge 10 -and $w[0] -ne 0) { return , $w }
  }
  return $null
}

# board.h board_stats_t 의 배치를 그대로 따른다
function Show($w) {
  $s = "ticks={0} stops={1} clips={2} ok={3} timeouts={4} bad={5} match={6}" -f `
       $w[0], $w[1], $w[2], $w[3], $w[4], $w[5], $w[6]
  $s += "`n  last_frame=0x{0:X4} edges_or=0x{1:X4} max_on={2} last_class={3}" -f `
        ($w[7] -band 0xFFFF), ($w[9] -band 0xFFFF), (($w[9] -shr 16) -band 0xFF), (($w[7] -shr 16) -band 0xFF)
  if ($w.Count -ge 18) {
    $e = @(); for ($i = 10; $i -lt 18; $i++) { $e += ($w[$i] -band 0xFFFF); $e += ($w[$i] -shr 16) }
    $s += "`n  edges ch0..15 = [" + ($e -join ',') + "]"
  }
  $s
}

do {
  $w = Get-Words
  if ($w) { "[{0:HH:mm:ss}] " -f (Get-Date) + (Show $w) } else { "read failed (board off or SWD busy?)" }
  if ($Watch -gt 0) { Start-Sleep -Seconds $Watch }
} while ($Watch -gt 0)
