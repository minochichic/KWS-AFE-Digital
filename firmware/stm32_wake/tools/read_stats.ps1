# board_stats 를 SWD 로 읽는다 (보드에 LED 가 없다).
#
#   read_stats.ps1                    # 한 번
#   read_stats.ps1 -Watch 2           # 2 초마다 (Ctrl+C)
#   read_stats.ps1 -Inject 0xFFFF     # 테스트 빌드: 가짜 칸 하나를 넣고 트리거 → 전송 → 결과까지 본다
#   read_stats.ps1 -Replay            # 재생 빌드: 골든 클립을 한 바퀴 보내고 FPGA 답을 정답지와 대조
#
# Stop 2 중에 읽으면 RAM 이 0 으로 읽힌다 (DBG_STOP 을 켜도). BOARD_TEST 빌드는 10 ms 마다
# 3 ms 깨어 있으므로 그 순간에 걸릴 때까지 다시 읽는다. 끝 단어 magic 이 맞아야 온전한 읽기다.
param(
  [double]$Watch = 0,
  [string]$Inject = "",
  [switch]$Replay,
  [string]$Elf = "$PSScriptRoot\..\kws_wake\Debug\kws_wake.elf"
)

$MAGIC = [Convert]::ToUInt32('B0A2D57A', 16)   # 0xB0A2D57A 리터럴은 PS 5.1 에서 음수 Int32 가 된다

$cli = Get-ChildItem "C:\ST" -Recurse -Filter STM32_Programmer_CLI.exe -ErrorAction SilentlyContinue |
       Select-Object -First 1 -ExpandProperty FullName
$nm  = Get-ChildItem "C:\ST" -Recurse -Filter arm-none-eabi-nm.exe -ErrorAction SilentlyContinue |
       Select-Object -First 1 -ExpandProperty FullName
$syms = & $nm -S $Elf

function Get-Sym($name) {
  $l = $syms | Select-String " $name$"
  if (-not $l) { throw "$name not found in $Elf" }
  $f = $l.Line -split '\s+'
  return @{ addr = '0x' + $f[0]; size = [Convert]::ToInt32($f[1], 16) }
}
$st  = Get-Sym 'board_stats'

function Read-Once {
  $o = & $cli -c port=SWD mode=HOTPLUG -r32 $st.addr $st.size 2>&1 | Select-String '^0x2000'
  $w = @()
  foreach ($l in $o) {
    $w += ($l.Line -split ':')[1].Trim() -split '\s+' | ForEach-Object { [Convert]::ToUInt32($_, 16) }
  }
  if ($w.Count -eq $st.size / 4 -and $w[0] -ne 0 -and $w[-1] -eq $MAGIC) { return , $w }
  return $null
}

# 읽는 도중 Stop 2 에 들어갔다 깨면 앞이나 가운데만 0 이 될 수 있다 (magic 은 멀쩡).
# 그래서 두 번 연속 읽어 누적 카운터(clips 이후)가 같을 때만 받아들인다.
function Get-Stats {
  $prev = $null
  for ($t = 0; $t -lt 60; $t++) {
    $w = Read-Once
    if (-not $w) { $prev = $null; continue }
    if ($prev -and (($w[2..6] -join ',') -eq ($prev[2..6] -join ',')) -and $w[11] -eq $prev[11]) {
      return , $w
    }
    $prev = $w
  }
  return $null
}

# board.h board_stats_t 의 배치를 그대로 따른다
function Show($w) {
  $bg = [int32]$w[10]
  $state = @('WARMUP', 'IDLE', 'CAPTURE', 'HOLD')[($w[9] -shr 24) -band 0xFF]
  $s = "[{0:HH:mm:ss}] ticks={1} stops={2} clips={3} ok={4} timeouts={5} bad={6} match={7}" -f `
       (Get-Date), $w[0], $w[1], $w[2], $w[3], $w[4], $w[5], $w[6]
  $s += "`n  state={0} bg={1:N2}ch (trigger at on>={2:N2}) last_frame=0x{3:X4} max_on={4} injected={5}" -f `
        $state, ($bg / 32), (($bg + 128) / 32), ($w[7] -band 0xFFFF), (($w[9] -shr 16) -band 0xFF), $w[11]
  if ($w[14] -ne 0) {
    $s += "`n  replay: " + (Show-Replay $w)
  }
  if ($w.Count -ge 24) {
    $e = @(); for ($i = 15; $i -lt 23; $i++) { $e += ($w[$i] -band 0xFFFF); $e += ($w[$i] -shr 16) }
    $s += "`n  edges_or=0x{0:X4} edges ch0..15 = [{1}]" -f ($w[9] -band 0xFFFF), ($e -join ',')
  }
  $s
}

function Show-Replay($w) {
  $state = @('idle', 'running', 'done')[[int][Math]::Min($w[14], 2)]
  $s = "{0}, sent {1}, ok {2}, timeouts {3}, bad {4}, match {5}" -f `
       $state, $w[2], $w[3], $w[4], $w[5], $w[6]
  if ($w[12] -ne [Convert]::ToUInt32('FFFFFFFF', 16)) {
    $why = @('class mismatch', 'timeout')[[int][Math]::Min((($w[13] -shr 16) -band 0xFF), 1)]
    if ((($w[13] -shr 16) -band 0xFF) -gt 1) { $why = 'bad sync/CRC' }
    $s += "; first bad clip #{0}: got {1}, expected {2} ({3})" -f `
          $w[12], ($w[13] -band 0xFF), (($w[13] -shr 8) -band 0xFF), $why
  }
  $s
}

if ($Replay) {
  $go = Get-Sym 'board_replay_go'      # 재생 빌드에만 있다 -- 없으면 여기서 멈춘다
  for ($t = 0; $t -lt 60; $t++) {
    & $cli -c port=SWD mode=HOTPLUG -w32 $go.addr 0x00000001 2>&1 | Out-Null
    $w = Read-Once
    if ($w -and $w[14] -eq 1) { break }
  }
  if (-not ($w -and $w[14] -eq 1)) { throw "replay did not start (is this a WAKE_REPLAY build?)" }
  "started"
  do {
    Start-Sleep -Seconds 3
    $w = Read-Once
    if ($w) { "[{0:HH:mm:ss}] {1}" -f (Get-Date), (Show-Replay $w) }
  } while (-not $w -or $w[14] -ne 2)
  $n = $w[2]
  if ($w[6] -eq $n) { "PASS: $($w[6])/$n clips match predictions_fixed" }
  else { "FAIL: $($w[6])/$n match" }
  return
}

if ($Inject) {
  $inj = Get-Sym 'board_inject'
  $mask = [Convert]::ToUInt32(($Inject -replace '^0x', ''), 16)
  $a = Get-Stats
  if (-not $a) { throw "read failed (board off or SWD busy?)" }
  "before:"; Show $a
  $done = $false
  for ($t = 0; $t -lt 60 -and -not $done; $t++) {
    & $cli -c port=SWD mode=HOTPLUG -w32 $inj.addr ('0x{0:X8}' -f $mask) 2>&1 | Out-Null
    $b = Get-Stats
    if ($b -and $b[11] -gt $a[11]) { $done = $true }
  }
  if (-not $done) { throw "inject not consumed after $t tries" }
  "injected 0x{0:X4} after {1} write(s); waiting 1.5 s (capture 0.7 s + send/timeout 0.22 s + busy)" -f $mask, $t
  Start-Sleep -Milliseconds 1500
  $c = Get-Stats
  "after:"; Show $c
  "delta: injected +{0}, clips +{1}, ok +{2}, timeouts +{3}, bad +{4}" -f `
    ($c[11] - $a[11]), ($c[2] - $a[2]), ($c[3] - $a[3]), ($c[4] - $a[4]), ($c[5] - $a[5])
  return
}

do {
  $w = Get-Stats
  if ($w) { Show $w } else { "read failed (board off or SWD busy?)" }
  if ($Watch -gt 0) { Start-Sleep -Seconds $Watch }
} while ($Watch -gt 0)
