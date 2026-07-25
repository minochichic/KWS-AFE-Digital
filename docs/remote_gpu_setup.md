# 원격 GPU 개발 환경 셋업 (WSL2 + RTX 5070 Ti)

Colab Pro 컴퓨팅 한도 소진 시 대체용. **Mac(개발) → VSCode Remote-SSH → 원격 윈도우
WSL2(GPU 학습)** 구성으로 학습을 이어간다. Colab 노트북(`notebooks/colab_bootstrap.ipynb`)
대응물은 `notebooks/gpu_local.ipynb`.

> 이 문서는 실제로 겪은 삽질을 기준으로 **깨끗한 재현 절차만** 적는다. 중간에 시도했다
> 버린 경로(예: torchcodec CUDA 빌드)는 §7 트러블슈팅에만 남긴다.

---

## 0. 구성 / 워크플로

```
Mac (편집·commit·push) ── GitHub(origin) ── 원격 WSL2 (pull·GPU 학습)
```

- **역할 분리**: Mac = 편집/push 전담, 원격 = pull/실행 전담. 한 파일을 양쪽에서
  고치지 않는다(충돌 방지).
- 원격은 WSL2라 디스크가 **영속적** — Colab처럼 세션마다 초기화되지 않는다. 그래서
  `git reset --hard`·Google Drive 마운트·데이터 재다운로드가 전부 불필요.

---

## 1. SSH 연결 (Mac → 원격)

Mac `~/.ssh/config`:

```
Host gpu
  HostName <원격 공인 IP>
  Port <포워딩 포트>
  User <계정>
  IdentityFile ~/.ssh/id_ed25519
  ConnectTimeout 60
  ServerAliveInterval 30
  ServerAliveCountMax 4
  TCPKeepAlive yes
```

- 키 등록: `ssh-copy-id -i ~/.ssh/id_ed25519.pub -p <포트> <계정>@<IP>`
- **`ConnectTimeout 60` 중요**: 짧은 간격으로 여러 번 접속하면 sshd `MaxStartups`에
  걸려 첫 배너가 20~30초 지연될 수 있다. VSCode 기본 타임아웃(15s)이면 실패하니
  `remote.SSH.connectTimeout`도 60으로 올린다.

---

## 2. VSCode Remote-SSH

1. 확장 **Remote - SSH** 설치 → `Remote-SSH: Connect to Host` → `gpu`
2. **원격에 확장 별도 설치**: `Python`, `Jupyter`(Microsoft)를 **"SSH: gpu" 섹션**에
   설치해야 노트북 실행 버튼·커널이 뜬다. 로컬에만 있으면 안 됨.
3. `File → Open Folder` → 원격 경로(예: `/home/<계정>/KWS-AFE-Digital`)

---

## 3. Python venv + PyTorch (Blackwell 필수)

```bash
sudo apt install -y python3.12-venv    # WSL 기본에 빠져 있음
cd ~/KWS-AFE-Digital
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# RTX 5070 Ti = Blackwell(sm_120). 반드시 cu128 빌드. 일반 pip torch(cu124 등)는
# "no kernel image" 에러가 난다.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

검증: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability(0))"`
→ `True (12, 0)` 이어야 Blackwell 정상.

---

## 4. Jupyter 커널

```bash
pip install jupyter ipykernel pyzmq
pip install -r requirements-colab.txt   # pyyaml / soundfile / pytest (torch는 건드리지 않음)
```

- **`pyzmq` 꼭 포함**: 없으면 VSCode에서 커널이 "Connecting…"에서 무한 대기한다
  (ipykernel이 ZeroMQ 채널을 못 연다).
- 커널 선택: 노트북 우상단 → `.venv/bin/python`.

---

## 5. 오디오 디코딩: torchcodec (CPU) + ffmpeg  ← 가장 까다로운 부분

**torchaudio 2.9+ 는 `torchaudio.load()`를 TorchCodec에 위임**한다. 그래서 wav 로딩에
torchcodec + FFmpeg가 필요하다(Colab의 옛 torchaudio는 soundfile로 바로 됐음).

```bash
# torch 버전과 매칭(torch 2.11 ↔ torchcodec 0.11.1). CPU 빌드를 쓴다.
pip install "torchcodec==0.11.1" --index-url https://download.pytorch.org/whl/cpu

# torchcodec 런타임이 요구하는 FFmpeg 시스템 라이브러리 (Ubuntu 24.04 = ffmpeg 6)
sudo apt install -y ffmpeg
```

- **왜 CPU 빌드인가**: torchcodec **CUDA 빌드**는 `libnvrtc.so.13`·`libnppicc.so.12`
  같은 CUDA 라이브러리를 로더 경로에서 못 찾아 실패한다(설치돼 있어도 rpath 문제).
  우리는 wav를 **CPU로 디코딩만** 하면 되므로 GPU 디코딩용 CUDA 의존이 없는 CPU 빌드가
  정답이다. cu128 torch와 함께 써도 오디오 디코딩엔 문제없다.
- **버전 매칭**: torchcodec은 torch의 C++ ABI에 묶여 있어 torch 버전과 맞춰야 한다
  (torch 2.11 → torchcodec 0.11.1). CPU 인덱스의 latest(0.15 등)를 그냥 깔면 ABI가
  안 맞을 수 있다.

---

## 6. 최종 검증

```python
from torchcodec.decoders import AudioDecoder      # import 되면 torchcodec+ffmpeg OK
import torchaudio, torch, glob, os
w, sr = torchaudio.load(glob.glob(os.path.expanduser(
    '~/datasets/speech_commands_v2/**/*.wav'), recursive=True)[0])
print(torch.cuda.is_available(), tuple(w.shape), sr)   # True (1, 16000) 16000
```

이후 `notebooks/gpu_local.ipynb`를 위에서부터 실행하면 데이터로더·학습이 돈다.

---

## 7. 트러블슈팅 요약 (이번에 겪은 것)

| 증상 | 원인 | 해결 |
|---|---|---|
| SSH `banner exchange` 타임아웃 | 빠른 재접속 → sshd MaxStartups | `ConnectTimeout` 늘림, 잠시 후 재시도 |
| VSCode 커널 "Connecting…" 무한 | venv에 `pyzmq` 없음 | `pip install pyzmq` |
| 노트북 셀 실행 버튼 안 뜸 | 원격에 Jupyter 확장 미설치 | "SSH: gpu"에 Python/Jupyter 설치 |
| `!git log` 셀이 안 끝남 | VSCode `!`셸이 pty → git 페이저(less) 대기 | `git --no-pager log ...` |
| `ModuleNotFoundError: torchcodec` | torchaudio 2.9+ 가 torchcodec에 위임 | torchcodec 설치 |
| `libnvrtc.so.13` / `libnppicc.so.12` 없음 | torchcodec **CUDA 빌드**의 CUDA lib 경로 문제 | **CPU 빌드**로 교체 |
| `libavutil.so.NN` 없음 | FFmpeg 미설치 | `sudo apt install ffmpeg` |

---

## 부록: 검증된 버전 스냅샷 (2026-07-26)

```
OS         WSL2 Ubuntu 24.04.4   (kernel 6.6.87-microsoft-standard-WSL2)
GPU        NVIDIA RTX 5070 Ti 16GB  (Blackwell sm_120, driver 595.79, CUDA 13.2)
Python     3.12.3
torch      2.11.0+cu128
torchaudio 2.11.0+cu128
torchcodec 0.11.1+cpu
ffmpeg     6.1.1
pyzmq      27.1.0
```
