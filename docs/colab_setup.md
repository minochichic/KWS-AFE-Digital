# GitHub ↔ Colab 셋팅 (한 번만 하면 됨)

로컬(맥)에서 코드를 쓰고 → GitHub에 push → Colab Pro에서 pull 해서 GPU 학습.
Colab 쪽은 **읽기 전용 워킹 카피**로 취급한다 (부트스트랩 셀이 매번 `git reset --hard`).

---

## 1. GitHub 저장소 만들기

https://github.com/new 에서:

- **Repository name**: `KWS-AFE-Digital`
- **Private** 권장 (논문 작업물)
- **"Add a README file" / .gitignore / license 전부 체크 해제**
  → 빈 저장소여야 아래 첫 push가 충돌 없이 들어간다.

## 2. 로컬 저장소 연결 (맥에서 1회)

```bash
cd ~/Documents/Minho/KWS-AFE-Digital
git init
git branch -M main
git remote add origin https://github.com/minochichic/KWS-AFE-Digital.git
git add .
git commit -m "Initial scaffold: config layer, guardrail tests, Colab bootstrap"
git push -u origin main
```

push 할 때 비밀번호를 물으면 계정 비밀번호가 아니라 **PAT**(아래 3번)을 넣는다.
매번 묻는 게 싫으면 한 번만:

```bash
git config --global credential.helper osxkeychain
```

## 3. PAT (Personal Access Token) 발급

Colab에서 private repo를 clone 하려면 토큰이 필요하다.

GitHub → 우상단 프로필 → **Settings** → 맨 아래 **Developer settings**
→ **Personal access tokens** → **Fine-grained tokens** → *Generate new token*

| 항목 | 값 |
|---|---|
| Token name | `colab-kws` |
| Expiration | 90 days (만료되면 재발급) |
| Repository access | **Only select repositories** → `KWS-AFE-Digital` |
| Permissions → Repository → **Contents** | **Read-only** |

> Contents를 Read-only로 두는 이유: Colab은 pull만 하면 된다.
> 토큰이 노트북 출력이나 로그에 새더라도 저장소를 덮어쓸 수 없다.
> (Colab에서 push도 하고 싶어지면 그때 Read and write로 올린다.)

*Generate token* → **`github_pat_...` 문자열은 이 화면에서만 보인다. 지금 복사.**

## 4. Colab 시크릿 등록

Colab 노트북 왼쪽 사이드바의 **🔑 (Secrets)** 아이콘 →
**+ Add new secret**

- Name: `GH_TOKEN`
- Value: 방금 복사한 `github_pat_...`
- **Notebook access 토글을 켠다** ← 이걸 안 켜면 `userdata.get`이 실패한다

이제 `notebooks/colab_bootstrap.ipynb`를 Colab에서 열고 (GitHub 탭에서 바로
열거나 업로드) 셀을 위에서부터 실행하면 된다.

## 5. 매일의 작업 흐름

```
맥:    코드 수정 → pytest → git add/commit/push
Colab: 부트스트랩 셀 1 재실행 (fetch + reset --hard) → 학습 셀 실행
```

**Colab 안에서 소스를 고치지 말 것.** 셀 1이 다음 실행 때 날려버린다.
빠르게 실험하고 싶으면 노트북 셀 안에서 하고, 살릴 값이 정해지면 로컬 코드에 반영한다.

## 6. 결과물 다루기

| 산출물 | 어디에 | 이유 |
|---|---|---|
| 데이터셋 (~2.3 GB) | `/content/datasets` (세션 로컬) | Drive는 I/O가 느려 학습이 병목된다 |
| 체크포인트 `.pt` | Drive (`runs/` 심볼릭 링크) | 세션이 끊겨도 살아남음. git에는 넣지 않는다 |
| 정확도/설정 요약 | 작은 `.json`/`.md` → git commit | sweep 결과는 버전 관리 대상 |

`.gitignore`가 `datasets/`, `runs/`, `*.pt`를 이미 막아두었다.
큰 파일을 실수로 push 하면 GitHub 100 MB 제한에 걸려 히스토리 정리가 번거로워진다.

## 7. Colab에서 결과 요약만 되돌려 push 하고 싶다면

PAT을 **Contents: Read and write**로 다시 발급한 뒤:

```python
!git config user.email "choismh1102@gmail.com"
!git config user.name "minochichic"
!git add experiments/results/
!git commit -m "sweep: C=32 T=64 -> 8x.x%"
!git push https://{GH_TOKEN}@github.com/{USER}/{REPO}.git HEAD:main
```

권장하진 않는다. sweep 결과 표를 로컬에 복사해 붙여넣는 편이 히스토리가 깔끔하다.
