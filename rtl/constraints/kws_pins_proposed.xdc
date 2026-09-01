# 핀 배치 제안 — KC705 + XM105 디버그 카드 (2026-09-01)
#
# ⚠️ 아직 읽히지 않는다. build.tcl 은 kws_top.xdc 만 read_xdc 한다.
#    동료가 이 배선을 확인해 주면 그때 kws_top.xdc 로 옮긴다.
#
# 이 파일의 요점은 Tcl 이 아니라 **제안**이다. docs/ICD.md 7 의 4번(핀 ↔ ch 매핑)을
# 여태 "동료가 정해줄 것"으로 두고 기다렸는데, 그럴 이유가 없다. 커넥터와 핀은
# 우리가 고를 수 있고, 동료는 그 표대로 배선하면 된다. 기다리는 대신 표를 준다.
#
# ---------------------------------------------------------------------------
# 매핑 사슬 — 표 세 개가 이어진다
#
#   ch 번호 → AFE 헤더 핀 → (리본) → XM105 J1 핀 → FMC LPC 핀 → FPGA 핀
#            └ 이 파일이 제안 ┘      └ UG537 표1-10 ┘  └ UG810 표1-29 ┘
#
# 왜 XM105 J1 인가: LPC 로 쓸 수 있는 XM105 커넥터는 J1(40핀, LA00~LA19),
# J20(16핀, LA20~LA27), J16(12핀, LA28~LA31 + 전원/GND), J15(6핀) 뿐이다.
# J2/J3/J23 은 HB·HA 계열이라 **HPC 전용**이고 KC705 의 LPC(J2)에서는 안 나온다.
#
# 왜 9~24번 핀인가: J1 의 1~8 번은 LA00_CC / LA01_CC 로 **클럭 가능 핀**이다.
# 평범한 입력으로 쓰면 나중에 외부 클럭이 필요해졌을 때 자리가 없다. 9~24 는
# 2x20 헤더에서 5~12 번 열에 해당해 **물리적으로 연속**이므로, 16가닥 리본에
# 2x8 IDC 소켓 하나로 끝난다.
#
# ---------------------------------------------------------------------------
# ch │ XM105 J1 │ FMC LPC 신호 │ FPGA 핀
# ---┼----------┼--------------┼---------
#  0 │    9     │ LA02_P       │ AF20
#  1 │   10     │ LA12_P       │ AA20
#  2 │   11     │ LA02_N       │ AF21
#  3 │   12     │ LA12_N       │ AB20
#  4 │   13     │ LA03_P       │ AG20
#  5 │   14     │ LA13_P       │ AB24
#  6 │   15     │ LA03_N       │ AH20
#  7 │   16     │ LA13_N       │ AC25
#  8 │   17     │ LA04_P       │ AH21
#  9 │   18     │ LA14_P       │ AD21
# 10 │   19     │ LA04_N       │ AJ21
# 11 │   20     │ LA14_N       │ AE21
# 12 │   21     │ LA05_P       │ AG22
# 13 │   22     │ LA15_P       │ AC24
# 14 │   23     │ LA05_N       │ AH22
# 15 │   24     │ LA15_N       │ AD24
#
# ch 순서가 J1 핀 번호 순서와 같다. IDC 리본은 도선 n 이 핀 n 에 가므로,
# **리본 도선 순서 = 채널 순서**가 된다 -- 사람이 세기 쉬운 쪽이 덜 틀린다.
# LA 쌍의 P/N 을 갈라 쓰는 것은 문제가 아니다: 우리 신호는 단일단이고
# 차동으로 쓰지 않는다.
#
# ⚠️ 검증 없이 믿지 말 것. 배선한 뒤 반드시 **주파수 스윕 대각선 테스트**를
#    돌린다 (125 Hz -> 5 kHz 스윕을 마이크에 들려주고, 활성 채널이 ch00 에서
#    ch15 로 단조 이동하는지). 섞이면 대각선이 부서지고 어떻게 섞였는지까지
#    보인다. 이걸 건너뛰면 증상이 "정확도가 좀 낮다"뿐이라 못 찾는다.
#    transfer_v4 의 행 순서가 역순이었던 게 같은 부류의 사고다.

set_property PACKAGE_PIN AF20 [get_ports {cmp[0]}]
set_property PACKAGE_PIN AA20 [get_ports {cmp[1]}]
set_property PACKAGE_PIN AF21 [get_ports {cmp[2]}]
set_property PACKAGE_PIN AB20 [get_ports {cmp[3]}]
set_property PACKAGE_PIN AG20 [get_ports {cmp[4]}]
set_property PACKAGE_PIN AB24 [get_ports {cmp[5]}]
set_property PACKAGE_PIN AH20 [get_ports {cmp[6]}]
set_property PACKAGE_PIN AC25 [get_ports {cmp[7]}]
set_property PACKAGE_PIN AH21 [get_ports {cmp[8]}]
set_property PACKAGE_PIN AD21 [get_ports {cmp[9]}]
set_property PACKAGE_PIN AJ21 [get_ports {cmp[10]}]
set_property PACKAGE_PIN AE21 [get_ports {cmp[11]}]
set_property PACKAGE_PIN AG22 [get_ports {cmp[12]}]
set_property PACKAGE_PIN AC24 [get_ports {cmp[13]}]
set_property PACKAGE_PIN AH22 [get_ports {cmp[14]}]
set_property PACKAGE_PIN AD24 [get_ports {cmp[15]}]

# 전압은 VADJ 가 정한다 (docs/ICD.md 7.1). LPC 의 VCCO 는 VADJ 이고 기본값이
# 2.5 V 이므로, 아래 한 줄은 **VADJ 를 1.8 V 로 내린 뒤에만** 맞다.
# 동글(EVM USB-TO-GPIO)이 없어 2.5 V 로 둔다면 여기가 LVCMOS25 가 되고
# 브레이크아웃 카드에 레벨 변환기가 들어간다.
set_property IOSTANDARD LVCMOS18 [get_ports {cmp[*]}]

# ---------------------------------------------------------------------------
# 전원과 접지 — J1 에는 없다
#
# J1 40핀은 **전부 신호**다. 접지가 한 핀도 없다. 그래서 별도 배선이 필요하다:
#
#   GROUND    XM105 J16 핀 3, 4   (2가닥)
#             XM105 J15 핀 2      (1가닥)
#   전원      XM105 J16 핀 1, 2 = HDR_POWER
#             J6 에 셧트를 어떻게 꽂느냐로 **3.3 V 또는 VADJ** 를 고른다
#             (UG537 p18). 셧트는 킷에 들어 있지 않으니 따로 구해야 한다.
#
# 권장: J16 의 HDR_POWER 를 **3.3 V** 로 두고 AFE 보드에서 자체 LDO 로 1.8 V 를
# 만든다. 그러면 AFE 의 1.8 V 가 VADJ 와 무관해지고(전압 레벨 결정과 전원이
# 분리된다), LDO 의 PSRR 이 한 겹 더 붙으며, 디커플링이 30 cm 케이블 끝이 아니라
# 부하 바로 옆에 온다. AFE 능동 소자는 OPA379 49개 x 2.9 µA + 비교기 16개로
# 약 150~280 µA 라 3.3 V 레일에 부담이 안 된다.
#
# 접지는 "가닥마다 교대"까지 필요 없다 -- 비교기 에지가 µs 급이라 크로스토크가
# 미미하다. 대신 **여러 가닥**이 필요하다: 16가닥이 동시에 스위칭할 때 접지
# 리턴이 한 가닥이면 그 인덕턴스에서 전위차가 생기고, AFE 의 0.9 V 중간전위는
# 16채널 공통 기준이라 그 잡음이 전 채널 공통 모드로 들어간다.
