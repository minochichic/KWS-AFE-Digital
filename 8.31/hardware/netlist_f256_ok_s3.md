# 디지털 판별부 넷리스트 — f256_ok_s3

`python -m wakeup.netlist` 로 생성. 손으로 고치지 말 것 — 생성기를 고친다.

- VDD 1.8 V, 74LVC/74LV 계열 · START 1프레임(채널 1개 이상) · 창 ±2 · timeout 카운트 60
- s1: Template `0 0 0 0 0 0 1 0`, k=7, 창 카운트 0..4
- s2: Template `0 0 0 0 0 X 1 X`, k=6, 창 카운트 6..10

## 배선 검사

통과 — 넷마다 구동원 하나, 떠 있는 입력 없음, 전원 핀 정상.

## 넷리스트 시뮬 대조

10800프레임, 시나리오 12개에서 gatelevel 모델과 **불일치 0** (WAKE 0, 내부 PASS·RUN 0). WAKE 가 선 프레임 1515개.

## 열린 항목

- 50핀 커넥터 핀 번호 -- ICD 확정 후 (CH0..CH7, VDD, GND)
- VLED: 1.8 V 로는 LED 순방향 전압을 못 넘는다. 3.3 V 등 별도 레일 필요
- 32.768 kHz 발진 상수(Rbias, drive 저항, 부하 C) -- 1.8 V 에서 시제품 확인
- CLR 은 자기 종료 펄스다 (CLR -> RUN 리셋 -> 카운터 0 -> 디코드 해제). 74LVC74A /RD 최소 펄스 폭과 대조
- 표준 배치로 적은 핀 번호 -- 실제 구매 부품 데이터시트와 대조

## 판정 노드

| 상태 | m | k | R7 / R8 (E96) | V_TH | 이상값 | margin | 이상 margin | 오프셋·공차 예산 |
|---|---|---|---|---|---|---|---|---|
| s1 | 8 | 7 | 169k / 732k | 1462.4 mV | 1462.5 mV | 112.4 mV | 112.5 mV | 15.7 mV |
| s2 | 6 | 6 | 75k / 825k | 1650.0 mV | 1650.0 mV | 150.0 mV | 150.0 mV | 13.8 mV |

예산 = 비교기 오프셋 최대 8 mV + 분압 1% 저항 최악 + Template 1% 저항 근사. margin 보다 작으면 판정이 바뀌지 않는다.

## 부품 (BOM)

| 참조 | 부품 | 패키지 | 쓰임 | 핀 배치 근거 |
|---|---|---|---|---|
| Q1 | MMBT3904 | SOT-23 | LED 드라이버 | SOT-23 일반 배치 (제조사 데이터시트 대조 필요) |
| U1 | 74LV4060 | SO16 | 프레임 클럭 128 Hz | Nexperia 74LV4060 Rev.5 표 2 (확인). Q7 = ÷256 |
| U2 | 74LVC14A | SO14 | /CH0, /CH1, /CH2, /CH3, /CH4, /CH5 | 7414 표준 배치 (데이터시트 대조 필요) |
| U3 | 74LVC14A | SO14 | /CH6, /CH7, POR (Schmitt), /C5, /C0, /C1 | 7414 표준 배치 (데이터시트 대조 필요) |
| U4 | 74LVC30A | SO14 | START = OR(CH0..CH7) | 7430 표준 배치 (Nexperia 74LVC30A 존재·1.8 V 확인, 핀 대조 필요) |
| U5 | 74LVC00A | SO14 | RUN set, /S1, /S2, (남는 단위: 입력 GND) | 7400 표준 배치 (데이터시트 대조 필요) |
| U6 | 74LV393 | SO14 | 프레임 카운터 6비트 | 74393 표준 배치 (데이터시트 대조 필요) |
| U7 | 74LVC138A | SO16 | 디코드 0..7 | 74138 표준 배치 (데이터시트 대조 필요) |
| U8 | 74LVC138A | SO16 | 디코드 8..15 | 74138 표준 배치 (데이터시트 대조 필요) |
| U9 | 74LVC30A | SO14 | WIN1 = 카운트 0..4 | 7430 표준 배치 (Nexperia 74LVC30A 존재·1.8 V 확인, 핀 대조 필요) |
| U10 | 74LVC30A | SO14 | WIN2 = 카운트 6..10 | 7430 표준 배치 (Nexperia 74LVC30A 존재·1.8 V 확인, 핀 대조 필요) |
| U11 | 74LVC30A | SO14 | CLR = 카운트 60 · CLK | 7430 표준 배치 (Nexperia 74LVC30A 존재·1.8 V 확인, 핀 대조 필요) |
| U12 | 74LVC08A | SO14 | /RD = CLR_N · POR_N, GATE = CLK · RUN, X1 = MATCH1·GATE, X2 = MATCH2·GATE | 7408 표준 배치 (데이터시트 대조 필요) |
| U13 | LPV7215 | SOT-23-5 | s1 정합 비교기 | TI LPV7215 데이터시트 5절 Pin Functions, DBV (확인) |
| U14 | LPV7215 | SOT-23-5 | s2 정합 비교기 | TI LPV7215 데이터시트 5절 Pin Functions, DBV (확인) |
| U15 | 74LVC74A | SO14 | RUN / PASS1 | 7474 표준 배치 (데이터시트 대조 필요) |
| U16 | 74LVC74A | SO14 | PASS2 | 7474 표준 배치 (데이터시트 대조 필요) |
| U17 | 74LVC08A | SO14 | WAKE, (남는 단위: 입력 GND), (남는 단위: 입력 GND), (남는 단위: 입력 GND) | 7408 표준 배치 (데이터시트 대조 필요) |

| 참조 | 종류 | 값 | 1번 | 2번 | 비고 |
|---|---|---|---|---|---|
| Y1 | Y | 32.768kHz 12.5pF | OSC_IN | OSC_X | AB26TRB-32.768KHZ-T 또는 FC-135 |
| R1 | R | 10M | OSC_IN | OSC_OUT | Rbias (시제품 조정) |
| R2 | R | 330k | OSC_OUT | OSC_X | drive 제한 (시제품 조정) |
| C1 | C | 22pF | OSC_IN | GND | 부하 C (시제품 조정) |
| C2 | C | 22pF | OSC_X | GND | 부하 C (시제품 조정) |
| C3 | C | 1uF | VDD | POR_RC | POR |
| R3 | R | 1M | POR_RC | GND | POR (τ ≈ 1 s) |
| R4 | R | 1M | /CH0 | SC1 | s1 ch0 (0) |
| R5 | R | 1M | /CH1 | SC1 | s1 ch1 (0) |
| R6 | R | 1M | /CH2 | SC1 | s1 ch2 (0) |
| R7 | R | 1M | /CH3 | SC1 | s1 ch3 (0) |
| R8 | R | 1M | /CH4 | SC1 | s1 ch4 (0) |
| R9 | R | 1M | /CH5 | SC1 | s1 ch5 (0) |
| R10 | R | 1M | CH6 | SC1 | s1 ch6 (1) |
| R11 | R | 1M | /CH7 | SC1 | s1 ch7 (0) |
| R12 | R | 169k | VDD | VTH1 | s1 분압 상단 R7 |
| R13 | R | 732k | VTH1 | GND | s1 분압 하단 R8 |
| R14 | R | 1M | /CH0 | SC2 | s2 ch0 (0) |
| R15 | R | 1M | /CH1 | SC2 | s2 ch1 (0) |
| R16 | R | 1M | /CH2 | SC2 | s2 ch2 (0) |
| R17 | R | 1M | /CH3 | SC2 | s2 ch3 (0) |
| R18 | R | 1M | /CH4 | SC2 | s2 ch4 (0) |
| R19 | R | 1M | CH6 | SC2 | s2 ch6 (1) |
| R20 | R | 75k | VDD | VTH2 | s2 분압 상단 R7 |
| R21 | R | 825k | VTH2 | GND | s2 분압 하단 R8 |
| R22 | R | 10k | WAKE | Q_B | 베이스 저항 |
| D1 | D | LED | LED_A | LED_K |  |
| R23 | R | 680 | VLED | LED_A | LED 전류 (VLED 3.3 V 가정, 약 2 mA) |
| C4 | C | 100nF | VDD | GND | U1 디커플링 |
| C5 | C | 100nF | VDD | GND | U2 디커플링 |
| C6 | C | 100nF | VDD | GND | U3 디커플링 |
| C7 | C | 100nF | VDD | GND | U4 디커플링 |
| C8 | C | 100nF | VDD | GND | U5 디커플링 |
| C9 | C | 100nF | VDD | GND | U6 디커플링 |
| C10 | C | 100nF | VDD | GND | U7 디커플링 |
| C11 | C | 100nF | VDD | GND | U8 디커플링 |
| C12 | C | 100nF | VDD | GND | U9 디커플링 |
| C13 | C | 100nF | VDD | GND | U10 디커플링 |
| C14 | C | 100nF | VDD | GND | U11 디커플링 |
| C15 | C | 100nF | VDD | GND | U12 디커플링 |
| C16 | C | 100nF | VDD | GND | U13 디커플링 |
| C17 | C | 100nF | VDD | GND | U14 디커플링 |
| C18 | C | 100nF | VDD | GND | U15 디커플링 |
| C19 | C | 100nF | VDD | GND | U16 디커플링 |
| C20 | C | 100nF | VDD | GND | U17 디커플링 |
| C21 | C | 10uF | VDD | GND | 벌크 |

## 부품별 핀


**Q1 MMBT3904**

| 핀 | 이름 | 넷 |
|---|---|---|
| 1 | B | Q_B |
| 2 | E | GND |
| 3 | C | LED_K |

**U1 74LV4060**

| 핀 | 이름 | 넷 |
|---|---|---|
| 1 | Q11 | NC |
| 2 | Q12 | NC |
| 3 | Q13 | NC |
| 4 | Q5 | NC |
| 5 | Q4 | NC |
| 6 | Q6 | NC |
| 7 | Q3 | NC |
| 8 | GND | GND |
| 9 | CTC | NC |
| 10 | RTC | OSC_OUT |
| 11 | RS | OSC_IN |
| 12 | MR | GND |
| 13 | Q8 | NC |
| 14 | Q7 | CLK |
| 15 | Q9 | NC |
| 16 | VCC | VDD |

**U2 74LVC14A**

| 핀 | 이름 | 넷 |
|---|---|---|
| 1 | 1A | CH0 |
| 2 | 1Y | /CH0 |
| 3 | 2A | CH1 |
| 4 | 2Y | /CH1 |
| 5 | 3A | CH2 |
| 6 | 3Y | /CH2 |
| 7 | GND | GND |
| 8 | 4Y | /CH3 |
| 9 | 4A | CH3 |
| 10 | 5Y | /CH4 |
| 11 | 5A | CH4 |
| 12 | 6Y | /CH5 |
| 13 | 6A | CH5 |
| 14 | VCC | VDD |

**U3 74LVC14A**

| 핀 | 이름 | 넷 |
|---|---|---|
| 1 | 1A | CH6 |
| 2 | 1Y | /CH6 |
| 3 | 2A | CH7 |
| 4 | 2Y | /CH7 |
| 5 | 3A | POR_RC |
| 6 | 3Y | POR_N |
| 7 | GND | GND |
| 8 | 4Y | /C5 |
| 9 | 4A | C5 |
| 10 | 5Y | /C0 |
| 11 | 5A | C0 |
| 12 | 6Y | /C1 |
| 13 | 6A | C1 |
| 14 | VCC | VDD |

**U4 74LVC30A**

| 핀 | 이름 | 넷 |
|---|---|---|
| 1 | A | /CH0 |
| 2 | B | /CH1 |
| 3 | C | /CH2 |
| 4 | D | /CH3 |
| 5 | E | /CH4 |
| 6 | F | /CH5 |
| 7 | GND | GND |
| 8 | Y | START |
| 9 | NC9 | NC |
| 10 | NC10 | NC |
| 11 | G | /CH6 |
| 12 | H | /CH7 |
| 13 | NC13 | NC |
| 14 | VCC | VDD |

**U5 74LVC00A**

| 핀 | 이름 | 넷 |
|---|---|---|
| 1 | 1A | START |
| 2 | 1B | POR_N |
| 3 | 1Y | /SD_RUN |
| 4 | 2A | X1 |
| 5 | 2B | WIN1 |
| 6 | 2Y | /S1 |
| 7 | GND | GND |
| 8 | 3Y | /S2 |
| 9 | 3A | X2 |
| 10 | 3B | WIN2 |
| 11 | 4Y | NC |
| 12 | 4A | GND |
| 13 | 4B | GND |
| 14 | VCC | VDD |

**U6 74LV393**

| 핀 | 이름 | 넷 |
|---|---|---|
| 1 | 1CP | CLK |
| 2 | 1MR | RUN_N |
| 3 | 1Q0 | C0 |
| 4 | 1Q1 | C1 |
| 5 | 1Q2 | C2 |
| 6 | 1Q3 | C3 |
| 7 | GND | GND |
| 8 | 2Q3 | NC |
| 9 | 2Q2 | NC |
| 10 | 2Q1 | C5 |
| 11 | 2Q0 | C4 |
| 12 | 2MR | RUN_N |
| 13 | 2CP | C3 |
| 14 | VCC | VDD |

**U7 74LVC138A**

| 핀 | 이름 | 넷 |
|---|---|---|
| 1 | A0 | C0 |
| 2 | A1 | C1 |
| 3 | A2 | C2 |
| 4 | /E1 | C3 |
| 5 | /E2 | C4 |
| 6 | E3 | /C5 |
| 7 | /Y7 | /D7 |
| 8 | GND | GND |
| 9 | /Y6 | /D6 |
| 10 | /Y5 | NC |
| 11 | /Y4 | /D4 |
| 12 | /Y3 | /D3 |
| 13 | /Y2 | /D2 |
| 14 | /Y1 | /D1 |
| 15 | /Y0 | /D0 |
| 16 | VCC | VDD |

**U8 74LVC138A**

| 핀 | 이름 | 넷 |
|---|---|---|
| 1 | A0 | C0 |
| 2 | A1 | C1 |
| 3 | A2 | C2 |
| 4 | /E1 | C4 |
| 5 | /E2 | C5 |
| 6 | E3 | C3 |
| 7 | /Y7 | NC |
| 8 | GND | GND |
| 9 | /Y6 | NC |
| 10 | /Y5 | NC |
| 11 | /Y4 | NC |
| 12 | /Y3 | NC |
| 13 | /Y2 | /D10 |
| 14 | /Y1 | /D9 |
| 15 | /Y0 | /D8 |
| 16 | VCC | VDD |

**U9 74LVC30A**

| 핀 | 이름 | 넷 |
|---|---|---|
| 1 | A | /D0 |
| 2 | B | /D1 |
| 3 | C | /D2 |
| 4 | D | /D3 |
| 5 | E | /D4 |
| 6 | F | VDD |
| 7 | GND | GND |
| 8 | Y | WIN1 |
| 9 | NC9 | NC |
| 10 | NC10 | NC |
| 11 | G | VDD |
| 12 | H | VDD |
| 13 | NC13 | NC |
| 14 | VCC | VDD |

**U10 74LVC30A**

| 핀 | 이름 | 넷 |
|---|---|---|
| 1 | A | /D6 |
| 2 | B | /D7 |
| 3 | C | /D8 |
| 4 | D | /D9 |
| 5 | E | /D10 |
| 6 | F | VDD |
| 7 | GND | GND |
| 8 | Y | WIN2 |
| 9 | NC9 | NC |
| 10 | NC10 | NC |
| 11 | G | VDD |
| 12 | H | VDD |
| 13 | NC13 | NC |
| 14 | VCC | VDD |

**U11 74LVC30A**

| 핀 | 이름 | 넷 |
|---|---|---|
| 1 | A | /C0 |
| 2 | B | /C1 |
| 3 | C | C2 |
| 4 | D | C3 |
| 5 | E | C4 |
| 6 | F | C5 |
| 7 | GND | GND |
| 8 | Y | CLR_N |
| 9 | NC9 | NC |
| 10 | NC10 | NC |
| 11 | G | CLK |
| 12 | H | VDD |
| 13 | NC13 | NC |
| 14 | VCC | VDD |

**U12 74LVC08A**

| 핀 | 이름 | 넷 |
|---|---|---|
| 1 | 1A | CLR_N |
| 2 | 1B | POR_N |
| 3 | 1Y | /RD |
| 4 | 2A | CLK |
| 5 | 2B | RUN |
| 6 | 2Y | GATE |
| 7 | GND | GND |
| 8 | 3Y | X1 |
| 9 | 3A | MATCH1 |
| 10 | 3B | GATE |
| 11 | 4Y | X2 |
| 12 | 4A | MATCH2 |
| 13 | 4B | GATE |
| 14 | VCC | VDD |

**U13 LPV7215**

| 핀 | 이름 | 넷 |
|---|---|---|
| 1 | OUT | MATCH1 |
| 2 | V- | GND |
| 3 | IN+ | SC1 |
| 4 | IN- | VTH1 |
| 5 | V+ | VDD |

**U14 LPV7215**

| 핀 | 이름 | 넷 |
|---|---|---|
| 1 | OUT | MATCH2 |
| 2 | V- | GND |
| 3 | IN+ | SC2 |
| 4 | IN- | VTH2 |
| 5 | V+ | VDD |

**U15 74LVC74A**

| 핀 | 이름 | 넷 |
|---|---|---|
| 1 | 1/RD | /RD |
| 2 | 1D | GND |
| 3 | 1CP | GND |
| 4 | 1/SD | /SD_RUN |
| 5 | 1Q | RUN |
| 6 | 1/Q | RUN_N |
| 7 | GND | GND |
| 8 | 2/Q | NC |
| 9 | 2Q | PASS1 |
| 10 | 2/SD | /S1 |
| 11 | 2CP | GND |
| 12 | 2D | GND |
| 13 | 2/RD | /RD |
| 14 | VCC | VDD |

**U16 74LVC74A**

| 핀 | 이름 | 넷 |
|---|---|---|
| 1 | 1/RD | /RD |
| 2 | 1D | GND |
| 3 | 1CP | GND |
| 4 | 1/SD | /S2 |
| 5 | 1Q | PASS2 |
| 6 | 1/Q | NC |
| 7 | GND | GND |
| 8 | 2/Q | NC |
| 9 | 2Q | NC |
| 10 | 2/SD | VDD |
| 11 | 2CP | GND |
| 12 | 2D | GND |
| 13 | 2/RD | VDD |
| 14 | VCC | VDD |

**U17 74LVC08A**

| 핀 | 이름 | 넷 |
|---|---|---|
| 1 | 1A | PASS1 |
| 2 | 1B | PASS2 |
| 3 | 1Y | WAKE |
| 4 | 2A | GND |
| 5 | 2B | GND |
| 6 | 2Y | NC |
| 7 | GND | GND |
| 8 | 3Y | NC |
| 9 | 3A | GND |
| 10 | 3B | GND |
| 11 | 4Y | NC |
| 12 | 4A | GND |
| 13 | 4B | GND |
| 14 | VCC | VDD |

## 넷별 연결

| 넷 | 연결 |
|---|---|
| /C0 | U3.10(5Y), U11.1(A) |
| /C1 | U3.12(6Y), U11.2(B) |
| /C5 | U3.8(4Y), U7.6(E3) |
| /CH0 | U2.2(1Y), U4.1(A), R4.1, R14.1 |
| /CH1 | U2.4(2Y), U4.2(B), R5.1, R15.1 |
| /CH2 | U2.6(3Y), U4.3(C), R6.1, R16.1 |
| /CH3 | U2.8(4Y), U4.4(D), R7.1, R17.1 |
| /CH4 | U2.10(5Y), U4.5(E), R8.1, R18.1 |
| /CH5 | U2.12(6Y), U4.6(F), R9.1 |
| /CH6 | U3.2(1Y), U4.11(G) |
| /CH7 | U3.4(2Y), U4.12(H), R11.1 |
| /D0 | U7.15(/Y0), U9.1(A) |
| /D1 | U7.14(/Y1), U9.2(B) |
| /D10 | U8.13(/Y2), U10.5(E) |
| /D2 | U7.13(/Y2), U9.3(C) |
| /D3 | U7.12(/Y3), U9.4(D) |
| /D4 | U7.11(/Y4), U9.5(E) |
| /D6 | U7.9(/Y6), U10.1(A) |
| /D7 | U7.7(/Y7), U10.2(B) |
| /D8 | U8.15(/Y0), U10.3(C) |
| /D9 | U8.14(/Y1), U10.4(D) |
| /RD | U12.3(1Y), U15.1(1/RD), U15.13(2/RD), U16.1(1/RD) |
| /S1 | U5.6(2Y), U15.10(2/SD) |
| /S2 | U5.8(3Y), U16.4(1/SD) |
| /SD_RUN | U5.3(1Y), U15.4(1/SD) |
| C0 | U3.11(5A), U6.3(1Q0), U7.1(A0), U8.1(A0) |
| C1 | U3.13(6A), U6.4(1Q1), U7.2(A1), U8.2(A1) |
| C2 | U6.5(1Q2), U7.3(A2), U8.3(A2), U11.3(C) |
| C3 | U6.6(1Q3), U6.13(2CP), U7.4(/E1), U8.6(E3), U11.4(D) |
| C4 | U6.11(2Q0), U7.5(/E2), U8.4(/E1), U11.5(E) |
| C5 | U3.9(4A), U6.10(2Q1), U8.5(/E2), U11.6(F) |
| CH0 | U2.1(1A), J1.ICD(CH0) |
| CH1 | U2.3(2A), J1.ICD(CH1) |
| CH2 | U2.5(3A), J1.ICD(CH2) |
| CH3 | U2.9(4A), J1.ICD(CH3) |
| CH4 | U2.11(5A), J1.ICD(CH4) |
| CH5 | U2.13(6A), J1.ICD(CH5) |
| CH6 | U3.1(1A), R10.1, R19.1, J1.ICD(CH6) |
| CH7 | U3.3(2A), J1.ICD(CH7) |
| CLK | U1.14(Q7), U6.1(1CP), U11.11(G), U12.4(2A) |
| CLR_N | U11.8(Y), U12.1(1A) |
| GATE | U12.6(2Y), U12.10(3B), U12.13(4B) |
| GND | Q1.2(E), U1.8(GND), U1.12(MR), U2.7(GND), U3.7(GND), U4.7(GND), U5.7(GND), U5.12(4A), U5.13(4B), U6.7(GND), U7.8(GND), U8.8(GND), U9.7(GND), U10.7(GND), U11.7(GND), U12.7(GND), U13.2(V-), U14.2(V-), U15.2(1D), U15.3(1CP), U15.7(GND), U15.11(2CP), U15.12(2D), U16.2(1D), U16.3(1CP), U16.7(GND), U16.11(2CP), U16.12(2D), U17.4(2A), U17.5(2B), U17.7(GND), U17.9(3A), U17.10(3B), U17.12(4A), U17.13(4B), R3.2, R13.2, R21.2, J1.ICD(GND), (+ 디커플링 C 20개) |
| LED_A | D1.1, R23.2 |
| LED_K | Q1.3(C), D1.2 |
| MATCH1 | U12.9(3A), U13.1(OUT) |
| MATCH2 | U12.12(4A), U14.1(OUT) |
| OSC_IN | U1.11(RS), Y1.1, R1.1, C1.1 |
| OSC_OUT | U1.10(RTC), R1.2, R2.1 |
| OSC_X | Y1.2, R2.2, C2.1 |
| PASS1 | U15.9(2Q), U17.1(1A) |
| PASS2 | U16.5(1Q), U17.2(1B) |
| POR_N | U3.6(3Y), U5.2(1B), U12.2(1B) |
| POR_RC | U3.5(3A), C3.2, R3.1 |
| Q_B | Q1.1(B), R22.2 |
| RUN | U12.5(2B), U15.5(1Q) |
| RUN_N | U6.2(1MR), U6.12(2MR), U15.6(1/Q) |
| SC1 | U13.3(IN+), R4.2, R5.2, R6.2, R7.2, R8.2, R9.2, R10.2, R11.2 |
| SC2 | U14.3(IN+), R14.2, R15.2, R16.2, R17.2, R18.2, R19.2 |
| START | U4.8(Y), U5.1(1A) |
| VDD | U1.16(VCC), U2.14(VCC), U3.14(VCC), U4.14(VCC), U5.14(VCC), U6.14(VCC), U7.16(VCC), U8.16(VCC), U9.6(F), U9.11(G), U9.12(H), U9.14(VCC), U10.6(F), U10.11(G), U10.12(H), U10.14(VCC), U11.12(H), U11.14(VCC), U12.14(VCC), U13.5(V+), U14.5(V+), U15.14(VCC), U16.10(2/SD), U16.13(2/RD), U16.14(VCC), U17.14(VCC), R12.1, R20.1, J1.ICD(VDD), (+ 디커플링 C 19개) |
| VLED | R23.1, J1.미정(VLED) |
| VTH1 | U13.4(IN-), R12.2, R13.1 |
| VTH2 | U14.4(IN-), R20.2, R21.1 |
| WAKE | U17.3(1Y), R22.1 |
| WIN1 | U5.5(2B), U9.8(Y) |
| WIN2 | U5.10(3B), U10.8(Y) |
| X1 | U5.4(2A), U12.8(3Y) |
| X2 | U5.9(3A), U12.11(4Y) |
