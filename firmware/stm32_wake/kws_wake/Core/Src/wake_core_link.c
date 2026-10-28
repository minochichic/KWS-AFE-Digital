/*
 * firmware/stm32_wake/core 를 이 프로젝트에 넣는다.
 *
 * core/ 는 Mac 테스트와 같은 파일을 그대로 써야 하므로 복사하지 않고 여기서 포함한다.
 * .cproject 에 링크 폴더를 거는 대신 이렇게 하면 CubeMX 재생성에도 안 깨진다.
 */
#include "../../../core/wake_core.c"
#include "../../../core/link.c"
