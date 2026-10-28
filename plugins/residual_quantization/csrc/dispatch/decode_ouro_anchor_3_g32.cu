#include "schedules.cuh"

void register_ouro_anchor_3_g32() {
    register_schedules<
        16, 128, ReferenceMode::Anchor, 3,
        RecordSchedule<2, 32, 2, 32, 2, 32, 2, 32>,
        RecordSchedule<4, 32, 4, 32, 4, 32, 4, 32>,
        RecordSchedule<4, 32, 2, 32, 2, 32, 2, 32>,
        RecordSchedule<2, 32, 2, 32, 2, 32, 4, 32>>();
}
