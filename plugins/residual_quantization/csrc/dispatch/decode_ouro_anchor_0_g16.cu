#include "schedules.cuh"

void register_ouro_anchor_0_g16() {
    register_schedules<
        16, 128, ReferenceMode::Anchor, 0,
        RecordSchedule<2, 16, 2, 16, 2, 16, 2, 16>,
        RecordSchedule<4, 32, 2, 16, 2, 16, 2, 16>,
        RecordSchedule<2, 16, 2, 16, 2, 16, 4, 32>, RecordSchedule<2, 16>>();
}
