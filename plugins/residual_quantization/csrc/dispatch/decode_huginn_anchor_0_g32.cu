#include "schedules.cuh"

void register_huginn_anchor_0_g32() {
    register_schedules<
        55, 96, ReferenceMode::Anchor, 0,
        RecordSchedule<2, 32, 2, 32, 2, 32, 2, 32>,
        RecordSchedule<4, 32, 4, 32, 4, 32, 4, 32>,
        RecordSchedule<4, 32, 2, 32, 2, 32, 2, 32>,
        RecordSchedule<2, 32, 2, 32, 2, 32, 4, 32>, RecordSchedule<2, 32>,
        RecordSchedule<4, 32>>();
}
