#include "schedules.cuh"

void register_huginn_anchor_3_g16() {
    register_schedules<
        55, 96, ReferenceMode::Anchor, 3,
        RecordSchedule<2, 16, 2, 16, 2, 16, 2, 16>,
        RecordSchedule<4, 32, 2, 16, 2, 16, 2, 16>,
        RecordSchedule<2, 16, 2, 16, 2, 16, 4, 32>>();
}
