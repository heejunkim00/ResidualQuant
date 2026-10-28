#include "schedules.cuh"

void register_huginn_adjacent_0_g16() {
    register_schedules<
        55, 96, ReferenceMode::Adjacent, 0,
        RecordSchedule<2, 16, 2, 16, 2, 16, 2, 16>,
        RecordSchedule<4, 32, 2, 16, 2, 16, 2, 16>,
        RecordSchedule<2, 16, 2, 16, 2, 16, 4, 32>, RecordSchedule<2, 16>>();
}
