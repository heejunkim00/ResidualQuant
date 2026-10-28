#include <pybind11/stl.h>
#include <torch/extension.h>

#include "common/api.h"

void register_ouro_anchor_0_g16();
void register_ouro_anchor_0_g32();
void register_ouro_anchor_3_g16();
void register_ouro_anchor_3_g32();
void register_ouro_adjacent_0_g16();
void register_ouro_adjacent_0_g32();
void register_huginn_anchor_0_g16();
void register_huginn_anchor_0_g32();
void register_huginn_anchor_3_g16();
void register_huginn_anchor_3_g32();
void register_huginn_adjacent_0_g16();
void register_huginn_adjacent_0_g32();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "attention", &attention, pybind11::arg("query"), pybind11::arg("key"),
        pybind11::arg("value"), pybind11::arg("kv_cache"),
        pybind11::arg("output"), pybind11::arg("query_start_loc"),
        pybind11::arg("seq_lens"), pybind11::arg("block_table"),
        pybind11::arg("key_rotation"), pybind11::arg("value_rotation"),
        pybind11::arg("key_mean"), pybind11::arg("records"),
        pybind11::arg("reference_mode"), pybind11::arg("anchor_index"),
        pybind11::arg("loop_index")
    );

    m.def(
        "quantize", &quantize, pybind11::arg("key_value_pairs"),
        pybind11::arg("kv_cache"), pybind11::arg("slot_mapping"),
        pybind11::arg("key_rotation"), pybind11::arg("value_rotation"),
        pybind11::arg("key_mean"), pybind11::arg("records"),
        pybind11::arg("reference_mode"), pybind11::arg("anchor_index"),
        pybind11::arg("coefficient_method")
    );

    register_ouro_anchor_0_g16();
    register_ouro_anchor_0_g32();
    register_ouro_anchor_3_g16();
    register_ouro_anchor_3_g32();
    register_ouro_adjacent_0_g16();
    register_ouro_adjacent_0_g32();
    register_huginn_anchor_0_g16();
    register_huginn_anchor_0_g32();
    register_huginn_anchor_3_g16();
    register_huginn_anchor_3_g32();
    register_huginn_adjacent_0_g16();
    register_huginn_adjacent_0_g32();
}
