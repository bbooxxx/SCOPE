#include "AccessTrace.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace scope_model {
namespace {

constexpr std::uint64_t kActivationBase = 0x100000000ULL;
constexpr std::uint64_t kWeightABase = 0x200000000ULL;
constexpr std::uint64_t kWeightBBase = 0x400000000ULL;
constexpr std::uint64_t kOutputBase = 0x600000000ULL;
constexpr std::uint64_t kStateBase = 0x700000000ULL;
constexpr std::uint64_t kColdStreamBase = 0x800000000ULL;

std::uint64_t mix(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

}  // namespace

AccessTrace::AccessTrace(TraceConfig config) : config_(std::move(config)) {
    if (config_.operator_name != "attention" && config_.operator_name != "ffn") {
        throw std::invalid_argument("operator must be attention or ffn");
    }
    if (config_.access_bytes == 0 || config_.working_set_stride_bytes == 0 ||
        config_.working_set_stride_bytes % config_.access_bytes != 0 ||
        config_.sampled_working_set_bytes < 4096) {
        throw std::invalid_argument("invalid access or sampled working-set size");
    }
    if (std::abs(config_.read_fraction - 0.75) > 1e-12) {
        throw std::invalid_argument("v4 trace currently requires read_fraction=0.75");
    }

    const std::uint64_t stream_bytes = config_.sampled_working_set_bytes / 2;
    const std::uint64_t stream_accesses =
        stream_bytes / config_.working_set_stride_bytes;
    if (stream_accesses == 0) {
        throw std::invalid_argument("sampled working set is too small");
    }
    cycle_accesses_ = stream_accesses * 4;
    sampled_tensor_bytes_ = 2 * stream_bytes;
}

std::uint64_t AccessTrace::permute(std::uint64_t value, std::uint64_t modulus,
                                   std::uint64_t salt) const {
    return mix(value ^ mix(config_.seed + salt)) % modulus;
}

Access AccessTrace::at(std::uint64_t ordinal) const {
    const std::uint64_t stream_bytes = config_.sampled_working_set_bytes / 2;
    const std::uint64_t stream_accesses =
        stream_bytes / config_.working_set_stride_bytes;
    const std::uint64_t local = (ordinal % cycle_accesses_) / 4;
    const std::uint64_t epoch = ordinal / cycle_accesses_;
    const std::uint64_t lane = ordinal % 4;
    const std::uint64_t offset_a =
        permute(local, stream_accesses, 0xA11U) *
            config_.working_set_stride_bytes +
        permute(local, config_.working_set_stride_bytes / config_.access_bytes,
                0xA12U) * config_.access_bytes;
    const std::uint64_t offset_b =
        permute(local, stream_accesses, 0xB22U) *
            config_.working_set_stride_bytes +
        permute(local, config_.working_set_stride_bytes / config_.access_bytes,
                0xB23U) * config_.access_bytes;

    const std::uint64_t activation_bytes = std::max<std::uint64_t>(
        64 * 1024,
        config_.sequence_tokens * config_.hidden_size * 2);
    const std::uint64_t output_bytes = std::max<std::uint64_t>(
        64 * 1024,
        config_.sequence_tokens *
            (config_.operator_name == "attention" ? config_.hidden_size
                                                   : config_.intermediate_size) *
            2);
    const std::uint64_t activation_slots = activation_bytes / config_.access_bytes;
    const std::uint64_t output_slots = output_bytes / config_.access_bytes;
    const std::uint64_t tile_reuse = std::max<std::uint64_t>(
        1, config_.tile_m * config_.tile_n / config_.tile_k);

    if (lane == 0) {
        const std::uint64_t tile_slot = (local / tile_reuse) % activation_slots;
        return {Operation::kLoad,
                kActivationBase + tile_slot * config_.access_bytes,
                config_.access_bytes};
    }
    if (lane == 1) {
        if (local % 16 == 0) {
            return {Operation::kLoad,
                    kColdStreamBase + epoch * stream_bytes + offset_a,
                    config_.access_bytes};
        }
        return {Operation::kLoad, kWeightABase + offset_a, config_.access_bytes};
    }
    if (lane == 2) {
        const std::uint64_t base = config_.operator_name == "attention"
                                       ? kWeightBBase
                                       : kWeightBBase + stream_bytes;
        return {Operation::kLoad, base + offset_b, config_.access_bytes};
    }

    const std::uint64_t tile_slot = (local / tile_reuse) % output_slots;
    const std::uint64_t base = config_.operator_name == "attention"
                                   ? kOutputBase
                                   : kStateBase;
    return {Operation::kStore, base + tile_slot * config_.access_bytes,
            config_.access_bytes};
}

}  // namespace scope_model
