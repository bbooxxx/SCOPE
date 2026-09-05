#include "AccessTrace.h"

#include <algorithm>
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
    if (!config_.sequence_tokens || !config_.hidden_size || !config_.intermediate_size ||
        !config_.tile_m || !config_.tile_n || !config_.tile_k || !config_.group_m) {
        throw std::invalid_argument("tensor dimensions and tile sizes must be positive");
    }
    if (config_.transaction_bytes == 0) {
        config_.transaction_bytes = config_.access_bytes;
    }
    if (config_.access_bytes == 0 || config_.bytes_per_element == 0 ||
        config_.transaction_bytes < config_.access_bytes ||
        config_.transaction_bytes % config_.access_bytes != 0 ||
        config_.cache_line_bytes == 0 ||
        config_.cache_line_bytes % config_.transaction_bytes != 0 ||
        config_.working_set_stride_bytes == 0 ||
        config_.working_set_stride_bytes % config_.access_bytes != 0) {
        throw std::invalid_argument("invalid access geometry");
    }
    if (config_.trace_kind == "tiled") {
        if (config_.sampled_working_set_bytes || config_.cycle_access_cap) {
            throw std::invalid_argument("tiled trace uses the complete tensor domain; use sample-accesses to bound a run");
        }
        build_tiled();
        return;
    }
    if (config_.trace_kind != "legacy_synthetic") {
        throw std::invalid_argument("trace-kind must be tiled or legacy_synthetic");
    }

    const std::uint64_t elements_per_access = std::max<std::uint64_t>(
        1, config_.access_bytes / config_.bytes_per_element);
    if (config_.operator_name == "attention") {
        const std::uint64_t q_tiles =
            (config_.sequence_tokens + config_.tile_m - 1) / config_.tile_m;
        const std::uint64_t q_elements =
            config_.sequence_tokens * config_.hidden_size;
        const std::uint64_t weight_elements =
            4 * config_.hidden_size * config_.hidden_size;
        analytical_working_set_bytes_ =
            (weight_elements + 6 * q_elements) * config_.bytes_per_element;
        analytical_loads_ =
            (weight_elements + (3 + 2 * q_tiles) * q_elements
             + elements_per_access - 1) /
            elements_per_access;
        analytical_stores_ =
            (5 * q_elements + elements_per_access - 1) / elements_per_access;
    } else {
        const std::uint64_t weight_elements =
            3 * config_.hidden_size * config_.intermediate_size;
        const std::uint64_t activation_elements =
            2 * config_.sequence_tokens * config_.hidden_size +
            2 * config_.sequence_tokens * config_.intermediate_size;
        analytical_working_set_bytes_ =
            (weight_elements + activation_elements) * config_.bytes_per_element;
        const std::uint64_t load_elements =
            2 * config_.sequence_tokens * config_.hidden_size +
            weight_elements +
            2 * config_.sequence_tokens * config_.intermediate_size;
        analytical_loads_ =
            (load_elements + elements_per_access - 1) / elements_per_access;
        const std::uint64_t store_elements =
            2 * config_.sequence_tokens * config_.intermediate_size +
            config_.sequence_tokens * config_.hidden_size;
        analytical_stores_ =
            (store_elements + elements_per_access - 1) / elements_per_access;
    }
    if (config_.sampled_working_set_bytes == 0) {
        config_.sampled_working_set_bytes = analytical_working_set_bytes_;
    }
    if (config_.sampled_working_set_bytes < 4096) {
        throw std::invalid_argument("sampled working set is too small");
    }
    const std::uint64_t stream_bytes = config_.sampled_working_set_bytes / 2;
    const std::uint64_t stream_lines = stream_bytes / config_.cache_line_bytes;
    if (stream_lines == 0) {
        throw std::invalid_argument("sampled working set is too small");
    }
    const std::uint64_t transactions_per_line =
        config_.cache_line_bytes / config_.transaction_bytes;
    cycle_accesses_ = stream_lines * 4 * transactions_per_line;
    if (config_.cycle_access_cap > 0) {
        cycle_accesses_ = std::min(cycle_accesses_, config_.cycle_access_cap);
    }
    sampled_tensor_bytes_ = 2 * stream_bytes;
}

double AccessTrace::analytical_read_fraction() const {
    return static_cast<double>(analytical_loads_) /
           static_cast<double>(analytical_loads_ + analytical_stores_);
}

std::uint64_t AccessTrace::permute(std::uint64_t value, std::uint64_t modulus,
                                   std::uint64_t salt) const {
    return mix(value ^ mix(config_.seed + salt)) % modulus;
}

Access AccessTrace::at(std::uint64_t ordinal) const {
    if (config_.trace_kind == "tiled") {
        ordinal %= cycle_accesses_;
        if (ordinal < segments_[last_segment_].begin ||
            ordinal >= segments_[last_segment_].end) {
            if (last_segment_ + 1 < segments_.size() &&
                ordinal >= segments_[last_segment_ + 1].begin &&
                ordinal < segments_[last_segment_ + 1].end) {
                ++last_segment_;
            } else {
                last_segment_ = std::lower_bound(
                    segments_.begin(), segments_.end(), ordinal,
                    [](const Segment& segment, std::uint64_t value) {
                        return segment.end <= value;
                    }) - segments_.begin();
            }
        }
        const auto& segment = segments_[last_segment_];
        const auto offset = ordinal - segment.begin;
        return {segment.operation,
                segment.base + (offset / segment.transactions_per_row) * segment.stride
                    + (offset % segment.transactions_per_row) * config_.transaction_bytes,
                config_.transaction_bytes, segment.phase};
    }
    const std::uint64_t stream_bytes = config_.sampled_working_set_bytes / 2;
    const std::uint64_t stream_lines = stream_bytes / config_.cache_line_bytes;
    const std::uint64_t cycle_ordinal = ordinal % cycle_accesses_;
    const std::uint64_t epoch = ordinal / cycle_accesses_;
    const std::uint64_t transactions_per_line =
        config_.cache_line_bytes / config_.transaction_bytes;
    const std::uint64_t line_group = cycle_ordinal / transactions_per_line;
    const std::uint64_t transaction_in_line =
        cycle_ordinal % transactions_per_line;
    const std::uint64_t weight_tile_reuse = std::max<std::uint64_t>(
        1, std::min<std::uint64_t>(4, config_.tile_m / 8));
    const std::uint64_t tile_group = line_group / weight_tile_reuse;
    const std::uint64_t cycle_line_groups =
        (cycle_accesses_ + transactions_per_line - 1) / transactions_per_line;
    const std::uint64_t operation_rank =
        permute(line_group, cycle_line_groups, 0x0F0U);
    const std::uint64_t load_quota = std::min(
        cycle_line_groups,
        static_cast<std::uint64_t>(
            analytical_read_fraction() * static_cast<double>(cycle_line_groups)));
    const bool is_load = operation_rank < load_quota;
    const std::uint64_t lane = permute(tile_group, 3, 0x1A2U);
    const std::uint64_t offset_a =
        permute(tile_group, stream_lines, 0xA11U) * config_.cache_line_bytes +
        transaction_in_line * config_.transaction_bytes;
    const std::uint64_t offset_b =
        permute(tile_group, stream_lines, 0xB22U) * config_.cache_line_bytes +
        transaction_in_line * config_.transaction_bytes;

    const std::uint64_t activation_bytes = std::max<std::uint64_t>(
        64 * 1024,
        config_.sequence_tokens * config_.hidden_size * 2);
    const std::uint64_t output_bytes = std::max<std::uint64_t>(
        64 * 1024,
        config_.sequence_tokens *
            (config_.operator_name == "attention" ? config_.hidden_size
                                                   : config_.intermediate_size) *
            2);
    const std::uint64_t activation_lines =
        activation_bytes / config_.cache_line_bytes;
    const std::uint64_t output_lines = output_bytes / config_.cache_line_bytes;
    const std::uint64_t activation_reuse = config_.operator_name == "attention"
        ? std::max<std::uint64_t>(
              1, (config_.sequence_tokens + config_.tile_n - 1) / config_.tile_n)
        : std::min<std::uint64_t>(
              8, std::max<std::uint64_t>(
                     1, (config_.intermediate_size + config_.tile_n - 1) /
                            config_.tile_n));

    if (!is_load) {
        const std::uint64_t output_line =
            permute(line_group, output_lines, 0x0D0U);
        const std::uint64_t base = config_.operator_name == "attention"
                                       ? kOutputBase
                                       : kStateBase;
        return {Operation::kStore,
                base + output_line * config_.cache_line_bytes +
                    transaction_in_line * config_.transaction_bytes,
                config_.transaction_bytes};
    }
    if (lane == 0) {
        const std::uint64_t activation_line =
            (line_group / activation_reuse) % activation_lines;
        return {Operation::kLoad,
                kActivationBase + activation_line * config_.cache_line_bytes +
                    transaction_in_line * config_.transaction_bytes,
                config_.transaction_bytes};
    }
    if (lane == 1) {
        if (line_group % 64 == 0) {
            return {Operation::kLoad,
                    kColdStreamBase + epoch * stream_bytes + offset_a,
                    config_.transaction_bytes};
        }
        return {Operation::kLoad, kWeightABase + offset_a,
                config_.transaction_bytes};
    }
    const std::uint64_t base = config_.operator_name == "attention"
                                   ? kWeightBBase
                                   : kWeightBBase + stream_bytes;
    return {Operation::kLoad, base + offset_b, config_.transaction_bytes};
}

}  // namespace scope_model
