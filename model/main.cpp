#include "AccessTrace.h"
#include "CacheModel.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, delimiter)) {
        values.push_back(item);
    }
    return values;
}

std::unordered_map<std::string, std::string> arguments(int argc, char** argv) {
    std::unordered_map<std::string, std::string> result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc || std::string(argv[index]).rfind("--", 0) != 0) {
            throw std::invalid_argument("arguments must be --name value pairs");
        }
        result[std::string(argv[index]).substr(2)] = argv[index + 1];
    }
    return result;
}

template <typename T>
T number(const std::unordered_map<std::string, std::string>& args,
         const std::string& name, T fallback) {
    const auto found = args.find(name);
    if (found == args.end()) {
        return fallback;
    }
    std::stringstream stream(found->second);
    T value{};
    stream >> value;
    if (!stream || !stream.eof()) {
        throw std::invalid_argument("invalid numeric argument: --" + name);
    }
    return value;
}

std::string text(const std::unordered_map<std::string, std::string>& args,
                 const std::string& name, const std::string& fallback) {
    const auto found = args.find(name);
    return found == args.end() ? fallback : found->second;
}

void print_array(const std::vector<double>& values) {
    std::cout << "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) std::cout << ",";
        std::cout << values[index];
    }
    std::cout << "]";
}

void print_array(const std::vector<std::uint64_t>& values) {
    std::cout << "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) std::cout << ",";
        std::cout << values[index];
    }
    std::cout << "]";
}

std::string tensor_name(std::uint64_t address) {
    if (address < 0x200000000ULL) return "activation/Q";
    if (address < 0x400000000ULL) return "weight/K";
    if (address < 0x600000000ULL) return "weight/V-or-FFN";
    if (address < 0x700000000ULL) return "output";
    if (address < 0x800000000ULL) return "FFN-state";
    return "cold-stream";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = arguments(argc, argv);
        scope_model::TraceConfig trace_config;
        trace_config.operator_name = text(args, "operator", "attention");
        trace_config.sequence_tokens = number<std::uint64_t>(
            args, "sequence-tokens", trace_config.sequence_tokens);
        trace_config.hidden_size = number<std::uint64_t>(
            args, "hidden-size", trace_config.hidden_size);
        trace_config.attention_heads = number<std::uint64_t>(
            args, "attention-heads", trace_config.attention_heads);
        trace_config.head_dimension = number<std::uint64_t>(
            args, "head-dimension", trace_config.head_dimension);
        trace_config.intermediate_size = number<std::uint64_t>(
            args, "intermediate-size", trace_config.intermediate_size);
        trace_config.tile_m = number<std::uint64_t>(args, "tile-m", trace_config.tile_m);
        trace_config.tile_n = number<std::uint64_t>(args, "tile-n", trace_config.tile_n);
        trace_config.tile_k = number<std::uint64_t>(args, "tile-k", trace_config.tile_k);
        trace_config.sampled_working_set_bytes = number<std::uint64_t>(
            args, "sampled-working-set-bytes",
            trace_config.sampled_working_set_bytes);
        trace_config.cycle_access_cap = number<std::uint64_t>(
            args, "cycle-access-cap", trace_config.cycle_access_cap);
        trace_config.access_bytes = number<std::size_t>(
            args, "access-bytes", trace_config.access_bytes);
        trace_config.working_set_stride_bytes = number<std::size_t>(
            args, "working-set-stride-bytes",
            trace_config.working_set_stride_bytes);
        trace_config.bytes_per_element = number<std::size_t>(
            args, "bytes-per-element", trace_config.bytes_per_element);
        trace_config.seed = number<std::uint64_t>(args, "seed", trace_config.seed);

        const std::size_t line_bytes = number<std::size_t>(args, "line-bytes", 64);
        const auto capacities_text = split(text(args, "capacities", ""), ',');
        const auto associativity_text = split(text(args, "associativities", ""), ',');
        const auto policies = split(text(args, "policies", ""), ',');
        if (capacities_text.size() != 3 || associativity_text.size() != 3 ||
            policies.size() != 3) {
            throw std::invalid_argument(
                "--capacities, --associativities, and --policies need three values");
        }
        std::vector<scope_model::CacheConfig> cache_configs;
        for (std::size_t level = 0; level < 3; ++level) {
            cache_configs.push_back({
                static_cast<std::uint64_t>(std::stoull(capacities_text[level])),
                static_cast<std::size_t>(std::stoull(associativity_text[level])),
                line_bytes,
                policies[level],
            });
        }

        const scope_model::AccessTrace trace(trace_config);
        const std::uint64_t warmup = number<std::uint64_t>(
            args, "warmup-accesses", trace.cycle_accesses());
        const std::uint64_t samples = number<std::uint64_t>(
            args, "sample-accesses", trace.cycle_accesses());
        scope_model::CacheHierarchy hierarchy(cache_configs, trace_config.seed);
        const scope_model::SimulationResult result = hierarchy.run(trace, warmup, samples);

        std::vector<double> hit_rates;
        std::vector<double> writebacks;
        std::vector<std::uint64_t> accesses;
        std::vector<std::uint64_t> hits;
        std::vector<std::uint64_t> compulsory_misses;
        std::vector<std::uint64_t> noncompulsory_misses;
        for (const auto& level : result.levels) {
            hit_rates.push_back(level.accesses
                                    ? static_cast<double>(level.hits) / level.accesses
                                    : 0.0);
            writebacks.push_back(
                static_cast<double>(level.writebacks) / result.measured_requests);
            accesses.push_back(level.accesses);
            hits.push_back(level.hits);
            compulsory_misses.push_back(level.compulsory_misses);
            noncompulsory_misses.push_back(level.noncompulsory_misses);
        }

        std::cout << std::setprecision(15);
        std::cout << "{\"schema_version\":5"
                  << ",\"model\":\"set_associative_trace\""
                  << ",\"operator\":\"" << trace_config.operator_name << "\""
                  << ",\"kernel\":\""
                  << (trace_config.operator_name == "attention"
                          ? "FlashAttention tiled Q/K/V online-softmax"
                          : "tile-based SwiGLU GEMM/GEMV")
                  << "\""
                  << ",\"isa_access_bytes\":" << trace_config.access_bytes
                  << ",\"working_set_stride_bytes\":"
                  << trace_config.working_set_stride_bytes
                  << ",\"bytes_per_element\":" << trace_config.bytes_per_element
                  << ",\"analytical_loads\":" << trace.analytical_loads()
                  << ",\"analytical_stores\":" << trace.analytical_stores()
                  << ",\"analytical_working_set_bytes\":"
                  << trace.analytical_working_set_bytes()
                  << ",\"analytical_read_fraction\":"
                  << trace.analytical_read_fraction()
                  << ",\"observed_read_fraction\":"
                  << static_cast<double>(result.measured_loads) /
                         result.measured_requests
                  << ",\"warmup_accesses\":" << warmup
                  << ",\"sample_accesses\":" << samples
                  << ",\"trace_cycle_accesses\":" << trace.cycle_accesses()
                  << ",\"cycle_access_cap\":" << trace_config.cycle_access_cap
                  << ",\"sampled_tensor_bytes\":" << trace.sampled_tensor_bytes()
                  << ",\"cold_stream_fraction\":0.015625"
                  << ",\"seed\":" << trace_config.seed
                  << ",\"hit_rates\":";
        print_array(hit_rates);
        std::cout << ",\"accesses\":";
        print_array(accesses);
        std::cout << ",\"hits\":";
        print_array(hits);
        std::cout << ",\"compulsory_misses\":";
        print_array(compulsory_misses);
        std::cout << ",\"noncompulsory_misses\":";
        print_array(noncompulsory_misses);
        std::cout << ",\"writebacks_per_request\":";
        print_array(writebacks);
        std::cout << ",\"offchip_writebacks_per_request\":"
                  << static_cast<double>(result.offchip_writebacks) /
                         result.measured_requests
                  << ",\"representative_access\":{\"operation\":\""
                  << (result.representative.operation == scope_model::Operation::kLoad
                          ? "load" : "store")
                  << "\",\"address\":" << result.representative.address
                  << ",\"size_bytes\":" << result.representative.size_bytes
                  << ",\"tensor\":\""
                  << tensor_name(result.representative.address)
                  << "\",\"hit_level\":\""
                  << result.representative.hit_level << "\"}"
                  << ",\"operator_shape\":{\"sequence_tokens\":"
                  << trace_config.sequence_tokens
                  << ",\"hidden_size\":" << trace_config.hidden_size
                  << ",\"attention_heads\":" << trace_config.attention_heads
                  << ",\"head_dimension\":" << trace_config.head_dimension
                  << ",\"intermediate_size\":" << trace_config.intermediate_size
                  << ",\"tile_m\":" << trace_config.tile_m
                  << ",\"tile_n\":" << trace_config.tile_n
                  << ",\"tile_k\":" << trace_config.tile_k << "}}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "scope_model error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
