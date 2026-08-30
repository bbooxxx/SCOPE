#ifndef SCOPE_MODEL_ACCESS_TRACE_H_
#define SCOPE_MODEL_ACCESS_TRACE_H_

#include <cstddef>
#include <cstdint>
#include <string>

namespace scope_model {

enum class Operation { kLoad, kStore };

struct Access {
    Operation operation;
    std::uint64_t address;
    std::size_t size_bytes;
};

struct TraceConfig {
    std::string operator_name;
    std::uint64_t sequence_tokens = 295;
    std::uint64_t hidden_size = 4096;
    std::uint64_t attention_heads = 32;
    std::uint64_t head_dimension = 128;
    std::uint64_t intermediate_size = 11008;
    std::uint64_t tile_m = 16;
    std::uint64_t tile_n = 64;
    std::uint64_t tile_k = 32;
    std::uint64_t sampled_working_set_bytes = 0;
    std::uint64_t cycle_access_cap = 0;
    std::size_t access_bytes = 16;
    std::size_t working_set_stride_bytes = 64;
    std::size_t bytes_per_element = 2;
    std::uint64_t seed = 7;
};

class AccessTrace {
  public:
    explicit AccessTrace(TraceConfig config);

    Access at(std::uint64_t ordinal) const;
    const TraceConfig& config() const { return config_; }
    std::uint64_t cycle_accesses() const { return cycle_accesses_; }
    std::uint64_t sampled_tensor_bytes() const { return sampled_tensor_bytes_; }
    std::uint64_t analytical_loads() const { return analytical_loads_; }
    std::uint64_t analytical_stores() const { return analytical_stores_; }
    std::uint64_t analytical_working_set_bytes() const {
        return analytical_working_set_bytes_;
    }
    double analytical_read_fraction() const;

  private:
    std::uint64_t permute(std::uint64_t value, std::uint64_t modulus,
                          std::uint64_t salt) const;

    TraceConfig config_;
    std::uint64_t cycle_accesses_ = 0;
    std::uint64_t sampled_tensor_bytes_ = 0;
    std::uint64_t analytical_loads_ = 0;
    std::uint64_t analytical_stores_ = 0;
    std::uint64_t analytical_working_set_bytes_ = 0;
};

}  // namespace scope_model

#endif  // SCOPE_MODEL_ACCESS_TRACE_H_
