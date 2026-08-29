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
    std::uint64_t sampled_working_set_bytes = 24ULL * 1024ULL * 1024ULL;
    std::size_t access_bytes = 16;
    std::size_t working_set_stride_bytes = 64;
    double read_fraction = 0.75;
    std::uint64_t seed = 7;
};

class AccessTrace {
  public:
    explicit AccessTrace(TraceConfig config);

    Access at(std::uint64_t ordinal) const;
    const TraceConfig& config() const { return config_; }
    std::uint64_t cycle_accesses() const { return cycle_accesses_; }
    std::uint64_t sampled_tensor_bytes() const { return sampled_tensor_bytes_; }

  private:
    std::uint64_t permute(std::uint64_t value, std::uint64_t modulus,
                          std::uint64_t salt) const;

    TraceConfig config_;
    std::uint64_t cycle_accesses_ = 0;
    std::uint64_t sampled_tensor_bytes_ = 0;
};

}  // namespace scope_model

#endif  // SCOPE_MODEL_ACCESS_TRACE_H_
