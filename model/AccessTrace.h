#ifndef SCOPE_MODEL_ACCESS_TRACE_H_
#define SCOPE_MODEL_ACCESS_TRACE_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace scope_model {

enum class Operation { kLoad, kStore };

struct Access {
    Operation operation;
    std::uint64_t address;
    std::size_t size_bytes;
    std::size_t phase = 0;
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
    std::size_t transaction_bytes = 0;
    std::size_t cache_line_bytes = 128;
    std::size_t working_set_stride_bytes = 64;
    std::size_t bytes_per_element = 2;
    std::uint64_t seed = 7;
    std::string trace_kind = "legacy_synthetic";
    std::uint64_t group_m = 8;
};

struct TracePhase {
    std::string name;
    std::uint64_t begin = 0, end = 0, loads = 0, stores = 0;
};

struct TraceTensor {
    std::string name;
    std::uint64_t base, bytes, rows, columns;
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
    const std::vector<TracePhase>& phases() const { return phases_; }
    const std::vector<TraceTensor>& tensors() const { return tensors_; }
    std::string tensor_name(std::uint64_t address) const;

  private:
    std::uint64_t permute(std::uint64_t value, std::uint64_t modulus,
                          std::uint64_t salt) const;

    TraceConfig config_;
    std::uint64_t cycle_accesses_ = 0;
    std::uint64_t sampled_tensor_bytes_ = 0;
    std::uint64_t analytical_loads_ = 0;
    std::uint64_t analytical_stores_ = 0;
    std::uint64_t analytical_working_set_bytes_ = 0;
    struct Segment {
        std::uint64_t begin, end, base, stride, transactions_per_row;
        Operation operation;
        std::size_t phase;
    };
    void build_tiled();
    void rectangle(Operation op, std::uint64_t base, std::uint64_t stride,
                   std::uint64_t rows, std::uint64_t width_bytes);
    void gemm(std::uint64_t a, std::uint64_t weight, std::uint64_t output,
              std::uint64_t m, std::uint64_t n, std::uint64_t k);
    std::vector<Segment> segments_;
    std::vector<TracePhase> phases_;
    std::vector<TraceTensor> tensors_;
    mutable std::size_t last_segment_ = 0;
};

}  // namespace scope_model

#endif  // SCOPE_MODEL_ACCESS_TRACE_H_
