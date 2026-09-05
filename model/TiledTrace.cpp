#include "AccessTrace.h"

#include <algorithm>
#include <stdexcept>

namespace scope_model {

void AccessTrace::rectangle(Operation op, std::uint64_t base, std::uint64_t stride,
                            std::uint64_t rows, std::uint64_t width_bytes) {
    if (!rows || !width_bytes) return;
    if (stride == width_bytes) {
        width_bytes *= rows;
        rows = 1;
    }
    const auto transaction = config_.transaction_bytes;
    if (rows > 1 && stride % transaction) {
        for (std::uint64_t row = 0; row < rows; ++row)
            rectangle(op, base + row * stride, stride, 1, width_bytes);
        return;
    }
    const auto per_row = (base % transaction + width_bytes + transaction - 1) / transaction;
    const auto count = rows * per_row;
    const auto isa_count = rows * ((width_bytes + config_.access_bytes - 1) / config_.access_bytes);
    segments_.push_back({cycle_accesses_, cycle_accesses_ + count,
                         base - base % transaction, stride, per_row, op, phases_.size() - 1});
    cycle_accesses_ += count;
    auto& phase = phases_.back();
    phase.end = cycle_accesses_;
    if (op == Operation::kLoad) {
        phase.loads += count;
        analytical_loads_ += isa_count;
    } else {
        phase.stores += count;
        analytical_stores_ += isa_count;
    }
}

void AccessTrace::gemm(std::uint64_t a, std::uint64_t weight, std::uint64_t output,
                      std::uint64_t m, std::uint64_t n, std::uint64_t k) {
    const auto b = config_.bytes_per_element;
    const auto m_tiles = (m + config_.tile_m - 1) / config_.tile_m;
    // Grouped CTA order: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
    // PyTorch Linear weights are stored [N,K]; each CTA stages its operands once per K tile.
    for (std::uint64_t group = 0; group < m_tiles; group += config_.group_m) {
        for (std::uint64_t col = 0; col < n; col += config_.tile_n) {
            for (std::uint64_t mt = group; mt < std::min(m_tiles, group + config_.group_m); ++mt) {
                const auto row = mt * config_.tile_m;
                const auto rows = std::min(config_.tile_m, m - row);
                const auto cols = std::min(config_.tile_n, n - col);
                for (std::uint64_t kk = 0; kk < k; kk += config_.tile_k) {
                    const auto inner = std::min(config_.tile_k, k - kk);
                    rectangle(Operation::kLoad, a + (row * k + kk) * b, k * b, rows, inner * b);
                    rectangle(Operation::kLoad, weight + (col * k + kk) * b, k * b, cols, inner * b);
                }
                // Accumulation stays in registers/shared memory; only the final BF16 tile is stored.
                rectangle(Operation::kStore, output + (row * n + col) * b, n * b, rows, cols * b);
            }
        }
    }
}

void AccessTrace::build_tiled() {
    const auto s = config_.sequence_tokens, h = config_.hidden_size;
    const auto intermediate = config_.intermediate_size;
    const auto b = config_.bytes_per_element;
    std::uint64_t next_address = 0x100000000ULL;
    auto tensor = [&](const std::string& name, std::uint64_t rows, std::uint64_t cols) {
        const auto bytes = rows * cols * b;
        const auto base = next_address;
        tensors_.push_back({name, base, bytes, rows, cols});
        next_address += ((bytes + 4095) / 4096) * 4096;
        analytical_working_set_bytes_ += bytes;
        return base;
    };
    auto phase = [&](const std::string& name) {
        phases_.push_back({name, cycle_accesses_, cycle_accesses_, 0, 0});
    };
    const auto x = tensor("input", s, h);
    if (config_.operator_name == "attention") {
        if (!config_.attention_heads || !config_.head_dimension ||
            config_.attention_heads * config_.head_dimension != h)
            throw std::invalid_argument("attention_heads * head_dimension must equal hidden_size");
        const auto wq = tensor("Wq", h, h), wk = tensor("Wk", h, h);
        const auto wv = tensor("Wv", h, h), wo = tensor("Wo", h, h);
        const auto q = tensor("Q", s, h), k = tensor("K", s, h), v = tensor("V", s, h);
        const auto attention = tensor("attention_output", s, h), output = tensor("output", s, h);
        phase("Q projection"); gemm(x, wq, q, s, h, h);
        phase("K projection"); gemm(x, wk, k, s, h, h);
        phase("V projection"); gemm(x, wv, v, s, h, h);
        phase("RoPE Q/K in place");
        rectangle(Operation::kLoad, q, h*b, s, h*b);
        rectangle(Operation::kLoad, k, h*b, s, h*b);
        rectangle(Operation::kStore, q, h*b, s, h*b);
        rectangle(Operation::kStore, k, h*b, s, h*b);
        phase("causal FlashAttention QK-softmax-PV");
        const auto d = config_.head_dimension;
        for (std::uint64_t head = 0; head < config_.attention_heads; ++head) {
            for (std::uint64_t row = 0; row < s; row += config_.tile_m) {
                const auto rows = std::min(config_.tile_m, s - row);
                rectangle(Operation::kLoad, q + (row*h + head*d)*b, h*b, rows, d*b);
                for (std::uint64_t key = 0; key < row + rows; key += config_.tile_n) {
                    const auto keys = std::min(config_.tile_n, row + rows - key);
                    rectangle(Operation::kLoad, k + (key*h + head*d)*b, h*b, keys, d*b);
                    rectangle(Operation::kLoad, v + (key*h + head*d)*b, h*b, keys, d*b);
                }
                // Scores, online softmax statistics and partial output never spill to global memory.
                rectangle(Operation::kStore, attention + (row*h + head*d)*b, h*b, rows, d*b);
            }
        }
        phase("output projection"); gemm(attention, wo, output, s, h, h);
    } else {
        const auto wg = tensor("Wgate", intermediate, h), wu = tensor("Wup", intermediate, h);
        const auto wd = tensor("Wdown", h, intermediate);
        const auto gate = tensor("gate", s, intermediate), up = tensor("up", s, intermediate);
        const auto activation = tensor("SwiGLU", s, intermediate), output = tensor("output", s, h);
        phase("gate projection"); gemm(x, wg, gate, s, intermediate, h);
        phase("up projection"); gemm(x, wu, up, s, intermediate, h);
        phase("SwiGLU SiLU-times-up");
        // An unfused epilogue explicitly reads gate/up and materializes the activation for down projection.
        for (std::uint64_t row = 0; row < s; row += config_.tile_m) {
            const auto rows = std::min(config_.tile_m, s-row);
            rectangle(Operation::kLoad, gate + row*intermediate*b, intermediate*b, rows, intermediate*b);
            rectangle(Operation::kLoad, up + row*intermediate*b, intermediate*b, rows, intermediate*b);
            rectangle(Operation::kStore, activation + row*intermediate*b, intermediate*b, rows, intermediate*b);
        }
        phase("down projection"); gemm(activation, wd, output, s, h, intermediate);
    }
    sampled_tensor_bytes_ = analytical_working_set_bytes_;
}

std::string AccessTrace::tensor_name(std::uint64_t address) const {
    for (const auto& tensor : tensors_)
        if (address >= tensor.base && address < tensor.base + tensor.bytes) return tensor.name;
    return "legacy synthetic address";
}

}  // namespace scope_model
