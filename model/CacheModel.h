#ifndef SCOPE_MODEL_CACHE_MODEL_H_
#define SCOPE_MODEL_CACHE_MODEL_H_

#include "AccessTrace.h"

#include <cstddef>
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace scope_model {

struct CacheConfig {
    std::uint64_t capacity_bytes;
    std::size_t associativity;
    std::size_t line_bytes;
    std::string policy;
    std::string indexing = "modulo";
};

struct CacheStats {
    std::uint64_t accesses = 0;
    std::uint64_t hits = 0;
    std::uint64_t writebacks = 0;
    std::uint64_t compulsory_misses = 0;
    std::uint64_t noncompulsory_misses = 0;
    std::uint64_t load_accesses = 0;
    std::uint64_t load_hits = 0;
    std::uint64_t store_hits = 0;
    std::uint64_t fills = 0;
};

struct RepresentativeAccess {
    bool valid = false;
    Operation operation = Operation::kLoad;
    std::uint64_t address = 0;
    std::size_t size_bytes = 0;
    std::string hit_level;
    std::size_t phase = 0;
};

struct PhaseStats {
    std::uint64_t requests = 0, loads = 0;
    std::array<std::uint64_t, 4> hit_counts{};
};

struct SimulationResult {
    std::vector<CacheStats> levels;
    std::uint64_t measured_requests = 0;
    std::uint64_t measured_loads = 0;
    std::uint64_t offchip_writebacks = 0;
    RepresentativeAccess representative;
    std::uint64_t offchip_loads = 0, offchip_stores = 0;
    std::vector<PhaseStats> phases;
};

class CacheHierarchy {
  public:
    CacheHierarchy(std::vector<CacheConfig> configs, std::uint64_t seed);
    SimulationResult run(const AccessTrace& trace, std::uint64_t warmup_accesses,
                         std::uint64_t sample_accesses);

  private:
    struct Entry {
        std::uint64_t line;
        bool dirty;
        std::uint64_t inserted_at;
        std::uint64_t touched_at;
    };

    class Cache {
      public:
        Cache(CacheConfig config, std::uint64_t seed);
        bool probe(std::uint64_t line, bool mark_dirty);
        bool insert(std::uint64_t line, bool dirty, Entry* evicted);
        bool first_reference(std::uint64_t line);

      private:
        std::size_t victim(const std::vector<Entry>& entries);
        std::uint64_t set_index(std::uint64_t line) const;

        CacheConfig config_;
        std::uint64_t num_sets_;
        unsigned index_bits_ = 0;
        std::uint64_t clock_ = 0;
        std::uint64_t random_state_;
        std::unordered_map<std::uint64_t, std::vector<Entry>> sets_;
        std::unordered_set<std::uint64_t> seen_lines_;
    };

    void writeback(std::size_t level, std::uint64_t line, bool record,
                   SimulationResult* result);
    void process(const Access& access, bool record, bool capture,
                 SimulationResult* result);

    std::vector<Cache> caches_;
    std::size_t line_bytes_ = 0;
};

}  // namespace scope_model

#endif  // SCOPE_MODEL_CACHE_MODEL_H_
