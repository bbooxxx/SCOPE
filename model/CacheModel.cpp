#include "CacheModel.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace scope_model {

CacheHierarchy::Cache::Cache(CacheConfig config, std::uint64_t seed)
    : config_(std::move(config)), random_state_(seed) {
    if (config_.capacity_bytes == 0 || config_.line_bytes == 0 ||
        config_.associativity == 0 ||
        config_.capacity_bytes % (config_.line_bytes * config_.associativity) != 0) {
        throw std::invalid_argument("invalid cache geometry");
    }
    if (config_.policy != "LRU" && config_.policy != "FIFO" &&
        config_.policy != "RANDOM") {
        throw std::invalid_argument("cache policy must be LRU, FIFO, or RANDOM");
    }
    num_sets_ = config_.capacity_bytes /
                (config_.line_bytes * config_.associativity);
    if (config_.indexing != "modulo" && config_.indexing != "xor_fold")
        throw std::invalid_argument("indexing must be modulo or xor_fold");
    for (auto value = num_sets_ - 1; value; value >>= 1) ++index_bits_;
}

std::uint64_t CacheHierarchy::Cache::set_index(std::uint64_t line) const {
    // Generic XOR tag/index folding, not a reverse-engineered NVIDIA set function.
    if (config_.indexing == "xor_fold" && index_bits_) line ^= line >> index_bits_;
    return line % num_sets_;
}

bool CacheHierarchy::Cache::probe(std::uint64_t line, bool mark_dirty) {
    ++clock_;
    auto found = sets_.find(set_index(line));
    if (found == sets_.end()) {
        return false;
    }
    for (Entry& entry : found->second) {
        if (entry.line != line) {
            continue;
        }
        entry.dirty = entry.dirty || mark_dirty;
        entry.touched_at = clock_;
        return true;
    }
    return false;
}

bool CacheHierarchy::Cache::first_reference(std::uint64_t line) {
    return seen_lines_.insert(line).second;
}

std::size_t CacheHierarchy::Cache::victim(const std::vector<Entry>& entries) {
    if (config_.policy == "RANDOM") {
        random_state_ = random_state_ * 6364136223846793005ULL + 1;
        return static_cast<std::size_t>(random_state_ % entries.size());
    }
    std::size_t selected = 0;
    std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t index = 0; index < entries.size(); ++index) {
        const std::uint64_t age = config_.policy == "LRU"
                                      ? entries[index].touched_at
                                      : entries[index].inserted_at;
        if (age < oldest) {
            oldest = age;
            selected = index;
        }
    }
    return selected;
}

bool CacheHierarchy::Cache::insert(std::uint64_t line, bool dirty,
                                   Entry* evicted) {
    ++clock_;
    std::vector<Entry>& entries = sets_[set_index(line)];
    for (Entry& entry : entries) {
        if (entry.line == line) {
            entry.dirty = entry.dirty || dirty;
            entry.touched_at = clock_;
            return false;
        }
    }
    bool did_evict = false;
    if (entries.size() >= config_.associativity) {
        const std::size_t selected = victim(entries);
        *evicted = entries[selected];
        entries.erase(entries.begin() + static_cast<std::ptrdiff_t>(selected));
        did_evict = true;
    }
    entries.push_back({line, dirty, clock_, clock_});
    return did_evict;
}

CacheHierarchy::CacheHierarchy(std::vector<CacheConfig> configs,
                               std::uint64_t seed) {
    if (configs.empty()) {
        throw std::invalid_argument("cache hierarchy cannot be empty");
    }
    line_bytes_ = configs.front().line_bytes;
    for (std::size_t index = 0; index < configs.size(); ++index) {
        if (configs[index].line_bytes != line_bytes_) {
            throw std::invalid_argument("all cache levels must use one line size");
        }
        caches_.emplace_back(configs[index], seed + index + 1);
    }
}

void CacheHierarchy::writeback(std::size_t level, std::uint64_t line,
                               bool record, SimulationResult* result) {
    if (level >= caches_.size()) {
        if (record) {
            ++result->offchip_writebacks;
        }
        return;
    }
    if (record) {
        ++result->levels[level].writebacks;
    }
    Entry evicted{};
    if (caches_[level].insert(line, true, &evicted) && evicted.dirty) {
        writeback(level + 1, evicted.line, record, result);
    }
}

void CacheHierarchy::process(const Access& access, bool record, bool capture,
                             SimulationResult* result) {
    if (record) {
        ++result->measured_requests;
        result->measured_loads += access.operation == Operation::kLoad;
        ++result->phases[access.phase].requests;
        result->phases[access.phase].loads += access.operation == Operation::kLoad;
    }
    const std::uint64_t line = access.address / line_bytes_;
    std::vector<std::size_t> missed;
    std::string hit_level = "OFF";
    std::size_t hit_index = caches_.size();
    for (std::size_t level = 0; level < caches_.size(); ++level) {
        if (record) {
            ++result->levels[level].accesses;
            result->levels[level].load_accesses += access.operation == Operation::kLoad;
        }
        const bool compulsory = caches_[level].first_reference(line);
        const bool dirty = access.operation == Operation::kStore && level == 0;
        if (caches_[level].probe(line, dirty)) {
            if (record) {
                ++result->levels[level].hits;
                result->levels[level].load_hits += access.operation == Operation::kLoad;
                result->levels[level].store_hits += access.operation == Operation::kStore;
            }
            hit_level = "L" + std::to_string(level + 1);
            hit_index = level;
            break;
        }
        if (record) {
            if (compulsory) {
                ++result->levels[level].compulsory_misses;
            } else {
                ++result->levels[level].noncompulsory_misses;
            }
        }
        missed.push_back(level);
    }
    if (record) {
        ++result->phases[access.phase].hit_counts[hit_index];
        if (hit_index == caches_.size()) {
            result->offchip_loads += access.operation == Operation::kLoad;
            result->offchip_stores += access.operation == Operation::kStore;
        }
    }
    for (auto it = missed.rbegin(); it != missed.rend(); ++it) {
        const std::size_t level = *it;
        if (record) ++result->levels[level].fills;
        const bool dirty = access.operation == Operation::kStore && level == 0;
        Entry evicted{};
        if (caches_[level].insert(line, dirty, &evicted) && evicted.dirty) {
            writeback(level + 1, evicted.line, record, result);
        }
    }
    if (capture) {
        result->representative = {
            true, access.operation, access.address, access.size_bytes, hit_level, access.phase};
    }
}

SimulationResult CacheHierarchy::run(const AccessTrace& trace,
                                     std::uint64_t warmup_accesses,
                                     std::uint64_t sample_accesses) {
    if (sample_accesses == 0) {
        throw std::invalid_argument("sample_accesses must be positive");
    }
    SimulationResult result;
    result.levels.resize(caches_.size());
    result.phases.resize(std::max<std::size_t>(1, trace.phases().size()));
    const std::uint64_t total = warmup_accesses + sample_accesses;
    for (std::uint64_t ordinal = 0; ordinal < total; ++ordinal) {
        const bool record = ordinal >= warmup_accesses;
        const bool capture = ordinal == warmup_accesses + sample_accesses / 2;
        process(trace.at(ordinal), record, capture, &result);
    }
    return result;
}

}  // namespace scope_model
