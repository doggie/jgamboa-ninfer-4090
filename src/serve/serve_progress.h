#pragma once

// Live, per-request inference progress for --llamacpplog. The engine executor invokes
// `on_event` from its execution thread at each prefill unit / decode round (see the
// `progress_sink` carried on the request options). This observer throttles to at most one
// line per second per phase, formats llama.cpp-style lines, and writes them to the console
// log. All state is guarded by one mutex that is never shared with the engine, and `on_event`
// does its console write outside that mutex, so attaching an observer cannot meaningfully
// perturb (or deadlock) the execution loop. When --llamacpplog is off no observer is attached
// and the executor's sink is empty, so this class is never instantiated and the hot path is
// untouched.

#include "ninfer/types.h"
#include "serve/console_log.h"
#include "serve/request_events.h"

#include <chrono>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

namespace ninfer::serve {

class ProgressObserver {
public:
    ProgressObserver() = default;

    // Serve-layer identity for the log prefix, set right after prepare() succeeds and before the
    // first engine event (which fires during run()). Must precede any on_event.
    void set_identity(std::uint64_t request_id, std::string x_request_id) {
        std::lock_guard lock(mutex_);
        id_     = request_id;
        x_id_   = std::move(x_request_id);
        // x_id_ keeps the full client header value; only the repeated per-second prefix shows the
        // short tail (change B) so the long request id stops crowding every line.
        prefix_ = "[req " + std::to_string(id_) + " x_request_id=" + short_hash(x_id_) + "] ";
        ready_  = true;
    }

    // Total prompt tokens for this request, set by prepare() once the prompt is tokenized. Used
    // for the prefill `progress = fraction of prompt` figure.
    void set_prompt_total(std::uint64_t prompt_tokens) {
        std::lock_guard lock(mutex_);
        prompt_total_ = prompt_tokens;
    }

    // Invoked by the engine per prefill unit / decode round. Throttles to one line per second per
    // phase (the first sample of each phase is always emitted) and logs a llama.cpp-style line.
    void on_event(const ProgressEvent& event) {
        std::string line;
        {
            std::lock_guard lock(mutex_);
            if (!ready_) { return; }
            const auto now = std::chrono::steady_clock::now();
            if (event.phase == ProgressEvent::Phase::CachePlanned) {
                // One-shot, non-throttled: emitted once per request, before the first prefill
                // unit, to surface the (planned) cache reuse the engine committed to.
                if (cache_plan_logged_) { return; }
                cache_plan_logged_ = true;
                line = format_cache_plan(event.tokens, event.cache_tier);
            } else if (event.phase == ProgressEvent::Phase::Prefill) {
                const bool due =
                    !last_prefill_time_.has_value() ||
                    (now - *last_prefill_time_) >= std::chrono::seconds(1);
                if (!due) { return; }
                last_prefill_time_ = now;
                line = format_prefill(event.tokens, event.phase_elapsed_seconds);
            } else {
                record_decode_sample(now, event.tokens);
                const bool due =
                    !last_decode_time_.has_value() ||
                    (now - *last_decode_time_) >= std::chrono::seconds(1);
                if (!due) { return; }
                last_decode_time_ = now;
                const double tg =
                    event.phase_elapsed_seconds > 0.0
                        ? static_cast<double>(event.tokens) / event.phase_elapsed_seconds
                        : 0.0;
                const std::optional<double> tg3 = decode_rate_over(now);
                // When the request has decoded for less than 3 s there is no 3 s window to
                // difference against, so report the overall rate instead of a misleading zero.
                line = format_decode(event.tokens, tg, tg3.value_or(tg));
            }
        }
        write_console_log(ConsoleLogLevel::Info, line);
    }

private:
    struct DecodeSample {
        std::chrono::steady_clock::time_point time;
        std::uint64_t tokens;
    };

    void record_decode_sample(std::chrono::steady_clock::time_point now, std::uint64_t tokens) {
        decode_samples_.push_back({now, tokens});
        // Bound the history to a little over the 3 s window so tg_3s always has a reference point.
        const auto horizon = now - std::chrono::seconds(4);
        while (!decode_samples_.empty() && decode_samples_.front().time < horizon) {
            decode_samples_.pop_front();
        }
    }

    // Average decode rate over the trailing 3 s. Returns nullopt when the request has decoded for
    // less than 3 s (no 3 s-old reference sample), so the caller can fall back to the overall
    // rate rather than present a misleading zero for a short decode.
    std::optional<double> decode_rate_over(std::chrono::steady_clock::time_point now) const {
        const auto window = now - std::chrono::seconds(3);
        const DecodeSample* reference = nullptr;
        for (const DecodeSample& sample : decode_samples_) {
            if (sample.time <= window) { reference = &sample; } else { break; }
        }
        if (reference == nullptr || decode_samples_.empty()) { return std::nullopt; }
        const std::uint64_t current = decode_samples_.back().tokens;
        if (current <= reference->tokens) { return 0.0; }
        const double seconds =
            std::chrono::duration<double>(decode_samples_.back().time - reference->time).count();
        if (seconds <= 0.0) { return 0.0; }
        return static_cast<double>(current - reference->tokens) / seconds;
    }

    std::string format_prefill(std::uint64_t tokens, double elapsed) const {
        const double progress = prompt_total_ > 0
                                    ? static_cast<double>(tokens) / static_cast<double>(prompt_total_)
                                    : 0.0;
        const double tps = elapsed > 0.0 ? static_cast<double>(tokens) / elapsed : 0.0;
        std::ostringstream out;
        out << prefix_ << "prompt processing, n_tokens = " << tokens << ", progress = "
            << std::fixed << std::setprecision(2) << progress << ", t = " << std::setprecision(2)
            << elapsed << " s / " << std::setprecision(2) << tps << " tokens per second";
        return out.str();
    }

    std::string format_decode(std::uint64_t n_gen, double tg, double tg_3s) const {
        std::ostringstream out;
        out << prefix_ << "n_gen = " << n_gen << ", tg = " << std::fixed << std::setprecision(2)
            << tg << " t/s, tg_3s = " << std::setprecision(2) << tg_3s << " t/s";
        return out.str();
    }

    // One-shot cache-usage line surfaced before prompt eval, mirroring llama-server which prints
    // the cached-token figure up front. `reused` is the *planned* reuse depth (the engine's
    // materialization decision), not the finally-measured value (that is reported as `confirmed`
    // at completion by format_llamacpp_done).
    std::string format_cache_plan(std::uint64_t reused, const std::string& tier) const {
        const double pct = prompt_total_ > 0
                              ? 100.0 * static_cast<double>(reused) / static_cast<double>(prompt_total_)
                              : 0.0;
        std::ostringstream out;
        out << prefix_ << "cache: reusing " << reused << " tokens ("
            << std::fixed << std::setprecision(2) << pct << "% of prompt) from " << tier
            << " [planned]";
        return out.str();
    }

    std::mutex mutex_;
    std::uint64_t id_ = 0;
    std::string x_id_;
    std::string prefix_;
    bool ready_ = false;
    std::uint64_t prompt_total_ = 0;
    bool cache_plan_logged_ = false;
    std::optional<std::chrono::steady_clock::time_point> last_prefill_time_;
    std::optional<std::chrono::steady_clock::time_point> last_decode_time_;
    std::deque<DecodeSample> decode_samples_;
};

} // namespace ninfer::serve
