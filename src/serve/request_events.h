#pragma once

#include "serve/generation_service.h"
#include "serve/request.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace ninfer::serve {

// Fork-local: the host-exposed total this deployment reports on the request line. Shared
// between the operational renderer and the JSONL record so the two cannot drift.
[[nodiscard]] inline double
request_host_exposed_seconds(const ninfer::GenerationEngineTiming& timing) noexcept {
    return timing.engine_boundary_exposed_seconds + timing.program_submit_exposed_seconds +
           timing.program_post_exposed_seconds + timing.engine_commit_output_exposed_seconds +
           timing.engine_maintenance_exposed_seconds;
}

// Tail length for console digest / request-id truncation. Sampled once from the environment so
// an operator can tune it without a rebuild: NINFER_LOG_HASH_TAIL (default 8; 0 disables
// truncation). Display-only -- it never affects cache behaviour, and the JSONL / API bodies
// always keep the full value.
[[nodiscard]] inline std::size_t console_hash_tail() {
    constexpr std::size_t kDefaultTail = 8;
    static const std::size_t cached = [] {
        const char* raw = std::getenv("NINFER_LOG_HASH_TAIL");
        if (raw == nullptr || *raw == '\0') { return kDefaultTail; }
        try {
            std::size_t index = 0;
            const std::size_t value = std::stoul(raw, &index);
            return (index == 0) ? kDefaultTail : value;
        } catch (...) { return kDefaultTail; }
    }();
    return cached;
}

// Shorten a long digest or request id for console display. The same full value always maps to
// the same short value (a fixed tail substring, never a re-hash), so every line of one request
// shows the same recognizable tail. The client x-request-id form "req_<hex>_<seq>" keeps its
// "req_" prefix and the trailing "_<seq>" and shortens only the middle hex digest; any other
// digest (e.g. a 16-hex session_digest) is reduced to its tail. Callers apply this to console
// text only -- JSONL / API bodies keep the full value and never call this.
[[nodiscard]] inline std::string short_hash(std::string_view full,
                                            std::size_t tail = console_hash_tail()) {
    if (tail == 0) { return std::string(full); }
    const std::size_t n = std::min<std::size_t>(tail, full.size());
    if (n >= full.size()) { return std::string(full); }
    constexpr std::string_view kReqPrefix = "req_";
    if (full.size() > kReqPrefix.size() &&
        full.compare(0, kReqPrefix.size(), kReqPrefix) == 0) {
        const std::string_view rest = full.substr(kReqPrefix.size());  // "<hex>_<seq>"
        const auto underscore       = rest.rfind('_');
        if (underscore != rest.npos && underscore > 0) {
            const std::string_view hex = rest.substr(0, underscore);  // "<hex>"
            const std::string_view seq = rest.substr(underscore);     // "_<seq>"
            const std::size_t hn = std::min<std::size_t>(tail, hex.size());
            if (hn >= hex.size()) { return std::string(full); }
            return std::string(kReqPrefix) + "\xE2\x80\xA6" +
                   std::string(hex.substr(hex.size() - hn)) + std::string(seq);
        }
    }
    return "\xE2\x80\xA6" + std::string(full.substr(full.size() - n));
}

struct RequestLogContext {
    std::uint64_t id = 0;
    std::string x_request_id;
    std::string protocol;
    bool llamacpp_log = false;
    std::string model;
    bool stream                             = false;
    std::size_t message_count               = 0;
    std::size_t media_item_count            = 0;
    int requested_output_tokens             = 0;
    bool requested_output_tokens_client_set = false;
    std::size_t tool_count                  = 0;
    ToolChoice tool_choice;
    bool has_tool_history = false;
    bool enable_thinking  = true;
    std::optional<std::uint32_t> thinking_budget;
    std::optional<RequestedReasoningEffort> requested_reasoning_effort;
    std::optional<bool> preserve_thinking;
    bool preserve_thinking_semantic_change = false;
    ninfer::ResolvedSamplingParameters sampling;
    double acquisition_seconds = 0.0;
    ninfer::PromptPreparationStats preparation;
};

struct RequestLogMetadata {
    std::string model;
    bool stream                            = false;
    bool output_tokens_explicit            = false;
    bool preserve_thinking_semantic_change = false;
};

// A parsed generation request that failed during synchronous preparation. It intentionally has a
// separate shape because sampler and prompt semantics may not have resolved.
struct RequestRejectionLogContext {
    std::uint64_t id = 0;
    std::string protocol;
    std::string model;
    bool stream                             = false;
    std::size_t message_count               = 0;
    std::size_t media_item_count            = 0;
    int requested_output_tokens             = 0;
    bool requested_output_tokens_client_set = false;
    std::size_t tool_count                  = 0;
    ToolChoice tool_choice;
    bool has_tool_history = false;
    std::optional<RequestedReasoningEffort> requested_reasoning_effort;
    ApiError error;
};

enum class RequestFailurePhase : std::uint8_t {
    Prepare,
    Generation,
    ResponseRender,
    ResponseStore,
    Transport,
    Http,
};

enum class RequestFailureClass : std::uint8_t {
    ClientInput,
    ClientDisconnected,
    Overload,
    Timeout,
    Unavailable,
    Upstream,
    Internal,
};

struct RequestFailure {
    RequestFailurePhase phase          = RequestFailurePhase::Generation;
    RequestFailureClass classification = RequestFailureClass::Internal;
    int http_status                    = 0;
    std::string error_type;
    std::string error_code;
    std::string param;
    // Used only by the independent JSONL measurement writer. Operational rendering never consumes
    // this field.
    std::string machine_message;
};

struct ThroughputReport {
    double interval_seconds               = 0.0;
    std::uint64_t computed_prefill_tokens = 0;
    std::uint64_t committed_decode_tokens = 0;
    std::uint64_t decode_rounds           = 0;
    std::uint64_t decode_row_rounds       = 0;
    ninfer::RuntimeStats previous;
    ninfer::RuntimeStats current;
};

RequestLogContext make_request_log_context(std::uint64_t id, std::string x_request_id,
                                           std::string protocol, const GenerationRequest& request,
                                           const RequestLogMetadata& metadata,
                                           const PreparedRequest& prepared, bool llamacpp_log);
RequestRejectionLogContext make_request_rejection_log_context(std::uint64_t id,
                                                              std::string protocol,
                                                              const GenerationRequest& request,
                                                              const RequestLogMetadata& metadata,
                                                              ApiError error);

[[nodiscard]] RequestFailure make_request_failure(RequestFailurePhase phase, const ApiError& error);
[[nodiscard]] RequestFailure make_generation_request_failure(const ApiError& error);
[[nodiscard]] RequestFailure make_internal_request_failure(RequestFailurePhase phase,
                                                           std::string machine_message);
[[nodiscard]] RequestFailure make_client_disconnected_failure(RequestFailurePhase phase);

} // namespace ninfer::serve
