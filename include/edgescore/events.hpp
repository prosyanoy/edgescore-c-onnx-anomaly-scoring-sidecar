// Turning HTTP payloads (Splunk HEC envelopes or plain NDJSON) into feature rows.
#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace edgescore {

struct EventRecord {
  std::vector<float> features;  // ordered like ModelMeta::features
  std::string id;               // optional caller-supplied correlation id
  std::string host;
  std::string source;
};

struct ParseResult {
  std::vector<EventRecord> events;
  bool ok = true;
  std::string error;            // human readable, set when !ok
  std::size_t error_index = 0;  // zero-based index of the offending event
};

// Accepts a stream of JSON objects separated by newlines or nothing at all.
// Each object is either a HEC envelope {"time":..,"host":..,"event":{...},"fields":{...}}
// or a bare telemetry object. Feature values are looked up in "event", then "fields",
// then the top level; numbers and numeric strings are accepted. All-or-nothing:
// the first bad event aborts the parse and nothing is returned.
ParseResult parse_events(std::string_view body, const std::vector<std::string>& features,
                         std::size_t max_events = 100000);

}  // namespace edgescore
