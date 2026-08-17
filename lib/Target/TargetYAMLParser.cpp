//===- TargetYAMLParser.cpp - Parse targets/*.yaml (spec Sec. 7) -*- C++ -*-==//
//
// A deliberately small reader for the strict YAML subset the target
// description format uses: nested block mappings of scalars, `#` comments,
// and nothing else. No anchors, aliases, flow collections, multi-line
// scalars, or sequences — the schema in docs/target-format.md needs none of
// them, and a verifier rejects anything it does not understand rather than
// guessing.
//
// This was originally written against llvm::yaml. It is dependency-free
// instead so that the target file, the cost model, and cim-bench can all be
// built and tested without an LLVM toolchain — otherwise reading a 40-line
// config file would require a multi-hour LLVM build first. The tradeoff is
// that this parser accepts only the subset above; see
// test/unit/target_spec_test.cpp, which parses every shipped target file
// and checks the values field by field.
//
//===----------------------------------------------------------------------===//

#include "cim/Target/TargetSpec.h"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

namespace cim {

namespace {

/// Strip a trailing `# ...` comment. Only treats `#` as a comment marker at
/// the start of the line or after whitespace, so a `#` inside a value is
/// preserved.
std::string stripComment(const std::string &line) {
  for (size_t i = 0; i < line.size(); ++i) {
    if (line[i] == '#' && (i == 0 || std::isspace(static_cast<unsigned char>(line[i - 1]))))
      return line.substr(0, i);
  }
  return line;
}

std::string trim(const std::string &s) {
  size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b])))
    ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])))
    --e;
  return s.substr(b, e - b);
}

std::string unquote(const std::string &s) {
  if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') ||
                        (s.front() == '\'' && s.back() == '\'')))
    return s.substr(1, s.size() - 2);
  return s;
}

/// Flattens the document into dotted paths, e.g.
/// "costs.program.latency_ns" -> "12000".
class FlatYaml {
public:
  bool load(std::istream &in, std::string &error) {
    std::vector<std::pair<int, std::string>> stack;
    std::string raw;
    int lineNo = 0;

    while (std::getline(in, raw)) {
      ++lineNo;
      const std::string line = stripComment(raw);
      if (trim(line).empty())
        continue;

      if (line.find('\t') != std::string::npos) {
        error = "line " + std::to_string(lineNo) + ": tabs are not valid YAML indentation";
        return false;
      }
      if (trim(line)[0] == '-') {
        error = "line " + std::to_string(lineNo) +
                ": sequences are not supported by this target-file reader";
        return false;
      }

      int indent = 0;
      while (indent < static_cast<int>(line.size()) && line[indent] == ' ')
        ++indent;

      const size_t colon = line.find(':');
      if (colon == std::string::npos) {
        error = "line " + std::to_string(lineNo) + ": expected 'key: value'";
        return false;
      }

      const std::string key = trim(line.substr(0, colon));
      const std::string value = trim(line.substr(colon + 1));
      if (key.empty()) {
        error = "line " + std::to_string(lineNo) + ": empty key";
        return false;
      }

      while (!stack.empty() && stack.back().first >= indent)
        stack.pop_back();

      if (value.empty()) {
        stack.push_back({indent, key});
        continue;
      }

      std::string path;
      for (const auto &entry : stack)
        path += entry.second + ".";
      path += key;
      values[path] = unquote(value);
    }
    return true;
  }

  bool has(const std::string &path) const { return values.count(path) != 0; }

  const std::string *find(const std::string &path) const {
    auto it = values.find(path);
    return it == values.end() ? nullptr : &it->second;
  }

  const std::map<std::string, std::string> &all() const { return values; }

private:
  std::map<std::string, std::string> values;
};

/// Field readers. Each records a problem rather than throwing, so a bad
/// target file produces a list of what is wrong instead of the first error.
struct Reader {
  const FlatYaml &yaml;
  std::vector<std::string> errors;

  void require(const std::string &path, std::string &out) {
    if (const std::string *v = yaml.find(path))
      out = *v;
    else
      errors.push_back("missing required field '" + path + "'");
  }

  void optional(const std::string &path, std::string &out) {
    if (const std::string *v = yaml.find(path))
      out = *v;
  }

  bool toUInt(const std::string &path, const std::string &text, uint32_t &out) {
    // strtoul silently *negates* a leading '-' and saturates on overflow, so
    // `count: -1` would arrive as 4294967295 and sail past the "must be
    // greater than zero" check below. Both are rejected here instead: a
    // target file that describes four billion tiles because someone typed a
    // minus sign is worse than one that fails to load.
    if (!text.empty() && (text[0] == '-' || text[0] == '+')) {
      errors.push_back("field '" + path + "': '" + text +
                       "' is not a non-negative integer");
      return false;
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0') {
      errors.push_back("field '" + path + "': '" + text + "' is not an integer");
      return false;
    }
    if (errno == ERANGE || parsed > 4294967295UL) {
      errors.push_back("field '" + path + "': '" + text +
                       "' does not fit in 32 bits");
      return false;
    }
    out = static_cast<uint32_t>(parsed);
    return true;
  }

  bool toDouble(const std::string &path, const std::string &text, double &out) {
    char *end = nullptr;
    const double parsed = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || *end != '\0') {
      errors.push_back("field '" + path + "': '" + text + "' is not a number");
      return false;
    }
    // strtod accepts "inf" and "nan". Either one propagates silently through
    // every energy and latency sum in the cost report, turning a typo into a
    // results file full of NaN rather than into a diagnostic.
    if (!std::isfinite(parsed)) {
      errors.push_back("field '" + path + "': '" + text +
                       "' is not a finite number");
      return false;
    }
    out = parsed;
    return true;
  }

  void requireUInt(const std::string &path, uint32_t &out) {
    if (const std::string *v = yaml.find(path))
      toUInt(path, *v, out);
    else
      errors.push_back("missing required field '" + path + "'");
  }

  void requireDouble(const std::string &path, double &out) {
    if (const std::string *v = yaml.find(path))
      toDouble(path, *v, out);
    else
      errors.push_back("missing required field '" + path + "'");
  }

  void optionalDouble(const std::string &path, double &out) {
    if (const std::string *v = yaml.find(path))
      toDouble(path, *v, out);
  }

  void optionalBool(const std::string &path, bool &out) {
    const std::string *v = yaml.find(path);
    if (!v)
      return;
    if (*v == "true")
      out = true;
    else if (*v == "false")
      out = false;
    else
      errors.push_back("field '" + path + "': '" + *v + "' is not a boolean");
  }

  void requireClass(const std::string &path, TargetClass &out) {
    const std::string *v = yaml.find(path);
    if (!v) {
      errors.push_back("missing required field '" + path + "'");
      return;
    }
    if (*v == "near_memory")
      out = TargetClass::NearMemory;
    else if (*v == "digital_cim")
      out = TargetClass::DigitalCIM;
    else if (*v == "analog_cim")
      out = TargetClass::AnalogCIM;
    else if (*v == "dpu")
      out = TargetClass::DPU;
    else
      errors.push_back("field '" + path + "': unknown target class '" + *v +
                       "' (expected near_memory, digital_cim, analog_cim, or dpu)");
  }

  void optionalProvenance(const std::string &path, Provenance &out) {
    const std::string *v = yaml.find(path);
    if (!v)
      return;
    if (*v == "measured")
      out = Provenance::Measured;
    else if (*v == "simulated")
      out = Provenance::Simulated;
    else if (*v == "estimated")
      out = Provenance::Estimated;
    else
      errors.push_back("field '" + path + "': unknown provenance '" + *v +
                       "' (expected measured, simulated, or estimated)");
  }
};

} // namespace

bool parseTargetSpec(std::istream &in, TargetSpec &outSpec, std::string *error) {
  FlatYaml yaml;
  std::string loadError;
  if (!yaml.load(in, loadError)) {
    if (error)
      *error = loadError;
    return false;
  }

  Reader r{yaml, {}};
  TargetSpec spec;

  r.require("name", spec.name);
  r.optional("description", spec.description);
  r.optional("version", spec.version);
  r.requireClass("class", spec.targetClass);
  r.optionalProvenance("provenance", spec.provenance);

  r.requireUInt("tiles.count", spec.tiles.count);
  r.requireUInt("tiles.rows", spec.tiles.rows);
  r.requireUInt("tiles.cols", spec.tiles.cols);
  r.require("tiles.weight_dtype", spec.tiles.weightDtype);
  r.require("tiles.activation_dtype", spec.tiles.activationDtype);
  r.optional("tiles.accumulator_dtype", spec.tiles.accumulatorDtype);
  r.optionalBool("tiles.persistent", spec.tiles.persistent);
  r.optional("tiles.persistence", spec.tiles.persistence);

  r.requireDouble("costs.program.latency_ns", spec.costs.program.latencyNs);
  r.requireDouble("costs.program.energy_pj", spec.costs.program.energyPj);
  r.requireDouble("costs.mvm.latency_ns", spec.costs.mvm.latencyNs);
  r.requireDouble("costs.mvm.energy_pj", spec.costs.mvm.energyPj);
  r.requireDouble("costs.transfer.bandwidth_gbps", spec.costs.transfer.bandwidthGbps);
  r.requireDouble("costs.transfer.energy_pj_per_byte",
                  spec.costs.transfer.energyPjPerByte);
  r.requireDouble("costs.requantize.latency_ns", spec.costs.requantize.latencyNs);
  r.requireDouble("costs.requantize.energy_pj", spec.costs.requantize.energyPj);
  r.requireDouble("costs.reduce_partial.latency_ns",
                  spec.costs.reducePartial.latencyNs);
  r.requireDouble("costs.reduce_partial.energy_pj",
                  spec.costs.reducePartial.energyPj);
  // Required, on exactly the same reasoning as reduce_partial above: an
  // N-operand cim.reduce_max lowers to N-1 real cimrt_reduce_max calls, so
  // leaving this unset would silently cost a whole pooling window at zero.
  r.requireDouble("costs.reduce_max.latency_ns", spec.costs.reduceMax.latencyNs);
  r.requireDouble("costs.reduce_max.energy_pj", spec.costs.reduceMax.energyPj);
  r.optionalDouble("costs.standby_leakage_uw_per_tile",
                   spec.costs.standbyLeakageUwPerTile);

  r.requireUInt("precision.output_effective_bits",
                spec.precision.outputEffectiveBits);

  r.optionalBool("capabilities.double_buffer_program",
                 spec.capabilities.doubleBufferProgram);
  r.optionalBool("capabilities.partial_sum_in_place",
                 spec.capabilities.partialSumInPlace);
  r.optionalBool("capabilities.autonomous_control",
                 spec.capabilities.autonomousControl);

  // Semantic checks. A target file that parses but describes an impossible
  // device produces confidently wrong cost numbers, which is worse than a
  // parse error.
  if (spec.tiles.count == 0)
    r.errors.push_back("tiles.count must be greater than zero");
  if (spec.tiles.rows == 0 || spec.tiles.cols == 0)
    r.errors.push_back("tiles.rows and tiles.cols must be greater than zero");
  if (spec.precision.outputEffectiveBits == 0)
    r.errors.push_back("precision.output_effective_bits must be greater than zero");
  if (!spec.tiles.persistence.empty() && spec.tiles.persistence != "volatile" &&
      spec.tiles.persistence != "nonvolatile") {
    r.errors.push_back("tiles.persistence must be 'volatile' or 'nonvolatile'");
  } else if (!spec.tiles.persistence.empty()) {
    // tiles.persistence is a documentation-only string -- see
    // TargetSpec.h's own comment -- nothing downstream reads it back;
    // tiles.persistent (bool) is what cim-partition, the cost model, and
    // every pass that branches on volatility actually consult. Left
    // free-floating, the two fields could describe two different devices
    // in the same file with nothing to catch it. Every shipped target
    // file already keeps them in agreement by convention; this makes
    // that convention load-bearing instead of merely followed, and also
    // catches the quieter version of the same mistake -- an author who
    // states intent via `persistence` but forgets the `persistent` bool
    // that is actually load-bearing, which would otherwise silently keep
    // its unset default (false) regardless of what `persistence` said.
    const bool declaredNonvolatile = spec.tiles.persistence == "nonvolatile";
    if (declaredNonvolatile != spec.tiles.persistent)
      r.errors.push_back(
          "tiles.persistence ('" + spec.tiles.persistence +
          "') contradicts tiles.persistent (" +
          (spec.tiles.persistent ? "true" : "false") +
          "), which is the field the cost model actually reads -- fix one "
          "of the two, or drop tiles.persistence entirely");
  }

  if (!r.errors.empty()) {
    if (error) {
      std::ostringstream os;
      for (size_t i = 0; i < r.errors.size(); ++i)
        os << (i ? "; " : "") << r.errors[i];
      *error = os.str();
    }
    return false;
  }

  outSpec = spec;
  return true;
}

bool parseTargetSpecFromFile(const std::string &path, TargetSpec &outSpec) {
  std::string error;
  if (!parseTargetSpecFromFile(path, outSpec, &error)) {
    std::cerr << "cim: " << error << "\n";
    return false;
  }
  return true;
}

bool parseTargetSpecFromFile(const std::string &path, TargetSpec &outSpec,
                              std::string *error) {
  std::ifstream in(path);
  if (!in) {
    if (error)
      *error = "failed to open target file '" + path + "'";
    return false;
  }
  std::string parseError;
  if (!parseTargetSpec(in, outSpec, &parseError)) {
    if (error)
      *error = "in target file '" + path + "': " + parseError;
    return false;
  }
  return true;
}

} // namespace cim
