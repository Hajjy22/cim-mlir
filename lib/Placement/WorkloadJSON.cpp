//===- WorkloadJSON.cpp - Read a cim-import-onnx --emit-workload file ---===//
//
// See WorkloadJSON.h for the schema and the rationale for a hand-rolled
// reader here rather than a JSON library.
//
// Structured as two layers, same shape as TargetYAMLParser.cpp: a small,
// generic `JsonValue` parser with no notion of this project's schema,
// followed by a schema-aware walk that turns a parsed `JsonValue` into a
// `WorkloadDocument` and produces every error message. Splitting it this
// way means a syntax error ("unexpected character") and a schema error
// ("'k' must be a non-negative integer") are never the same code path,
// and test/unit/workload_json_test.cpp can exercise both independently.
//
//===----------------------------------------------------------------------===//

#include "cim/Placement/WorkloadJSON.h"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace cim {

namespace {

//===----------------------------------------------------------------------===//
// A minimal JSON value parser: object, array, string, number, true/false/
// null. Restricted to what RFC 8259 defines -- no comments, no trailing
// commas, no unquoted keys -- since the only producer is
// python's `json.dumps`, a strict reader loses nothing real and gains a
// stronger guarantee that a byte it accepts is byte-for-byte what any
// other JSON reader (including the differential test's own `json.loads`)
// would accept too.
//===----------------------------------------------------------------------===//

struct JsonValue {
  enum class Kind { Null, Bool, Number, String, Array, Object } kind =
      Kind::Null;
  bool boolValue = false;
  double numberValue = 0.0;
  std::string stringValue;
  std::vector<JsonValue> arrayValue;
  std::vector<std::pair<std::string, JsonValue>> objectValue;

  const JsonValue *find(const std::string &key) const {
    for (const auto &kv : objectValue)
      if (kv.first == key)
        return &kv.second;
    return nullptr;
  }
};

const char *kindName(JsonValue::Kind k) {
  switch (k) {
  case JsonValue::Kind::Null: return "null";
  case JsonValue::Kind::Bool: return "a boolean";
  case JsonValue::Kind::Number: return "a number";
  case JsonValue::Kind::String: return "a string";
  case JsonValue::Kind::Array: return "an array";
  case JsonValue::Kind::Object: return "an object";
  }
  return "a value";
}

class JsonParser {
public:
  explicit JsonParser(const std::string &input) : text(input) {}

  /// Parses exactly one JSON value, then requires the rest of the input to
  /// be whitespace only -- a trailing `}garbage` is a syntax error, not
  /// something to ignore.
  bool parse(JsonValue &out, std::string *error) {
    skipWhitespace();
    if (!parseValue(out, error))
      return false;
    skipWhitespace();
    if (pos != text.size()) {
      fail(error, "trailing content after the top-level JSON value");
      return false;
    }
    return true;
  }

private:
  const std::string &text;
  size_t pos = 0;

  void fail(std::string *error, const std::string &what) const {
    if (!error)
      return;
    // Line/column, 1-based, the same convention TargetYAMLParser's own
    // diagnostics use for its own line numbers.
    size_t line = 1, col = 1;
    for (size_t i = 0; i < pos && i < text.size(); ++i) {
      if (text[i] == '\n') { ++line; col = 1; } else { ++col; }
    }
    std::ostringstream os;
    os << "line " << line << ", column " << col << ": " << what;
    *error = os.str();
  }

  bool atEnd() const { return pos >= text.size(); }
  char peek() const { return text[pos]; }

  void skipWhitespace() {
    while (!atEnd() && std::isspace(static_cast<unsigned char>(peek())))
      ++pos;
  }

  bool expect(char c, std::string *error) {
    if (atEnd() || peek() != c) {
      fail(error, std::string("expected '") + c + "'");
      return false;
    }
    ++pos;
    return true;
  }

  bool literal(const char *word, std::string *error) {
    const size_t len = std::strlen(word);
    if (text.compare(pos, len, word) != 0) {
      fail(error, std::string("expected '") + word + "'");
      return false;
    }
    pos += len;
    return true;
  }

  bool parseValue(JsonValue &out, std::string *error) {
    if (atEnd()) {
      fail(error, "unexpected end of input; expected a JSON value");
      return false;
    }
    switch (peek()) {
    case '{': return parseObject(out, error);
    case '[': return parseArray(out, error);
    case '"': return parseString(out, error);
    case 't':
      if (!literal("true", error)) return false;
      out.kind = JsonValue::Kind::Bool;
      out.boolValue = true;
      return true;
    case 'f':
      if (!literal("false", error)) return false;
      out.kind = JsonValue::Kind::Bool;
      out.boolValue = false;
      return true;
    case 'n':
      if (!literal("null", error)) return false;
      out.kind = JsonValue::Kind::Null;
      return true;
    default:
      if (peek() == '-' || std::isdigit(static_cast<unsigned char>(peek())))
        return parseNumber(out, error);
      fail(error, std::string("unexpected character '") + peek() +
                      "'; expected a JSON value");
      return false;
    }
  }

  bool parseObject(JsonValue &out, std::string *error) {
    if (!expect('{', error))
      return false;
    out.kind = JsonValue::Kind::Object;
    skipWhitespace();
    if (!atEnd() && peek() == '}') {
      ++pos;
      return true;
    }
    while (true) {
      skipWhitespace();
      JsonValue keyVal;
      if (atEnd() || peek() != '"') {
        fail(error, "expected a string key");
        return false;
      }
      if (!parseString(keyVal, error))
        return false;
      skipWhitespace();
      if (!expect(':', error))
        return false;
      skipWhitespace();
      JsonValue val;
      if (!parseValue(val, error))
        return false;
      out.objectValue.emplace_back(keyVal.stringValue, std::move(val));
      skipWhitespace();
      if (!atEnd() && peek() == ',') {
        ++pos;
        continue;
      }
      break;
    }
    skipWhitespace();
    return expect('}', error);
  }

  bool parseArray(JsonValue &out, std::string *error) {
    if (!expect('[', error))
      return false;
    out.kind = JsonValue::Kind::Array;
    skipWhitespace();
    if (!atEnd() && peek() == ']') {
      ++pos;
      return true;
    }
    while (true) {
      skipWhitespace();
      JsonValue val;
      if (!parseValue(val, error))
        return false;
      out.arrayValue.push_back(std::move(val));
      skipWhitespace();
      if (!atEnd() && peek() == ',') {
        ++pos;
        continue;
      }
      break;
    }
    skipWhitespace();
    return expect(']', error);
  }

  bool parseString(JsonValue &out, std::string *error) {
    if (!expect('"', error))
      return false;
    out.kind = JsonValue::Kind::String;
    std::string &s = out.stringValue;
    while (true) {
      if (atEnd()) {
        fail(error, "unterminated string");
        return false;
      }
      char c = text[pos++];
      if (c == '"')
        break;
      if (c == '\\') {
        if (atEnd()) {
          fail(error, "unterminated escape sequence");
          return false;
        }
        char esc = text[pos++];
        switch (esc) {
        case '"': s += '"'; break;
        case '\\': s += '\\'; break;
        case '/': s += '/'; break;
        case 'b': s += '\b'; break;
        case 'f': s += '\f'; break;
        case 'n': s += '\n'; break;
        case 'r': s += '\r'; break;
        case 't': s += '\t'; break;
        case 'u': {
          if (pos + 4 > text.size()) {
            fail(error, "truncated \\u escape");
            return false;
          }
          unsigned code = 0;
          for (int i = 0; i < 4; ++i) {
            char h = text[pos++];
            code <<= 4;
            if (h >= '0' && h <= '9') code |= static_cast<unsigned>(h - '0');
            else if (h >= 'a' && h <= 'f') code |= static_cast<unsigned>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') code |= static_cast<unsigned>(h - 'A' + 10);
            else {
              fail(error, "invalid \\u escape digit");
              return false;
            }
          }
          // A minimal UTF-8 encoder. Layer/op names in practice are ASCII
          // (ONNX node names), so this only has to be CORRECT for the
          // basic multilingual plane, not exhaustive over surrogate pairs.
          if (code <= 0x7F) {
            s += static_cast<char>(code);
          } else if (code <= 0x7FF) {
            s += static_cast<char>(0xC0 | (code >> 6));
            s += static_cast<char>(0x80 | (code & 0x3F));
          } else {
            s += static_cast<char>(0xE0 | (code >> 12));
            s += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
            s += static_cast<char>(0x80 | (code & 0x3F));
          }
          break;
        }
        default:
          fail(error, std::string("invalid escape '\\") + esc + "'");
          return false;
        }
      } else if (static_cast<unsigned char>(c) < 0x20) {
        fail(error, "control character in string literal");
        return false;
      } else {
        s += c;
      }
    }
    return true;
  }

  bool parseNumber(JsonValue &out, std::string *error) {
    const size_t start = pos;
    if (!atEnd() && peek() == '-')
      ++pos;
    if (atEnd() || !std::isdigit(static_cast<unsigned char>(peek()))) {
      fail(error, "invalid number");
      return false;
    }
    while (!atEnd() && std::isdigit(static_cast<unsigned char>(peek())))
      ++pos;
    if (!atEnd() && peek() == '.') {
      ++pos;
      if (atEnd() || !std::isdigit(static_cast<unsigned char>(peek()))) {
        fail(error, "invalid number: digit expected after '.'");
        return false;
      }
      while (!atEnd() && std::isdigit(static_cast<unsigned char>(peek())))
        ++pos;
    }
    if (!atEnd() && (peek() == 'e' || peek() == 'E')) {
      ++pos;
      if (!atEnd() && (peek() == '+' || peek() == '-'))
        ++pos;
      if (atEnd() || !std::isdigit(static_cast<unsigned char>(peek()))) {
        fail(error, "invalid number: digit expected in exponent");
        return false;
      }
      while (!atEnd() && std::isdigit(static_cast<unsigned char>(peek())))
        ++pos;
    }
    out.kind = JsonValue::Kind::Number;
    out.numberValue = std::strtod(text.substr(start, pos - start).c_str(), nullptr);
    return true;
  }
};

//===----------------------------------------------------------------------===//
// Schema-aware extraction.
//===----------------------------------------------------------------------===//

/// A non-negative integer that fits uint32_t, or a schema error naming
/// `field` and the value actually found.
bool asUint32Field(const JsonValue &obj, const std::string &field,
                   const std::string &where, uint32_t &out,
                   std::string *error) {
  const JsonValue *v = obj.find(field);
  if (!v) {
    if (error) *error = where + ": missing required field '" + field + "'";
    return false;
  }
  if (v->kind != JsonValue::Kind::Number ||
      v->numberValue != std::floor(v->numberValue) || v->numberValue < 0 ||
      v->numberValue > 4294967295.0) {
    std::ostringstream os;
    os << where << ": field '" << field
       << "' must be a non-negative integer that fits uint32";
    if (error) *error = os.str();
    return false;
  }
  out = static_cast<uint32_t>(v->numberValue);
  return true;
}

bool asStringField(const JsonValue &obj, const std::string &field,
                   const std::string &where, std::string &out,
                   std::string *error, bool required = true) {
  const JsonValue *v = obj.find(field);
  if (!v) {
    if (required && error)
      *error = where + ": missing required field '" + field + "'";
    return !required;
  }
  if (v->kind != JsonValue::Kind::String) {
    if (error)
      *error = where + ": field '" + field + "' must be a string, found " +
               kindName(v->kind);
    return false;
  }
  out = v->stringValue;
  return true;
}

} // namespace

bool parseWorkloadDocument(const std::string &text, WorkloadDocument &doc,
                           std::string *error) {
  JsonParser parser(text);
  JsonValue root;
  if (!parser.parse(root, error))
    return false;

  if (root.kind != JsonValue::Kind::Object) {
    if (error)
      *error = "the top-level JSON value must be an object, found " +
               std::string(kindName(root.kind));
    return false;
  }

  if (!asStringField(root, "model", "top level", doc.model, error))
    return false;

  const JsonValue *layersVal = root.find("layers");
  if (!layersVal) {
    if (error) *error = "top level: missing required field 'layers'";
    return false;
  }
  if (layersVal->kind != JsonValue::Kind::Array) {
    if (error)
      *error = "top level: field 'layers' must be an array, found " +
               std::string(kindName(layersVal->kind));
    return false;
  }

  const JsonValue *skippedVal = root.find("skipped");
  if (!skippedVal) {
    // Required, not merely optional-and-defaulted-to-empty: a document
    // that never mentions `skipped` is indistinguishable, on the wire,
    // from a producer that forgot to disclose anything it left out. See
    // WorkloadDocument's own comment.
    if (error) *error = "top level: missing required field 'skipped'";
    return false;
  }
  if (skippedVal->kind != JsonValue::Kind::Array) {
    if (error)
      *error = "top level: field 'skipped' must be an array, found " +
               std::string(kindName(skippedVal->kind));
    return false;
  }

  doc.layers.clear();
  doc.layers.reserve(layersVal->arrayValue.size());
  for (size_t i = 0; i < layersVal->arrayValue.size(); ++i) {
    const JsonValue &elem = layersVal->arrayValue[i];
    const std::string where =
        "layers[" + std::to_string(i) + "]";
    if (elem.kind != JsonValue::Kind::Object) {
      if (error)
        *error = where + " must be an object, found " +
                 std::string(kindName(elem.kind));
      return false;
    }
    WorkloadLayer layer;
    if (!asStringField(elem, "name", where, layer.name, error))
      return false;
    if (!asStringField(elem, "op_type", where, layer.opType, error))
      return false;
    if (!asUint32Field(elem, "k", where, layer.k, error))
      return false;
    if (!asUint32Field(elem, "n", where, layer.n, error))
      return false;
    // k/n are a contraction dimension and an output-channel count; zero
    // in either is not a degenerate-but-valid layer, it is a layer that
    // cannot exist (a matmul with zero weights). Left unchecked,
    // partitionBlockCount(0, n, ...) silently returns 0 blocks for that
    // layer -- it would sail through as one of "layers_analyzed" while
    // contributing nothing whatsoever to the placement result, the exact
    // confident-but-partial number this schema's `skipped`/`note` fields
    // exist to prevent everywhere else.
    if (layer.k == 0 || layer.n == 0) {
      if (error)
        *error = where + ": k=" + std::to_string(layer.k) +
                 ", n=" + std::to_string(layer.n) +
                 " -- a layer must have a positive contraction dimension "
                 "and a positive output-channel count; zero in either is "
                 "not a real weight matrix";
      return false;
    }
    doc.layers.push_back(std::move(layer));
  }

  doc.skipped.clear();
  doc.skipped.reserve(skippedVal->arrayValue.size());
  for (size_t i = 0; i < skippedVal->arrayValue.size(); ++i) {
    const JsonValue &elem = skippedVal->arrayValue[i];
    const std::string where = "skipped[" + std::to_string(i) + "]";
    if (elem.kind != JsonValue::Kind::Object) {
      if (error)
        *error = where + " must be an object, found " +
                 std::string(kindName(elem.kind));
      return false;
    }
    WorkloadSkip skip;
    if (!asStringField(elem, "name", where, skip.name, error))
      return false;
    if (!asStringField(elem, "op_type", where, skip.opType, error))
      return false;
    if (!asStringField(elem, "reason", where, skip.reason, error))
      return false;
    doc.skipped.push_back(std::move(skip));
  }

  return true;
}

bool parseWorkloadDocumentFromFile(const std::string &path,
                                    WorkloadDocument &doc,
                                    std::string *error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    if (error) *error = "failed to open workload file '" + path + "'";
    return false;
  }
  std::ostringstream buf;
  buf << in.rdbuf();
  const bool ok = parseWorkloadDocument(buf.str(), doc, error);
  if (!ok && error)
    *error = "in workload file '" + path + "': " + *error;
  return ok;
}

} // namespace cim
