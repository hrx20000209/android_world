// V0 Android GUI exploration selector.
//
// The program intentionally has no Android framework dependency and no third
// party library. It reads a small JSON array, applies a fixed multi-hot
// vocabulary, and evaluates score = w^T x + b. The same binary therefore runs
// on a workstation and on an arm64 Android shell.

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/resource.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct JsonValue {
  enum class Type { Null, Bool, Number, String, Array, Object };
  Type type = Type::Null;
  bool boolean = false;
  double number = 0.0;
  std::string string;
  std::vector<JsonValue> array;
  std::map<std::string, JsonValue> object;
};

class JsonParser {
 public:
  explicit JsonParser(std::string source) : source_(std::move(source)) {}

  JsonValue parse() {
    skip_space();
    JsonValue value = parse_value();
    skip_space();
    if (position_ != source_.size()) {
      fail("trailing characters");
    }
    return value;
  }

 private:
  [[noreturn]] void fail(const std::string& message) const {
    throw std::runtime_error("JSON parse error at " + std::to_string(position_) +
                             ": " + message);
  }

  void skip_space() {
    while (position_ < source_.size() &&
           std::isspace(static_cast<unsigned char>(source_[position_]))) {
      ++position_;
    }
  }

  bool consume(char expected) {
    skip_space();
    if (position_ < source_.size() && source_[position_] == expected) {
      ++position_;
      return true;
    }
    return false;
  }

  void expect(char expected) {
    if (!consume(expected)) {
      fail(std::string("expected '") + expected + "'");
    }
  }

  JsonValue parse_value() {
    skip_space();
    if (position_ >= source_.size()) fail("unexpected end");
    switch (source_[position_]) {
      case '{': return parse_object();
      case '[': return parse_array();
      case '"': {
        JsonValue result;
        result.type = JsonValue::Type::String;
        result.string = parse_string();
        return result;
      }
      case 't': return parse_literal("true", JsonValue::Type::Bool, true);
      case 'f': return parse_literal("false", JsonValue::Type::Bool, false);
      case 'n': return parse_literal("null", JsonValue::Type::Null, false);
      default:
        if (source_[position_] == '-' ||
            std::isdigit(static_cast<unsigned char>(source_[position_]))) {
          JsonValue result;
          result.type = JsonValue::Type::Number;
          result.number = parse_number();
          return result;
        }
        fail("unexpected value");
    }
  }

  JsonValue parse_literal(const char* literal, JsonValue::Type type, bool value) {
    const std::string expected(literal);
    if (source_.compare(position_, expected.size(), expected) != 0) {
      fail("invalid literal");
    }
    position_ += expected.size();
    JsonValue result;
    result.type = type;
    result.boolean = value;
    return result;
  }

  std::string parse_string() {
    expect('"');
    std::string result;
    while (position_ < source_.size()) {
      char ch = source_[position_++];
      if (ch == '"') return result;
      if (ch != '\\') {
        result.push_back(ch);
        continue;
      }
      if (position_ >= source_.size()) fail("unfinished escape");
      char escaped = source_[position_++];
      switch (escaped) {
        case '"': result.push_back('"'); break;
        case '\\': result.push_back('\\'); break;
        case '/': result.push_back('/'); break;
        case 'b': result.push_back('\b'); break;
        case 'f': result.push_back('\f'); break;
        case 'n': result.push_back('\n'); break;
        case 'r': result.push_back('\r'); break;
        case 't': result.push_back('\t'); break;
        case 'u':
          // Input labels are normally ASCII. Preserve a Unicode escape as a
          // UTF-8 replacement rather than pulling a large Unicode dependency.
          for (int i = 0; i < 4; ++i) {
            if (position_ >= source_.size() ||
                !std::isxdigit(static_cast<unsigned char>(source_[position_]))) {
              fail("invalid unicode escape");
            }
            ++position_;
          }
          result.append("?");
          break;
        default: fail("unknown escape");
      }
    }
    fail("unterminated string");
  }

  double parse_number() {
    const std::size_t start = position_;
    if (source_[position_] == '-') ++position_;
    while (position_ < source_.size() &&
           std::isdigit(static_cast<unsigned char>(source_[position_]))) {
      ++position_;
    }
    if (position_ < source_.size() && source_[position_] == '.') {
      ++position_;
      while (position_ < source_.size() &&
             std::isdigit(static_cast<unsigned char>(source_[position_]))) {
        ++position_;
      }
    }
    if (position_ < source_.size() &&
        (source_[position_] == 'e' || source_[position_] == 'E')) {
      ++position_;
      if (position_ < source_.size() &&
          (source_[position_] == '+' || source_[position_] == '-')) ++position_;
      while (position_ < source_.size() &&
             std::isdigit(static_cast<unsigned char>(source_[position_]))) {
        ++position_;
      }
    }
    try {
      return std::stod(source_.substr(start, position_ - start));
    } catch (const std::exception&) {
      fail("invalid number");
    }
  }

  JsonValue parse_array() {
    expect('[');
    JsonValue result;
    result.type = JsonValue::Type::Array;
    skip_space();
    if (consume(']')) return result;
    while (true) {
      result.array.push_back(parse_value());
      skip_space();
      if (consume(']')) return result;
      expect(',');
    }
  }

  JsonValue parse_object() {
    expect('{');
    JsonValue result;
    result.type = JsonValue::Type::Object;
    skip_space();
    if (consume('}')) return result;
    while (true) {
      skip_space();
      if (position_ >= source_.size() || source_[position_] != '"') {
        fail("object key must be a string");
      }
      std::string key = parse_string();
      expect(':');
      result.object.emplace(std::move(key), parse_value());
      skip_space();
      if (consume('}')) return result;
      expect(',');
    }
  }

  std::string source_;
  std::size_t position_ = 0;
};

const JsonValue* field(const JsonValue& object, const char* key) {
  if (object.type != JsonValue::Type::Object) return nullptr;
  auto found = object.object.find(key);
  return found == object.object.end() ? nullptr : &found->second;
}

std::string string_field(const JsonValue& object, const char* key) {
  const JsonValue* value = field(object, key);
  return value && value->type == JsonValue::Type::String ? value->string : "";
}

int index_field(const JsonValue& object, int fallback) {
  const JsonValue* value = field(object, "index");
  if (!value || value->type != JsonValue::Type::Number) return fallback;
  return static_cast<int>(value->number);
}

std::string lower(std::string value) {
  for (char& ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

std::unordered_set<std::string> tokenize(const std::string& value) {
  std::unordered_set<std::string> result;
  std::string token;
  for (char raw : lower(value)) {
    const unsigned char ch = static_cast<unsigned char>(raw);
    if (std::isalnum(ch)) {
      token.push_back(static_cast<char>(ch));
    } else if (!token.empty()) {
      result.insert(token);
      token.clear();
    }
  }
  if (!token.empty()) result.insert(token);
  return result;
}

struct Model {
  double bias = 0.0;
  std::unordered_map<std::string, double> weights;
};

Model load_model(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open weights: " + path);
  Model model;
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream parts(line);
    std::string token;
    double value = 0.0;
    if (!(parts >> token)) continue;
    if (token[0] == '#') continue;
    if (token == "version" || token == "vocab_size") continue;
    if (!(parts >> value)) continue;
    if (token == "bias") model.bias = value;
    else model.weights[lower(token)] = value;
  }
  return model;
}

struct Candidate {
  int index = 0;
  std::string text;
  std::string content_desc;
  std::string resource_id;
  double score = 0.0;
  std::size_t original_position = 0;
};

std::vector<Candidate> load_candidates(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open screen JSON: " + path);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  JsonValue root = JsonParser(buffer.str()).parse();
  if (root.type != JsonValue::Type::Array) {
    throw std::runtime_error("screen JSON root must be an array");
  }
  std::vector<Candidate> candidates;
  candidates.reserve(root.array.size());
  for (std::size_t position = 0; position < root.array.size(); ++position) {
    const JsonValue& item = root.array[position];
    if (item.type != JsonValue::Type::Object) continue;
    Candidate candidate;
    candidate.original_position = position;
    candidate.index = index_field(item, static_cast<int>(position));
    candidate.text = string_field(item, "text");
    candidate.content_desc = string_field(item, "content_desc");
    if (candidate.content_desc.empty()) {
      candidate.content_desc = string_field(item, "contentDescription");
    }
    candidate.resource_id = string_field(item, "resource_id");
    if (candidate.resource_id.empty()) {
      candidate.resource_id = string_field(item, "resourceId");
    }
    candidates.push_back(std::move(candidate));
  }
  return candidates;
}

double score(const Candidate& candidate, const Model& model) {
  const std::string joined = candidate.text + " " + candidate.content_desc + " " +
                             candidate.resource_id;
  const auto tokens = tokenize(joined);
  double result = model.bias;
  for (const std::string& token : tokens) {
    const auto found = model.weights.find(token);
    if (found != model.weights.end()) result += found->second;
  }
  return result;
}

std::vector<Candidate> score_all(std::vector<Candidate> candidates, const Model& model) {
  for (Candidate& candidate : candidates) candidate.score = score(candidate, model);
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const Candidate& left, const Candidate& right) {
                     if (left.score != right.score) return left.score > right.score;
                     return left.original_position < right.original_position;
                   });
  return candidates;
}

std::size_t file_size(const std::string& path) {
  struct stat info {};
  return stat(path.c_str(), &info) == 0 ? static_cast<std::size_t>(info.st_size) : 0;
}

std::size_t peak_memory_kb() {
  std::ifstream status("/proc/self/status");
  std::string key;
  std::size_t value = 0;
  std::string unit;
  while (status >> key >> value >> unit) {
    if (key == "VmHWM:") return value;
  }
  struct rusage usage {};
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    // Linux/Android reports ru_maxrss in KiB. This is only a fallback when
    // /proc/self/status is unavailable to a restricted shell process.
    return static_cast<std::size_t>(usage.ru_maxrss);
  }
  return 0;
}

std::string display_label(const Candidate& candidate) {
  if (!candidate.text.empty()) return candidate.text;
  if (!candidate.content_desc.empty()) return candidate.content_desc;
  return candidate.resource_id;
}

void print_candidates(const std::vector<Candidate>& candidates) {
  std::cout << std::fixed << std::setprecision(4);
  for (const Candidate& candidate : candidates) {
    std::cout << "[" << candidate.index << "] " << display_label(candidate)
              << " score=" << candidate.score << "\n";
  }
  if (!candidates.empty()) {
    const Candidate& selected = candidates.front();
    std::cout << "SELECTED: index=" << selected.index << " text=\""
              << display_label(selected) << "\"\n";
  } else {
    std::cout << "SELECTED: none\n";
  }
}

void benchmark(const std::vector<Candidate>& candidates, const Model& model,
               int runs, const std::string& executable_path,
               const std::string& weights_path) {
  if (runs < 1) throw std::runtime_error("--benchmark must be positive");
  std::vector<double> micros;
  micros.reserve(static_cast<std::size_t>(runs));
  volatile double sink = 0.0;
  for (int iteration = 0; iteration < runs; ++iteration) {
    const auto started = std::chrono::steady_clock::now();
    std::vector<Candidate> ranked = score_all(candidates, model);
    const auto finished = std::chrono::steady_clock::now();
    sink += ranked.empty() ? 0.0 : ranked.front().score;
    micros.push_back(std::chrono::duration<double, std::micro>(finished - started).count());
  }
  std::sort(micros.begin(), micros.end());
  double average = 0.0;
  for (double value : micros) average += value;
  average /= static_cast<double>(micros.size());
  const auto percentile = [&](double fraction) {
    const std::size_t index = std::min(
        micros.size() - 1,
        static_cast<std::size_t>(std::ceil(fraction * micros.size())) - 1);
    return micros[index];
  };
  std::cout << std::fixed << std::setprecision(3)
            << "BENCHMARK candidates=" << candidates.size()
            << " runs=" << runs
            << " avg_us=" << average
            << " p50_us=" << percentile(0.50)
            << " p95_us=" << percentile(0.95)
            << " model_bytes=" << file_size(weights_path)
            << " executable_bytes=" << file_size(executable_path)
            << " peak_memory_kb~=" << peak_memory_kb()
            << "\n";
  (void)sink;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 5) {
    std::cerr << "usage: " << argv[0]
              << " screen.json weights.txt [--benchmark N]\n";
    return 2;
  }
  try {
    const std::string screen_path = argv[1];
    const std::string weights_path = argv[2];
    const std::vector<Candidate> candidates = load_candidates(screen_path);
    const Model model = load_model(weights_path);
    if (argc == 3) {
      print_candidates(score_all(candidates, model));
    } else {
      if (std::string(argv[3]) != "--benchmark") {
        throw std::runtime_error("unknown option: " + std::string(argv[3]));
      }
      benchmark(candidates, model, std::atoi(argv[4]), argv[0], weights_path);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << "\n";
    return 1;
  }
}
