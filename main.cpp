#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dima_laser.hpp"

namespace {

struct InputData {
  std::vector<double> xl;
  std::vector<double> xr;
  std::vector<std::complex<double>> eps;
};

std::string trim(const std::string& s) {
  std::size_t i = 0;
  while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])) != 0) {
    ++i;
  }
  std::size_t j = s.size();
  while (j > i && std::isspace(static_cast<unsigned char>(s[j - 1])) != 0) {
    --j;
  }
  return s.substr(i, j - i);
}

std::string normalize_delims(std::string line) {
  for (char& c : line) {
    if (c == ',' || c == ';' || c == '\t') {
      c = ' ';
    }
  }
  return line;
}

InputData load_from_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open input file: " + path);
  }

  InputData d;
  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    const std::string t = trim(line);
    if (t.empty()) {
      continue;
    }
    if (t[0] == '#') {
      continue;
    }
    std::istringstream iss(normalize_delims(t));
    double xl = 0.0;
    double xr = 0.0;
    double er = 0.0;
    double ei = 0.0;
    if (iss >> xl >> xr >> er >> ei) {
      d.xl.push_back(xl);
      d.xr.push_back(xr);
      d.eps.emplace_back(er, ei);
      continue;
    }

    // If parse failed, allow text header lines (e.g. xl,xr,eps_real,eps_im),
    // but do not reject scientific notation (already handled above).
    bool has_letter = false;
    for (char c : t) {
      if (std::isalpha(static_cast<unsigned char>(c)) != 0) {
        has_letter = true;
        break;
      }
    }
    if (has_letter) {
      continue;
    }

    throw std::runtime_error("Bad row at line " + std::to_string(line_no) +
                             ". Expected: xl xr eps_real eps_im");
  }

  if (d.eps.empty()) {
    throw std::runtime_error("Input file has no numeric rows");
  }
  return d;
}

InputData default_manual_case() {
  InputData d;
  d.xl = {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0,
          100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0};
  d.xr = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
          110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0};
  d.eps.resize(d.xl.size(), std::complex<double>(-19.5, 491.0));
  return d;
}

void validate_data(const InputData& d) {
  if (d.xl.size() != d.xr.size() || d.xl.size() != d.eps.size()) {
    throw std::runtime_error("Size mismatch: xl/xr/eps must have same size");
  }
  for (std::size_t i = 0; i < d.xl.size(); ++i) {
    if (!(d.xr[i] > d.xl[i])) {
      throw std::runtime_error("Invalid cell at i=" + std::to_string(i) + ": xr must be > xl");
    }
    if (i > 0 && d.xl[i] < d.xl[i - 1]) {
      throw std::runtime_error("xl must be non-decreasing");
    }
  }
}

void print_usage(const char* prog) {
  std::cout << "Usage:\n";
  std::cout << "  " << prog << " [--lambda-um VALUE] [--input FILE] [--print-q]\n\n";
  std::cout << "Options:\n";
  std::cout << "  --input FILE      Read columns: xl xr eps_real eps_im (nm, nm, -, -)\n";
  std::cout << "  --lambda-um VAL   Wavelength in um (default: 10.6)\n";
  std::cout << "  --print-q         Print per-cell q[m]\n";
  std::cout << "  --help            Show this message\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string input_path;
    double lambda_um = 10.6;
    bool print_q = false;

    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--help" || a == "-h") {
        print_usage(argv[0]);
        return 0;
      }
      if (a == "--print-q") {
        print_q = true;
        continue;
      }
      if (a == "--input") {
        if (i + 1 >= argc) {
          throw std::runtime_error("--input requires a file path");
        }
        input_path = argv[++i];
        continue;
      }
      if (a == "--lambda-um") {
        if (i + 1 >= argc) {
          throw std::runtime_error("--lambda-um requires a numeric value");
        }
        lambda_um = std::stod(argv[++i]);
        continue;
      }
      throw std::runtime_error("Unknown argument: " + a);
    }

    const InputData data = input_path.empty() ? default_manual_case() : load_from_file(input_path);
    validate_data(data);

    double R = 0.0;
    double T = 0.0;
    std::vector<double> q;
    dima_laser(data.xl, data.xr, data.eps, lambda_um, R, T, q);

    const double Q = std::accumulate(q.begin(), q.end(), 0.0);
    const double balance = R + T + Q;
    const double err = std::abs(balance - 1.0);
    const double tol = 1.0e-8;

    std::cout << std::setprecision(16);
    std::cout << "cells = " << q.size() << "\n";
    std::cout << "lambda_um = " << lambda_um << "\n";
    if (!input_path.empty()) {
      std::cout << "input = " << input_path << "\n";
    } else {
      std::cout << "input = internal manual case\n";
    }
    std::cout << "R = " << R << "\n";
    std::cout << "T = " << T << "\n";
    std::cout << "Q = " << Q << " (sum of q[m])\n";
    std::cout << "R + T + Q = " << balance << "\n";
    std::cout << "|R + T + Q - 1| = " << err << "\n";
    std::cout << "Conservation: " << ((err <= tol) ? "PASS" : "FAIL") << " (tol=" << tol << ")\n";

    if (print_q) {
      std::cout << "\nPer-cell q[m]:\n";
      for (std::size_t m = 0; m < q.size(); ++m) {
        std::cout << "  " << m << " " << q[m] << "\n";
      }
    }

    return (err <= 1.0e-6) ? 0 : 2;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}
