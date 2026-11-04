// PyTorch binding: CPU kernels always, CUDA kernels when built with EVKIT_WITH_CUDA.
// Ops take contiguous float32 tensors; device dispatch is by the input tensor's device.
#include <torch/extension.h>

#include <memory>

#include "evkit/evb.hpp"
#include "evkit/events.hpp"
#include "evkit/ssm_cpu.hpp"
#include "evkit/tokenize.hpp"
#ifdef EVKIT_WITH_CUDA
#include <ATen/cuda/CUDAContext.h>

#include "evkit/cuda_ops.hpp"
#endif

namespace {

using at::Tensor;

void need(const Tensor& t, const char* name) {
  TORCH_CHECK(t.defined() && t.is_contiguous() && t.scalar_type() == at::kFloat, name, " must be a contiguous float32 tensor");
}
const float* opt_ptr(const c10::optional<Tensor>& t) {
  return (t.has_value() && t->defined() && t->numel()) ? t->data_ptr<float>() : nullptr;
}
#ifdef EVKIT_WITH_CUDA
void* stream() { return at::cuda::getCurrentCUDAStream().stream(); }
#endif

// --------------------------------------------------------------------------------------- scan
std::vector<Tensor> scan_fwd(Tensor v, Tensor dt, Tensor rate, Tensor Bm, Tensor Cm, Tensor skip,
                             c10::optional<Tensor> h0, int64_t n_save, bool fast_exp, int64_t mode) {
  for (auto* p : {&v, &dt, &rate, &Bm, &Cm, &skip}) need(*p, "scan_fwd input");
  const int64_t B = v.size(0), L = v.size(1), D = v.size(2), N = rate.size(1);
  TORCH_CHECK(dt.sizes() == at::IntArrayRef({B, L}) && rate.size(0) == D && n_save >= 0 && n_save <= D, "scan_fwd: bad shapes");
  const evkit::ScanShape s{B, L, D, N, n_save};
  auto y = at::empty({B, L, D}, v.options());
  auto H = at::empty({B, L, n_save, N}, v.options());
  auto h_last = at::empty({B, D, N}, v.options());
  Tensor ckpt;
  if (v.is_cuda()) {
#ifdef EVKIT_WITH_CUDA
    ckpt = at::empty({B, evkit::gpu::num_chunks(L), D, N}, v.options());
    auto ws = at::empty({static_cast<int64_t>(evkit::gpu::scan_fwd_workspace_floats(s))}, v.options());
    evkit::gpu::scan_fwd(s, v.data_ptr<float>(), dt.data_ptr<float>(), rate.data_ptr<float>(), Bm.data_ptr<float>(),
                         Cm.data_ptr<float>(), skip.data_ptr<float>(), opt_ptr(h0), y.data_ptr<float>(),
                         n_save ? H.data_ptr<float>() : nullptr, h_last.data_ptr<float>(), ckpt.data_ptr<float>(),
                         ws.data_ptr<float>(), fast_exp, static_cast<evkit::gpu::ScanMode>(mode), stream());
#else
    TORCH_CHECK(false, "evkit was built without CUDA");
#endif
  } else {
    ckpt = at::empty({0}, v.options());
    evkit::cpu::scan_fwd(s, v.data_ptr<float>(), dt.data_ptr<float>(), rate.data_ptr<float>(), Bm.data_ptr<float>(),
                         Cm.data_ptr<float>(), skip.data_ptr<float>(), opt_ptr(h0), y.data_ptr<float>(),
                         n_save ? H.data_ptr<float>() : nullptr, h_last.data_ptr<float>());
  }
  return {y, H, h_last, ckpt};
}

std::vector<Tensor> scan_bwd(Tensor v, Tensor dt, Tensor rate, Tensor Bm, Tensor Cm, Tensor skip,
                             c10::optional<Tensor> h0, Tensor ckpt, Tensor gy, c10::optional<Tensor> gH,
                             c10::optional<Tensor> g_hlast, int64_t n_save, bool fast_exp, int64_t mode) {
  for (auto* p : {&v, &dt, &rate, &Bm, &Cm, &skip, &gy}) need(*p, "scan_bwd input");
  const int64_t B = v.size(0), L = v.size(1), D = v.size(2), N = rate.size(1);
  const evkit::ScanShape s{B, L, D, N, n_save};
  auto gv = at::empty_like(v), gdt = at::empty_like(dt), grate = at::empty_like(rate), gB = at::empty_like(Bm),
       gC = at::empty_like(Cm), gskip = at::empty_like(skip);
  const bool want_h0 = h0.has_value() && h0->defined() && h0->numel();
  Tensor gh0 = want_h0 ? at::empty_like(*h0) : at::empty({0}, v.options());
  if (v.is_cuda()) {
#ifdef EVKIT_WITH_CUDA
    auto ws = at::empty({static_cast<int64_t>(evkit::gpu::scan_bwd_workspace_floats(s))}, v.options());
    evkit::gpu::scan_bwd(s, v.data_ptr<float>(), dt.data_ptr<float>(), rate.data_ptr<float>(), Bm.data_ptr<float>(),
                         Cm.data_ptr<float>(), skip.data_ptr<float>(), opt_ptr(h0), ckpt.data_ptr<float>(),
                         gy.data_ptr<float>(), opt_ptr(gH), opt_ptr(g_hlast), gv.data_ptr<float>(), gdt.data_ptr<float>(),
                         grate.data_ptr<float>(), gB.data_ptr<float>(), gC.data_ptr<float>(), gskip.data_ptr<float>(),
                         want_h0 ? gh0.data_ptr<float>() : nullptr, ws.data_ptr<float>(), fast_exp,
                         static_cast<evkit::gpu::ScanMode>(mode), stream());
#else
    TORCH_CHECK(false, "evkit was built without CUDA");
#endif
  } else {
    evkit::cpu::scan_bwd(s, v.data_ptr<float>(), dt.data_ptr<float>(), rate.data_ptr<float>(), Bm.data_ptr<float>(),
                         Cm.data_ptr<float>(), skip.data_ptr<float>(), opt_ptr(h0), gy.data_ptr<float>(), opt_ptr(gH),
                         opt_ptr(g_hlast), gv.data_ptr<float>(), gdt.data_ptr<float>(), grate.data_ptr<float>(),
                         gB.data_ptr<float>(), gC.data_ptr<float>(), gskip.data_ptr<float>(),
                         want_h0 ? gh0.data_ptr<float>() : nullptr);
  }
  return {gv, gdt, grate, gB, gC, gskip, gh0};
}

// --------------------------------------------------------------------------------------- readout
Tensor readout_fwd(Tensor H, Tensor tau, Tensor rate, Tensor Cm, bool fast_exp) {
  for (auto* p : {&H, &tau, &rate, &Cm}) need(*p, "readout_fwd input");
  const evkit::ReadoutShape s{H.size(0), H.size(1), tau.size(2), H.size(2), H.size(3)};
  TORCH_CHECK(tau.size(0) == s.B && tau.size(1) == s.L && rate.size(0) == s.n && rate.size(1) == s.N, "readout_fwd: bad shapes");
  auto Y = at::empty({s.B, s.L, s.S, s.n}, H.options());
  if (H.is_cuda()) {
#ifdef EVKIT_WITH_CUDA
    evkit::gpu::readout_fwd(s, H.data_ptr<float>(), tau.data_ptr<float>(), rate.data_ptr<float>(), Cm.data_ptr<float>(),
                            Y.data_ptr<float>(), fast_exp, stream());
#else
    TORCH_CHECK(false, "evkit was built without CUDA");
#endif
  } else {
    evkit::cpu::readout_fwd(s, H.data_ptr<float>(), tau.data_ptr<float>(), rate.data_ptr<float>(), Cm.data_ptr<float>(),
                            Y.data_ptr<float>());
  }
  return Y;
}

std::vector<Tensor> readout_bwd(Tensor H, Tensor tau, Tensor rate, Tensor Cm, Tensor gY, bool fast_exp) {
  for (auto* p : {&H, &tau, &rate, &Cm, &gY}) need(*p, "readout_bwd input");
  const evkit::ReadoutShape s{H.size(0), H.size(1), tau.size(2), H.size(2), H.size(3)};
  auto gH = at::empty_like(H), gtau = at::empty_like(tau), grate = at::empty_like(rate), gC = at::empty_like(Cm);
  if (H.is_cuda()) {
#ifdef EVKIT_WITH_CUDA
    auto ws = at::empty({static_cast<int64_t>(evkit::gpu::readout_bwd_workspace_floats(s))}, H.options());
    evkit::gpu::readout_bwd(s, H.data_ptr<float>(), tau.data_ptr<float>(), rate.data_ptr<float>(), Cm.data_ptr<float>(),
                            gY.data_ptr<float>(), gH.data_ptr<float>(), gtau.data_ptr<float>(), grate.data_ptr<float>(),
                            gC.data_ptr<float>(), ws.data_ptr<float>(), fast_exp, stream());
#else
    TORCH_CHECK(false, "evkit was built without CUDA");
#endif
  } else {
    evkit::cpu::readout_bwd(s, H.data_ptr<float>(), tau.data_ptr<float>(), rate.data_ptr<float>(), Cm.data_ptr<float>(),
                            gY.data_ptr<float>(), gH.data_ptr<float>(), gtau.data_ptr<float>(), grate.data_ptr<float>(),
                            gC.data_ptr<float>());
  }
  return {gH, gtau, grate, gC};
}

// --------------------------------------------------------------------------------------- tokenize
std::vector<Tensor> tokenize(Tensor t, Tensor x, Tensor y, Tensor p, int64_t width, int64_t height, int64_t grid,
                             int64_t sub, int64_t seed, int64_t index_offset, int64_t t_prev) {
  TORCH_CHECK(t.scalar_type() == at::kLong && x.scalar_type() == at::kShort && y.scalar_type() == at::kShort &&
                  p.scalar_type() == at::kByte, "tokenize expects t:int64, x/y:int16, p:uint8");
  t = t.contiguous(); x = x.contiguous(); y = y.contiguous(); p = p.contiguous();
  const int64_t n = t.numel();
  const evkit::TokenizeParams prm{static_cast<int>(width), static_cast<int>(height), static_cast<int>(grid),
                                  static_cast<int>(sub), static_cast<uint64_t>(seed),
                                  static_cast<uint64_t>(index_offset), t_prev};
  auto io = t.options().dtype(at::kInt), fo = t.options().dtype(at::kFloat);
  auto patch = at::empty({n}, io), sb = at::empty({n}, io), pol = at::empty({n}, io);
  auto u = at::empty({n}, fo), v = at::empty({n}, fo), dt = at::empty({n}, fo);
  auto* xs = reinterpret_cast<const uint16_t*>(x.data_ptr<int16_t>());
  auto* ys = reinterpret_cast<const uint16_t*>(y.data_ptr<int16_t>());
  if (t.is_cuda()) {
#ifdef EVKIT_WITH_CUDA
    evkit::gpu::tokenize(t.data_ptr<int64_t>(), xs, ys, p.data_ptr<uint8_t>(), n, prm, patch.data_ptr<int32_t>(),
                         sb.data_ptr<int32_t>(), pol.data_ptr<int32_t>(), u.data_ptr<float>(), v.data_ptr<float>(),
                         dt.data_ptr<float>(), stream());
#else
    TORCH_CHECK(false, "evkit was built without CUDA");
#endif
  } else {
    evkit::tokenize_cpu(t.data_ptr<int64_t>(), xs, ys, p.data_ptr<uint8_t>(), n, prm, patch.data_ptr<int32_t>(),
                        sb.data_ptr<int32_t>(), pol.data_ptr<int32_t>(), u.data_ptr<float>(), v.data_ptr<float>(),
                        dt.data_ptr<float>());
  }
  return {patch, sb, pol, u, v, dt};
}

// --------------------------------------------------------------------------------------- I/O
template <class V>
Tensor to_tensor(const V& v, at::ScalarType st) {
  auto out = at::empty({static_cast<int64_t>(v.size())}, at::TensorOptions().dtype(st));
  std::memcpy(out.data_ptr(), v.data(), v.size() * sizeof(typename V::value_type));
  return out;
}

py::dict decode_file(const std::string& path, int64_t n_threads, const std::string& fmt) {
  evkit::RawFormat hint = fmt == "evt2" ? evkit::RawFormat::kEvt2 : fmt == "evt3" ? evkit::RawFormat::kEvt3
                        : fmt == "dat" ? evkit::RawFormat::kDat : evkit::RawFormat::kUnknown;
  evkit::EventBuffer ev;
  evkit::RawHeader h;
  {
    py::gil_scoped_release nogil;
    h = evkit::decode_file(path, ev, static_cast<int>(n_threads), hint);
    ev.sort_by_time(static_cast<int>(n_threads));
  }
  py::dict d;
  d["t"] = to_tensor(ev.t, at::kLong);
  d["x"] = to_tensor(ev.x, at::kShort);
  d["y"] = to_tensor(ev.y, at::kShort);
  d["p"] = to_tensor(ev.p, at::kByte);
  d["width"] = h.width;
  d["height"] = h.height;
  d["header"] = h.text;
  return d;
}

evkit::EventBuffer from_tensors(Tensor t, Tensor x, Tensor y, Tensor p) {
  t = t.contiguous().cpu(); x = x.contiguous().cpu(); y = y.contiguous().cpu(); p = p.contiguous().cpu();
  const int64_t n = t.numel();
  evkit::EventBuffer ev;
  evkit::resize(ev, n);
  std::memcpy(ev.t.data(), t.data_ptr<int64_t>(), n * 8);
  std::memcpy(ev.x.data(), x.data_ptr<int16_t>(), n * 2);
  std::memcpy(ev.y.data(), y.data_ptr<int16_t>(), n * 2);
  std::memcpy(ev.p.data(), p.data_ptr<uint8_t>(), n);
  return ev;
}

void write_evb(const std::string& path, Tensor t, Tensor x, Tensor y, Tensor p, int64_t width, int64_t height,
               const std::string& sensor) {
  auto ev = from_tensors(t, x, y, p);
  py::gil_scoped_release nogil;
  evkit::write_evb(path, std::move(ev), static_cast<int>(width), static_cast<int>(height), sensor);
}

struct PyEvb {
  std::shared_ptr<evkit::EvbReader> r;
  explicit PyEvb(const std::string& path) : r(std::make_shared<evkit::EvbReader>(path)) {}
  py::dict slice(int64_t i0, int64_t i1) const {
    i0 = std::max<int64_t>(0, i0);
    i1 = std::min<int64_t>(static_cast<int64_t>(r->size()), i1);
    const int64_t n = std::max<int64_t>(0, i1 - i0);
    auto mk = [&](const void* src, int64_t elem, at::ScalarType st) {
      auto out = at::empty({n}, at::TensorOptions().dtype(st));
      std::memcpy(out.data_ptr(), static_cast<const uint8_t*>(src) + i0 * elem, n * elem);
      return out;
    };
    py::dict d;
    d["t"] = mk(r->t(), 8, at::kLong);
    d["x"] = mk(r->x(), 2, at::kShort);
    d["y"] = mk(r->y(), 2, at::kShort);
    d["p"] = mk(r->p(), 1, at::kByte);
    return d;
  }
  py::dict window(int64_t t0, int64_t t1) const {
    auto [a, b] = r->time_range(t0, t1);
    return slice(static_cast<int64_t>(a), static_cast<int64_t>(b));
  }
  py::dict info() const {
    const auto& h = r->header();
    py::dict d;
    d["n_events"] = h.n_events; d["width"] = h.width; d["height"] = h.height;
    d["t0"] = h.t0; d["t1"] = h.t1; d["sensor"] = std::string(h.sensor);
    return d;
  }
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "evkit: event-stream kernels (event-driven SSM scan, decayed readout, tokenisation, raw decoders)";
#ifdef EVKIT_WITH_CUDA
  m.attr("with_cuda") = true;
#else
  m.attr("with_cuda") = false;
#endif
  m.def("scan_fwd", &scan_fwd, py::arg("v"), py::arg("dt"), py::arg("rate"), py::arg("Bm"), py::arg("Cm"), py::arg("skip"),
        py::arg("h0"), py::arg("n_save"), py::arg("fast_exp") = true, py::arg("mode") = 0);
  m.def("scan_bwd", &scan_bwd);
  m.def("readout_fwd", &readout_fwd, py::arg("H"), py::arg("tau"), py::arg("rate"), py::arg("Cm"), py::arg("fast_exp") = true);
  m.def("readout_bwd", &readout_bwd);
  m.def("tokenize", &tokenize, py::arg("t"), py::arg("x"), py::arg("y"), py::arg("p"), py::arg("width"), py::arg("height"),
        py::arg("grid"), py::arg("sub"), py::arg("seed") = 0, py::arg("index_offset") = 0, py::arg("t_prev") = -1);
  m.def("decode_file", &decode_file, py::arg("path"), py::arg("n_threads") = 0, py::arg("format") = "");
  m.def("write_evb", &write_evb, py::arg("path"), py::arg("t"), py::arg("x"), py::arg("y"), py::arg("p"),
        py::arg("width"), py::arg("height"), py::arg("sensor") = "");
  py::class_<PyEvb>(m, "EvbReader")
      .def(py::init<const std::string&>())
      .def("__len__", [](const PyEvb& e) { return e.r->size(); })
      .def("slice", &PyEvb::slice)
      .def("window", &PyEvb::window)
      .def("info", &PyEvb::info);
}
