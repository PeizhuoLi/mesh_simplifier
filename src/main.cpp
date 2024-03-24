#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <igl/decimate.h>
#include <igl/decimate_trivial_callbacks.h>
#include <igl/edge_flaps.h>
#include <igl/slice.h>
#include <igl/slice_mask.h>


#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;


bool my_collapse_edge(
        int e,
        Eigen::MatrixXd &V,
        Eigen::MatrixXi &F,
        Eigen::MatrixXi &E,
        Eigen::VectorXi &EMAP,
        Eigen::MatrixXi &EF,
        Eigen::MatrixXi &EI,
        Eigen::MatrixXd &C,
        int &e1,
        int &e2,
        int &f1,
        int &f2) {
    using namespace Eigen;
    using namespace igl;
    std::tuple<double, int, int> p;

    // Why is this computed up here?
    // If we just need original face neighbors of edge, could we gather that more
    // directly than gathering face neighbors of each vertex?
    std::vector<int> /*Nse,*/Nsf, Nsv;
    circulation(e, true, F, EMAP, EF, EI,/*Nse,*/Nsv, Nsf);
    std::vector<int> /*Nde,*/Ndf, Ndv;
    circulation(e, false, F, EMAP, EF, EI,/*Nde,*/Ndv, Ndf);


    bool collapsed = true;

    collapsed = collapse_edge(
            e, C.row(e),
            Nsv, Nsf, Ndv, Ndf,
            V, F, E, EMAP, EF, EI, e1, e2, f1, f2);

    if (collapsed) {
        // TODO: visits edges multiple times, ~150% more updates than should
        //
        // update local neighbors
        // loop over original face neighbors
        //
        // Can't use previous computed Nse and Nde because those refer to EMAP
        // before it was changed...
        std::vector<int> Nf;
        Nf.reserve(Nsf.size() + Ndf.size()); // preallocate memory
        Nf.insert(Nf.end(), Nsf.begin(), Nsf.end());
        Nf.insert(Nf.end(), Ndf.begin(), Ndf.end());
        // https://stackoverflow.com/a/1041939/148668
        std::sort(Nf.begin(), Nf.end());
        Nf.erase(std::unique(Nf.begin(), Nf.end()), Nf.end());
        // Collect all edges that must be updated
        std::vector<int> Ne;
        Ne.reserve(3 * Nf.size());
        for (auto &n: Nf) {
            if (F(n, 0) != IGL_COLLAPSE_EDGE_NULL ||
                F(n, 1) != IGL_COLLAPSE_EDGE_NULL ||
                F(n, 2) != IGL_COLLAPSE_EDGE_NULL) {
                for (int v = 0; v < 3; v++) {
                    // get edge id
                    const int ei = EMAP(v * F.rows() + n);
                    Ne.push_back(ei);
                }
            }
        }
        // Only process edge once
        std::sort(Ne.begin(), Ne.end());
        Ne.erase(std::unique(Ne.begin(), Ne.end()), Ne.end());
        for (auto &ei: Ne) {
            // compute cost and potential placement
            double cost;
            RowVectorXd place;
            shortest_edge_and_midpoint(ei, V, F, E, EMAP, EF, EI, cost, place);
            // Replace in queue
            C.row(ei) = place;
        }
    }
    return collapsed;
}


auto my_decimate_by_sequence(const Eigen::MatrixXd &OV,
                             const Eigen::MatrixXi &OF,
                             const std::vector<int> &q,
                             Eigen::MatrixXd &U,
                             Eigen::MatrixXi &G,
                             Eigen::VectorXi &J,
                             Eigen::VectorXi &I) {
    // Decimate 1
    using namespace igl;
    using namespace Eigen;
    using namespace std;
    // Working copies
    Eigen::MatrixXd V = OV;
    Eigen::MatrixXi F = OF;
    VectorXi EMAP;
    MatrixXi E, EF, EI;
    edge_flaps(F, E, EMAP, EF, EI);
    {
        Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> BF;
        Eigen::Array<bool, Eigen::Dynamic, 1> BE;
        if (!is_edge_manifold(F, E.rows(), EMAP, BF, BE)) {
            return false;
        }
    }

    // Could reserve with https://stackoverflow.com/a/29236236/148668
    // If an edge were collapsed, we'd collapse it to these points:
    MatrixXd C(E.rows(), V.cols());
    // Pushing into a vector then using constructor was slower. Maybe using
    // std::move + make_heap would squeeze out something?

    igl::parallel_for(E.rows(), [&](const int e) {
                          double cost = e;
                          RowVectorXd p(1, 3);
                          shortest_edge_and_midpoint(e, V, F, E, EMAP, EF, EI, cost, p);
                          C.row(e) = p;
                      },
                      10000
    );

    for (auto &&e: q) {
        int e1, e2, f1, f2;
        my_collapse_edge(e, V, F, E, EMAP, EF, EI, C, e1, e2, f1, f2);
    }
    // remove all IGL_COLLAPSE_EDGE_NULL faces
    MatrixXi F2(F.rows(), 3);
    J.resize(F.rows());
    int m = 0;
    for (int f = 0; f < F.rows(); f++) {
        if (F(f, 0) != IGL_COLLAPSE_EDGE_NULL ||
            F(f, 1) != IGL_COLLAPSE_EDGE_NULL ||
            F(f, 2) != IGL_COLLAPSE_EDGE_NULL) {
            F2.row(m) = F.row(f);
            J(m) = f;
            m++;
        }
    }
    F2.conservativeResize(m, F2.cols());
    J.conservativeResize(m);
    VectorXi _1;
    igl::remove_unreferenced(V, F2, U, G, _1, I);
    return true;
}

auto create_decimate_sequence_wrapped(const py::EigenDRef<Eigen::MatrixXd> &V, const py::EigenDRef<Eigen::MatrixXi> &F,
                                      const size_t max_m) {
    Eigen::MatrixXd U;
    Eigen::MatrixXi G;
    Eigen::VectorXi J, I;
    std::vector<int> decimate_seq;
    //    auto res_flag = igl::decimate(V, F, max_m, U, G, J, I);
    // Original number of faces
    const int orig_m = F.rows();
    // Tracking number of faces
    int m = F.rows();
    typedef Eigen::MatrixXd DerivedV;
    typedef Eigen::MatrixXi DerivedF;
    DerivedV VO;
    DerivedF FO;
    igl::connect_boundary_to_infinity(V, F, VO, FO);
    Eigen::VectorXi EMAP;
    Eigen::MatrixXi E, EF, EI;
    igl::edge_flaps(FO, E, EMAP, EF, EI);
    // decimate will not work correctly on non-edge-manifold meshes. By extension
    // this includes meshes with non-manifold vertices on the boundary since these
    // will create a non-manifold edge when connected to infinity.
    {
        Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> BF;
        Eigen::Array<bool, Eigen::Dynamic, 1> BE;
        if (!igl::is_edge_manifold(FO, E.rows(), EMAP, BF, BE)) {
            throw std::runtime_error("Mesh is not edge-manifold");
        }
    }
    igl::decimate_pre_collapse_callback always_try;
    igl::decimate_post_collapse_callback construct_sequence;
    igl::decimate_trivial_callbacks(always_try, construct_sequence);
    construct_sequence = [&decimate_seq](
            const Eigen::MatrixXd &,/*V*/
            const Eigen::MatrixXi &,/*F*/
            const Eigen::MatrixXi &,/*E*/
            const Eigen::VectorXi &,/*EMAP*/
            const Eigen::MatrixXi &,/*EF*/
            const Eigen::MatrixXi &,/*EI*/
            const igl::min_heap<std::tuple<double, int, int> > &,/*Q*/
            const Eigen::VectorXi &,/*EQ*/
            const Eigen::MatrixXd &,/*C*/
            const int e,/*e*/
            const int,/*e1*/
            const int,/*e2*/
            const int,/*f1*/
            const int,/*f2*/
            const bool collapsed                                  /*collapsed*/
    ) -> void { if (collapsed) decimate_seq.push_back(e); };
    bool res_flag = igl::decimate(
            VO,
            FO,
            igl::shortest_edge_and_midpoint,
            igl::max_faces_stopping_condition(m, orig_m, max_m),
            always_try,
            construct_sequence,
            E,
            EMAP,
            EF,
            EI,
            U,
            G,
            J,
            I);
    const Eigen::Array<bool, Eigen::Dynamic, 1> keep = (J.array() < orig_m);
    igl::slice_mask(Eigen::MatrixXi(G), keep, 1, G);
    igl::slice_mask(Eigen::VectorXi(J), keep, 1, J);
    Eigen::VectorXi _1, I2;
    igl::remove_unreferenced(Eigen::MatrixXd(U), Eigen::MatrixXi(G), U, G, _1, I2);
    igl::slice(Eigen::VectorXi(I), I2, 1, I);
    return std::make_tuple(res_flag, U, G, decimate_seq);
}

auto decimate_by_sequence_wrapped(const py::EigenDRef<Eigen::MatrixXd> &V, const py::EigenDRef<Eigen::MatrixXi> &F,
                                  const std::vector<int> &q) {
    Eigen::MatrixXd U;
    Eigen::MatrixXi G;
    Eigen::VectorXi J, I;
    // Original number of faces
    const int orig_m = F.rows();
    // Tracking number of faces
    typedef Eigen::MatrixXd DerivedV;
    typedef Eigen::MatrixXi DerivedF;
    DerivedV VO;
    DerivedF FO;
    igl::connect_boundary_to_infinity(V, F, VO, FO);
    Eigen::VectorXi EMAP;
    Eigen::MatrixXi E, EF, EI;
    igl::edge_flaps(FO, E, EMAP, EF, EI);
    // decimate will not work correctly on non-edge-manifold meshes. By extension
    // this includes meshes with non-manifold vertices on the boundary since these
    // will create a non-manifold edge when connected to infinity.
    {
        Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> BF;
        Eigen::Array<bool, Eigen::Dynamic, 1> BE;
        if (!igl::is_edge_manifold(FO, E.rows(), EMAP, BF, BE)) {
            throw std::runtime_error("Mesh is not edge-manifold");
        }
    }
    bool res_flag = my_decimate_by_sequence(
            VO,
            FO,
            q,
            U,
            G,
            J,
            I);
    const Eigen::Array<bool, Eigen::Dynamic, 1> keep = (J.array() < orig_m);
    igl::slice_mask(Eigen::MatrixXi(G), keep, 1, G);
    igl::slice_mask(Eigen::VectorXi(J), keep, 1, J);
    Eigen::VectorXi _1, I2;
    igl::remove_unreferenced(Eigen::MatrixXd(U), Eigen::MatrixXi(G), U, G, _1, I2);
    igl::slice(Eigen::VectorXi(I), I2, 1, I);
    return std::make_tuple(U, G);
}

PYBIND11_MODULE(mesh_simplifier, m) {
    m.doc() = R"pbdoc(
        Pybind11 wrapped mesh simplifier of libigl
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("create_decimate_sequence", &create_decimate_sequence_wrapped,
          "A function which creates a decimation sequence",
          py::arg("V"), py::arg("F"), py::arg("max_m"));

    m.def("decimate_by_sequence", &decimate_by_sequence_wrapped,
          "A function which decimates a mesh by a given mesh collapes sequence",
          py::arg("V"), py::arg("F"), py::arg("q"));

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
