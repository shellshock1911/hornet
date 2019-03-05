#include "Static/DegreeCentrality/DegreeCentrality.cuh"
#include <Graph/GraphStd.hpp>

namespace hornets_nest {

//------------------------------------------------------------------------------
///////////////
// OPERATORS //
///////////////

struct DCOperator {
    ctr_t* __restrict__ d_centralities;
    double norm_factor;

    OPERATOR(Vertex const &vertex) {
        d_centralities[std::move(vertex.id())] = \
                       std::move(vertex.degree() * norm_factor);
    }
};


DegreeCentrality::DegreeCentrality(HornetGraph& hornet) :
                                   norm_factor((1.0 / (hornet.nV() - 1.0))),
                                   StaticAlgorithm(hornet) {
    gpu::allocate(d_centralities, hornet.nV());
    reset();
}

DegreeCentrality::~DegreeCentrality() {
    gpu::free(d_centralities);
}

void DegreeCentrality::reset() {
    auto centralities = d_centralities;
    forAllnumV(hornet, [=] __device__ (int const i){ centralities[i] = 0.0; } );
}

void DegreeCentrality::run() {
    forAllVertices(hornet,
                   DCOperator { d_centralities, norm_factor });
}

void DegreeCentrality::release() {
    gpu::free(d_centralities);
    d_centralities = nullptr;
}

} // namespace hornets_nest
