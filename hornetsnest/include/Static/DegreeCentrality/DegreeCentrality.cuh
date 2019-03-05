#pragma once

#include "HornetAlg.hpp"

namespace hornets_nest {

using HornetGraph = gpu::Csr<EMPTY, EMPTY>;

using ctr_t = double;

class DegreeCentrality : public StaticAlgorithm<HornetGraph> {
public:
    DegreeCentrality(HornetGraph& hornet);
    ~DegreeCentrality();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override { return true; }

private:
    ctr_t* d_centralities { nullptr };
    double const norm_factor;
};

} // namespace hornets_nest
