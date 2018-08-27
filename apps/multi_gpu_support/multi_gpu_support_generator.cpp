#include <Halide.h>

using namespace Halide;

class MultiGPUSupport : public Halide::Generator<MultiGPUSupport> {
    Input<Buffer<float>> input{"input", 2};
    
    Output<Buffer<float>> output{"output", 2};
    
public:
    void generate() {
        Var x("x"), y("y");

        output(x, y) = input(x, y) * 2.0f + 1.0f;
        
        Target target = get_target();
        target.set_feature(Target::UserContext);
        
        if (target.has_gpu_feature()) {
            Var bx("bx"), by("by"), tx("tx"), ty("ty");
            output.gpu_tile(x, y, bx, by, tx, ty, 16, 16).compute_root();
        }
    }
};

HALIDE_REGISTER_GENERATOR(MultiGPUSupport, multi_gpu_support)
