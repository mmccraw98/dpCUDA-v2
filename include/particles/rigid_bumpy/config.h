#pragma once

#include "../base/config.h"

struct RigidBumpyParticleConfigDict : public BaseParticleConfigDict {
public:
    RigidBumpyParticleConfigDict() {
        insert("type_name", "RigidBumpy");
    }
};
