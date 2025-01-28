#pragma once

#include "../base/config.h"

struct DiskParticleConfigDict : public PointParticleConfigDict {
public:
    DiskParticleConfigDict() {
        insert("type_name", "Disk");
    }
};