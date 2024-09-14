#ifndef DISK_H
#define DISK_H

#include "particle.h"

class Disk : public Particle {
public:
    Disk();  // Constructor
    ~Disk();  // Destructor

    // Override unimplemented function from Particle class
    void unimplementedFunction() override;

    // Override resizeData to resize the vector to size 15
    void resizeData() override;
};

#endif // DISK_H
