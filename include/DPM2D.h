// HEADER FILE FOR DPM2D CLASS

#ifndef DPM2D_H
#define DPM2D_H

#include "constants.h"
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace std;
using std::vector;
using std::string;

class DPM2D
{
public:

	// constructor and deconstructor
	DPM2D(long nParticles, long dim, long nVertexPerParticle, long randomSeed = -1);
	~DPM2D();

    void initializeBox(double area);
    thrust::host_vector<double> getBoxSize();
};

#endif /* DPM2D_H */