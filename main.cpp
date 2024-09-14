#include "include/defs.h"
#include "include/particle.h"
#include "include/disk.h"
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <functional>
#include <utility>
#include <thrust/host_vector.h>
#include <experimental/filesystem>

int main() {
    Disk disk;
    disk.multiplyData(3.0);  // Example usage
    disk.unimplementedFunction();
    return 0;
}