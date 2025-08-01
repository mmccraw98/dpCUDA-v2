#include <stdio.h>
#include <cmath>
#include "../../../include/constants.h"
#include "../../../include/particles/base/kernels.cuh"
#include "../../../include/particles/rigid_bumpy/kernels.cuh"

__constant__ double d_vertex_radius;

__constant__ long* d_num_vertex_neighbors_ptr;
__constant__ long* d_vertex_neighbor_list_ptr;
__constant__ long d_max_vertex_neighbors_allocated;

__constant__ long* d_particle_start_index_ptr;
__constant__ long* d_num_vertices_in_particle_ptr;
__constant__ long* d_vertex_particle_index_ptr;

// ----------------------------------------------------------------------
// ----------------------- Dynamics and Updates -------------------------
// ----------------------------------------------------------------------

__global__ void kernelUpdateRigidPositions(
    double* last_positions_x, double* last_positions_y,
    double* positions_x, double* positions_y, double* angles,
    double* delta_x, double* delta_y, double* angle_delta,
    const double* last_neigh_positions_x, const double* last_neigh_positions_y,
    const double* last_cell_positions_x, const double* last_cell_positions_y,
    double* neigh_displacements_sq, double* cell_displacements_sq,
    const double* velocities_x, const double* velocities_y,
    const double* angular_velocities, const double dt)
{
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    // ---------------------------------------------------
    // 1) Store the old (current) positions before updating
    // ---------------------------------------------------
    double old_x = positions_x[particle_id];
    double old_y = positions_y[particle_id];
    double angle = angles[particle_id];

    // Write them to "last_positions"
    last_positions_x[particle_id] = old_x;
    last_positions_y[particle_id] = old_y;

    // ---------------------------------------------------
    // 2) Velocity-Verlet style update to the new positions
    // ---------------------------------------------------
    double vel_x    = velocities_x[particle_id];
    double vel_y    = velocities_y[particle_id];
    double ang_vel  = angular_velocities[particle_id];

    double temp_delta_x     = vel_x   * dt;
    double temp_delta_y     = vel_y   * dt;
    double temp_angle_delta = ang_vel * dt;

    delta_x[particle_id]      = temp_delta_x;
    delta_y[particle_id]      = temp_delta_y;
    angle_delta[particle_id]  = temp_angle_delta;

    double new_x     = old_x + temp_delta_x;
    double new_y     = old_y + temp_delta_y;
    double new_angle = angle + temp_angle_delta;

    positions_x[particle_id] = new_x;
    positions_y[particle_id] = new_y;
    angles[particle_id]      = new_angle;

    // ---------------------------------------------------
    // 3) Update neighbor/cell displacements
    // ---------------------------------------------------
    double dx_neigh = new_x - last_neigh_positions_x[particle_id];
    double dy_neigh = new_y - last_neigh_positions_y[particle_id];
    neigh_displacements_sq[particle_id] = dx_neigh * dx_neigh + dy_neigh * dy_neigh;

    double dx_cell = new_x - last_cell_positions_x[particle_id];
    double dy_cell = new_y - last_cell_positions_y[particle_id];
    cell_displacements_sq[particle_id] = dx_cell * dx_cell + dy_cell * dy_cell;
}

__global__ void kernelCalculateRigidDampedForces(double* forces_x, double* forces_y, double* torques, const double* velocities_x, const double* velocities_y, const double* angular_velocities, const double damping_coefficient) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    forces_x[particle_id] -= damping_coefficient * velocities_x[particle_id];
    forces_y[particle_id] -= damping_coefficient * velocities_y[particle_id];
    torques[particle_id] -= damping_coefficient * angular_velocities[particle_id];
}

__global__ void kernelMixRigidVelocitiesAndForces(double* velocities_x, double* velocities_y, double* angular_velocities, const double* forces_x, const double* forces_y, const double* torques, const double alpha) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;
    double force_x = forces_x[particle_id];
    double force_y = forces_y[particle_id];
    double vel_x = velocities_x[particle_id];
    double vel_y = velocities_y[particle_id];
    double force_norm = std::sqrt(force_x * force_x + force_y * force_y);
    double vel_norm = std::sqrt(vel_x * vel_x + vel_y * vel_y);
    double mixing_ratio = 0.0;
    if (force_norm > 1e-16) {
        mixing_ratio = vel_norm / force_norm * alpha;
    } else {
        vel_x = 0.0;
        vel_y = 0.0;
    }

    double ang_vel = angular_velocities[particle_id];
    double torque = torques[particle_id];
    double torque_norm = std::abs(torque);
    double ang_vel_norm = std::abs(ang_vel);
    double torque_mixing_ratio = 0.0;
    if (torque_norm > 1e-16) {
        torque_mixing_ratio = ang_vel_norm / torque_norm * alpha;
    } else {
        ang_vel = 0.0;
        torque_mixing_ratio = 0.0;
    }

    velocities_x[particle_id] = vel_x * (1 - alpha) + force_x * mixing_ratio;
    velocities_y[particle_id] = vel_y * (1 - alpha) + force_y * mixing_ratio;
    angular_velocities[particle_id] = ang_vel * (1 - alpha) + torque * torque_mixing_ratio;
}

__global__ void kernelUpdateRigidVelocities(double* velocities_x, double* velocities_y, double* angular_velocities, const double* forces_x, const double* forces_y, const double* torques, const double* masses, const double* moments_of_inertia, const double dt, bool rotation) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    // Load values into registers to minimize global memory access
    double force_x = forces_x[particle_id];
    double force_y = forces_y[particle_id];
    double vel_x = velocities_x[particle_id];
    double vel_y = velocities_y[particle_id];
    double torque = torques[particle_id];
    double moment_of_inertia = moments_of_inertia[particle_id];
    double mass = masses[particle_id];
    double ang_vel = angular_velocities[particle_id];

    // Update velocities
    if (rotation) {
        ang_vel += torque * dt / moment_of_inertia;
    } else {
        ang_vel = 0.0;
    }
    vel_x += force_x * dt / mass;
    vel_y += force_y * dt / mass;

    // Store the results back to global memory
    velocities_x[particle_id] = vel_x;
    velocities_y[particle_id] = vel_y;
    angular_velocities[particle_id] = ang_vel;
}

__global__ void kernelTranslateAndRotateVertices1(
    const double* last_positions_x, const double* last_positions_y,
    const double* positions_x,      const double* positions_y,
    double* vertex_positions_x,     double* vertex_positions_y,
    const double* delta_x,          const double* delta_y,
    const double* angle_delta)
{
    long vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_id >= d_n_vertices) return;
    
    long particle_id = d_vertex_particle_index_ptr[vertex_id];

    // ------------------------------
    // 1) Load old vs. new centers
    // ------------------------------
    double old_x = last_positions_x[particle_id];
    double old_y = last_positions_y[particle_id];
    double new_x = positions_x[particle_id];
    double new_y = positions_y[particle_id];

    // The vertex's old (current) position in global coords
    double vx = vertex_positions_x[vertex_id];
    double vy = vertex_positions_y[vertex_id];

    // The net translations & angle changes
    double dx     = delta_x[particle_id];
    double dy     = delta_y[particle_id];
    double dtheta = angle_delta[particle_id];

    // ----------------------------------------------------
    // 2) Shift so the old center is at (0, 0), then rotate
    // ----------------------------------------------------
    double rx = vx - old_x; 
    double ry = vy - old_y;

    double cosA = cos(dtheta);
    double sinA = sin(dtheta);

    double rx_rot = rx * cosA - ry * sinA;
    double ry_rot = rx * sinA + ry * cosA;

    // --------------------------------------
    // 3) Now place it at the new center
    // --------------------------------------
    double vx_new = new_x + rx_rot;
    double vy_new = new_y + ry_rot;

    vertex_positions_x[vertex_id] = vx_new;
    vertex_positions_y[vertex_id] = vy_new;
}

__global__ void kernelTranslateAndRotateVertices2(const double* positions_x, const double* positions_y, double* vertex_positions_x, double* vertex_positions_y, const double* delta_x, const double* delta_y, const double* angle_delta) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];
    double delta_x_particle = delta_x[particle_id];
    double delta_y_particle = delta_y[particle_id];
    double angle_delta_particle = angle_delta[particle_id];
    double cos_angle = cos(angle_delta_particle);
    double sin_angle = sin(angle_delta_particle);

    // Apply the transformation to all vertices of the particle
    for (long v = 0; v < d_num_vertices_in_particle_ptr[particle_id]; v++) {
        long vertex_id = d_particle_start_index_ptr[particle_id] + v;
        double vertex_pos_x = vertex_positions_x[vertex_id];
        double vertex_pos_y = vertex_positions_y[vertex_id];

        double rel_x = vertex_pos_x - pos_x;
        double rel_y = vertex_pos_y - pos_y;

        // Rotate
        double rot_x = rel_x * cos_angle - rel_y * sin_angle;
        double rot_y = rel_x * sin_angle + rel_y * cos_angle;

        // Translate
        vertex_positions_x[vertex_id] = rot_x + pos_x + delta_x_particle;
        vertex_positions_y[vertex_id] = rot_y + pos_y + delta_y_particle;
    }
}

__global__ void kernelZeroRigidBumpyParticleForceAndPotentialEnergy(double* forces_x, double* forces_y, double* torques, double* potential_energy) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    forces_x[particle_id] = 0.0;
    forces_y[particle_id] = 0.0;
    torques[particle_id] = 0.0;
    potential_energy[particle_id] = 0.0;
}

__global__ void kernelZeroRigidBumpyVertexForceAndPotentialEnergy(double* vertex_forces_x, double* vertex_forces_y, double* vertex_torques, double* vertex_potential_energy) {
    long vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_id >= d_n_vertices) return;

    vertex_forces_x[vertex_id] = 0.0;
    vertex_forces_y[vertex_id] = 0.0;
    vertex_torques[vertex_id] = 0.0;
    vertex_potential_energy[vertex_id] = 0.0;
}

__global__ void kernelCalculateTranslationalAndRotationalKineticEnergy(
    const double* __restrict__ velocities_x, const double* __restrict__ velocities_y,
    const double* __restrict__ masses, const double* __restrict__ angular_velocities,
    const double* __restrict__ moments_of_inertia, double* __restrict__ kinetic_energy) 
{
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    double vel_x = velocities_x[particle_id];
    double vel_y = velocities_y[particle_id];
    double mass = masses[particle_id];
    double ang_vel = angular_velocities[particle_id];
    double moment_of_inertia = moments_of_inertia[particle_id];

    kinetic_energy[particle_id] = 0.5 * mass * (vel_x * vel_x + vel_y * vel_y) + 0.5 * moment_of_inertia * ang_vel * ang_vel;
}

__global__ void kernelStopRattlerVelocities(double* velocities_x, double* velocities_y, double* angular_velocities, const long* __restrict__ contact_counts, const double rattler_threshold) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    if (contact_counts[particle_id] < rattler_threshold) {
        velocities_x[particle_id] = 0.0;
        velocities_y[particle_id] = 0.0;
        angular_velocities[particle_id] = 0.0;
    }
}

__global__ void kernelSetRandomCagePositions(double* positions_x, double* positions_y, double* angles, const long* __restrict__ particle_cage_id, const long* __restrict__ cage_start_index, const double* __restrict__ cage_size_x, const double* __restrict__ cage_size_y, const double* __restrict__ random_numbers_x, const double* __restrict__ random_numbers_y, const double* __restrict__ random_numbers_t, const double* __restrict__ cage_center_x, const double* __restrict__ cage_center_y) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;
    long cage_id = particle_cage_id[particle_id];
    long cage_start_id = cage_start_index[cage_id];
    if (particle_id != cage_start_id) return;  // only the first particle in the cage will be updated

    double pos_x = random_numbers_x[cage_id] * cage_size_x[cage_id] + cage_center_x[cage_id];
    double pos_y = random_numbers_y[cage_id] * cage_size_y[cage_id] + cage_center_y[cage_id];
    double theta = random_numbers_t[cage_id] * 2 * M_PI;

    positions_x[particle_id] = pos_x;
    positions_y[particle_id] = pos_y;
    angles[particle_id] = theta;
}

__global__ void kernelSetRandomVoronoiPositions(double* positions_x, double* positions_y, double* angles, const double* __restrict__ cage_center_x, const double* __restrict__ cage_center_y, const long* __restrict__ particle_cage_id, const long* __restrict__ cage_start_index, const double* __restrict__ voro_pos_x, const double* __restrict__ voro_pos_y, const long* __restrict__ voro_start_index, const long* __restrict__ voro_size, const double* __restrict__ voro_triangle_areas, const double* __restrict__ random_u, const double* __restrict__ random_v, const double* __restrict__ random_tri, const double* __restrict__ random_t) {
    long cage_id = blockIdx.x * blockDim.x + threadIdx.x;
    long particle_id = cage_start_index[cage_id];
    if (particle_id >= d_n_particles) return;

    long voro_start = voro_start_index[cage_id];
    long voro_len = voro_size[cage_id];
    if (voro_len == 0) return;

    if (voro_len < 3) {
        positions_x[particle_id] = voro_pos_x[voro_start];
        positions_y[particle_id] = voro_pos_y[voro_start];
        return;
    }
    // anchor vertex a
    const double ax = cage_center_x[cage_id];
    const double ay = cage_center_y[cage_id];

    // pick triangle proportionally to its area
    double target = random_tri[cage_id];  // r_tri in [0,1)
    double frac_area_sum_prev = 0.0;
    long i_sel = 0;

    // find the index within the fractional triangle areas that is greater than or equal to the target
    for (long i = 0; i < voro_len; ++i) {
        double frac_area_sum = voro_triangle_areas[voro_start + i];
        if ((target > frac_area_sum_prev) && (target <= frac_area_sum)) { i_sel = i; break; }
        frac_area_sum_prev = frac_area_sum;
    }

    // selected triangle vertices
    double bx = voro_pos_x[voro_start + i_sel], by = voro_pos_y[voro_start + i_sel];
    long c_index = mod(i_sel + 1, voro_len);
    double cx = voro_pos_x[voro_start + c_index], cy = voro_pos_y[voro_start + c_index];

    // uniform sample inside triangle
    double u = random_u[cage_id];
    double v = random_v[cage_id];
    if (u + v > 1.0) { u = 1.0 - u; v = 1.0 - v; }

    positions_x[particle_id] = ax + u*(bx - ax) + v*(cx - ax);
    positions_y[particle_id] = ay + u*(by - ay) + v*(cy - ay);
    angles[particle_id] = random_t[cage_id] * 2 * M_PI;
    // printf("particle_id: %ld, pos_x: %f, pos_y: %f, angle: %f\n", particle_id, positions_x[particle_id], positions_y[particle_id], angles[particle_id]);
}

// ----------------------------------------------------------------------
// ------------------------- Force Routines -----------------------------
// ----------------------------------------------------------------------

__global__ void kernelCalcRigidBumpyWallForces(const double* positions_x, const double* positions_y, const double* vertex_positions_x, const double* vertex_positions_y, double* vertex_forces_x, double* vertex_forces_y, double* vertex_torques, double* vertex_potential_energy) {
    long vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_id >= d_n_vertices) return;

    double vertex_pos_x = vertex_positions_x[vertex_id];
    double vertex_pos_y = vertex_positions_y[vertex_id];
    double particle_pos_x = positions_x[d_vertex_particle_index_ptr[vertex_id]];
    double particle_pos_y = positions_y[d_vertex_particle_index_ptr[vertex_id]];
    double force_x, force_y;
    double interaction_energy = calcWallInteraction(vertex_pos_x, vertex_pos_y, d_vertex_radius, force_x, force_y);
    vertex_forces_x[vertex_id] += force_x;
    vertex_forces_y[vertex_id] += force_y;
    vertex_torques[vertex_id] += calcTorque(force_x, force_y, vertex_pos_x, vertex_pos_y, particle_pos_x, particle_pos_y);
    vertex_potential_energy[vertex_id] += interaction_energy;
}



__global__ void kernelCalcRigidBumpyForces1(
    const double* __restrict__ positions_x, const double* __restrict__ positions_y,
    const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y, 
    double* __restrict__ vertex_forces_x, double* __restrict__ vertex_forces_y, double* __restrict__ vertex_torques, double* __restrict__ vertex_potential_energy) 
{
    // TODO: could probably make this faster by copying the relevant data into shared memory

    long vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_id >= d_n_vertices) return;

    long num_neighbors = d_num_vertex_neighbors_ptr[vertex_id];
    // printf("vertex_id: %ld, num_neighbors: %ld\n", vertex_id, num_neighbors);
    if (num_neighbors == 0) return;

    double vertex_pos_x = vertex_positions_x[vertex_id];
    double vertex_pos_y = vertex_positions_y[vertex_id];
    long particle_id = d_vertex_particle_index_ptr[vertex_id];
    double particle_pos_x = positions_x[particle_id];
    double particle_pos_y = positions_y[particle_id];
    double force_acc_x = 0.0, force_acc_y = 0.0;
    double energy = 0.0;

    for (long n = 0; n < num_neighbors; n++) {
        long other_id = d_vertex_neighbor_list_ptr[vertex_id * d_max_vertex_neighbors_allocated + n];
        if (other_id == -1 || other_id == vertex_id) continue;

        // Load neighbor's data
        double other_vertex_pos_x = vertex_positions_x[other_id];
        double other_vertex_pos_y = vertex_positions_y[other_id];

        // Calculate force and energy using the interaction function
        double force_x, force_y;
        double interaction_energy = calcPointPointInteraction(
            vertex_pos_x, vertex_pos_y, d_vertex_radius, other_vertex_pos_x, other_vertex_pos_y, d_vertex_radius, 
            force_x, force_y
        );

        // Accumulate force and energy
        force_acc_x += force_x;
        force_acc_y += force_y;
        energy += interaction_energy;
    }

    // Calculate torque
    double torque = calcTorque(force_acc_x, force_acc_y, vertex_pos_x, vertex_pos_y, particle_pos_x, particle_pos_y);

    // Store results in global memory
    vertex_forces_x[vertex_id] += force_acc_x;
    vertex_forces_y[vertex_id] += force_acc_y;
    vertex_torques[vertex_id] += torque;
    vertex_potential_energy[vertex_id] += energy;
}

__global__ void kernelCalcRigidBumpyParticleForces1(
    const double* __restrict__ vertex_forces_x, const double* __restrict__ vertex_forces_y, const double* __restrict__ vertex_torques, const double* __restrict__ vertex_potential_energy,
    double* __restrict__ particle_forces_x, double* __restrict__ particle_forces_y, double* __restrict__ particle_torques, double* __restrict__ particle_potential_energy) 
{
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    double force_acc_x = 0.0, force_acc_y = 0.0;
    double torque_acc = 0.0;
    double energy = 0.0;

    for (long v = 0; v < d_num_vertices_in_particle_ptr[particle_id]; v++) {
        long vertex_id = d_particle_start_index_ptr[particle_id] + v;
        force_acc_x += vertex_forces_x[vertex_id];
        force_acc_y += vertex_forces_y[vertex_id];
        torque_acc += vertex_torques[vertex_id];
        energy += vertex_potential_energy[vertex_id];
    }

    particle_forces_x[particle_id] += force_acc_x;
    particle_forces_y[particle_id] += force_acc_y;
    particle_torques[particle_id] += torque_acc;
    particle_potential_energy[particle_id] += energy;
}

__global__ void kernelCalcRigidBumpyForces2(const double* __restrict__ positions_x, const double* __restrict__ positions_y, const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y, double* __restrict__ particle_forces_x, double* __restrict__ particle_forces_y, double* __restrict__ particle_torques, double* __restrict__ particle_potential_energy) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    double force_acc_x = 0.0, force_acc_y = 0.0;
    double torque_acc = 0.0;
    double energy = 0.0;
    double particle_pos_x = positions_x[particle_id];
    double particle_pos_y = positions_y[particle_id];
    double other_vertex_pos_x, other_vertex_pos_y;

    // loop over all vertices of the particle
    for (long v = 0; v < d_num_vertices_in_particle_ptr[particle_id]; v++) {
        long vertex_id = d_particle_start_index_ptr[particle_id] + v;
        double vertex_pos_x = vertex_positions_x[vertex_id];
        double vertex_pos_y = vertex_positions_y[vertex_id];
        long num_neighbors = d_num_vertex_neighbors_ptr[vertex_id];
        for (long n = 0; n < num_neighbors; n++) {
            long other_id = d_vertex_neighbor_list_ptr[vertex_id * d_max_vertex_neighbors_allocated + n];
            if (other_id == -1 || other_id == vertex_id) continue;
            other_vertex_pos_x = vertex_positions_x[other_id];
            other_vertex_pos_y = vertex_positions_y[other_id];
            double temp_force_x, temp_force_y;
            double interaction_energy = calcPointPointInteraction(vertex_pos_x, vertex_pos_y, d_vertex_radius, other_vertex_pos_x, other_vertex_pos_y, d_vertex_radius, temp_force_x, temp_force_y);
            force_acc_x += temp_force_x;
            force_acc_y += temp_force_y;
            torque_acc += calcTorque(temp_force_x, temp_force_y, vertex_pos_x, vertex_pos_y, particle_pos_x, particle_pos_y);
            energy += interaction_energy;
        }
    }

    particle_forces_x[particle_id] = force_acc_x;
    particle_forces_y[particle_id] = force_acc_y;
    particle_torques[particle_id] = torque_acc;
    particle_potential_energy[particle_id] = energy;
}


__global__ void kernelCalcRigidBumpyForceDistancePairs(
    const double* positions_x,
    const double* positions_y,
    const double* vertex_positions_x,
    const double* vertex_positions_y,
    double* potential_pairs,
    double* force_pairs_x,
    double* force_pairs_y,
    double* distance_pairs_x,
    double* distance_pairs_y,
    long* this_pair_id,
    long* other_pair_id,
    double* overlap_pairs,
    double* radsum_pairs,
    const double* radii,
    const long* static_particle_index,
    double* pair_separation_angle,
    double* angle_pairs_i,
    double* angle_pairs_j,
    long* this_vertex_contact_count,
    const double* angles,
    double* pair_friction_coefficient,
    double* pair_vertex_overlaps,
    double* hessian_pairs_xx,
    double* hessian_pairs_xy,
    double* hessian_pairs_yx,
    double* hessian_pairs_yy,
    double* hessian_pairs_xt,
    double* hessian_pairs_yt,
    double* hessian_pairs_tt,
    double* hessian_pairs_tx,
    double* hessian_pairs_ty,
    double* hessian_ii_xx,
    double* hessian_ii_xy,
    double* hessian_ii_yx,
    double* hessian_ii_yy,
    double* hessian_ii_xt,
    double* hessian_ii_yt,
    double* hessian_ii_tt,
    double* hessian_ii_tx,
    double* hessian_ii_ty
) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;
    long static_particle_id = static_particle_index[particle_id];

    long num_neighbors = d_num_neighbors_ptr[particle_id];
    if (num_neighbors == 0) return;

    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];
    double theta_i = angles[particle_id];
    double rad = radii[particle_id];

    // loop over the particle neighbors
    for (long n = 0; n < num_neighbors; n++) {
        long other_id = d_neighbor_list_ptr[particle_id * d_max_neighbors_allocated + n];
        if (other_id == -1 || other_id == particle_id) continue;
        long other_static_id = static_particle_index[other_id];

        double other_pos_x = positions_x[other_id];
        double other_pos_y = positions_y[other_id];
        double theta_j = angles[other_id];
        
        double force_x = 0.0, force_y = 0.0;
        long vertex_count_i = 0;
        long vertex_count_j = 0;

        double interaction_energy = 0.0;
        double temp_energy;
        double temp_vertex_overlap = 0.0;
        double other_rad = radii[other_id];

        std::array<double, 9> hess_ij_ab = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        std::array<double, 9> hess_ii_ab = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        // loop over the vertices of this particle
        for (long u = 0; u < d_num_vertices_in_particle_ptr[particle_id]; u++) {
            long vertex_id = d_particle_start_index_ptr[particle_id] + u;
            double vertex_pos_x = vertex_positions_x[vertex_id];
            double vertex_pos_y = vertex_positions_y[vertex_id];
            double phi_u = u * 2 * M_PI / d_num_vertices_in_particle_ptr[particle_id];
            double rad_dx = vertex_pos_x - pos_x;
            double rad_dy = vertex_pos_y - pos_y;
            double R_i = sqrt(rad_dx * rad_dx + rad_dy * rad_dy);

            bool is_contact = false;

            // loop over the neighbors of the vertex
            for (long n_v = 0; n_v < d_num_vertex_neighbors_ptr[vertex_id]; n_v++) {
                long other_vertex_id = d_vertex_neighbor_list_ptr[vertex_id * d_max_vertex_neighbors_allocated + n_v];
                long other_particle_id = d_vertex_particle_index_ptr[other_vertex_id];

                // calculate the interaction force between the vertex and the other vertex only if it belongs to the other particle
                if (other_vertex_id == -1 || other_vertex_id == vertex_id || other_particle_id != other_id) continue;
                double other_vertex_pos_x = vertex_positions_x[other_vertex_id];
                double other_vertex_pos_y = vertex_positions_y[other_vertex_id];
                // get the index of the other vertex in the particle
                long v = other_vertex_id - d_particle_start_index_ptr[other_id];
                if (v < 0) {
                    printf("v: %ld, other_vertex_id: %ld, d_particle_start_index_ptr[other_id]: %ld\n", v, other_vertex_id, d_particle_start_index_ptr[other_id]);
                }
                double phi_v = v * 2 * M_PI / d_num_vertices_in_particle_ptr[other_id];
                double other_rad_dx = other_vertex_pos_x - other_pos_x;
                double other_rad_dy = other_vertex_pos_y - other_pos_y;
                double R_j = sqrt(other_rad_dx * other_rad_dx + other_rad_dy * other_rad_dy);

                double temp_force_x, temp_force_y;
                temp_energy = calcPointPointInteraction(vertex_pos_x, vertex_pos_y, d_vertex_radius, other_vertex_pos_x, other_vertex_pos_y, d_vertex_radius, temp_force_x, temp_force_y);
                interaction_energy += temp_energy;
                force_x += temp_force_x;
                force_y += temp_force_y;
                if (temp_energy > 0.0) {
                    is_contact = true;
                    double dx = pbcDistance(vertex_pos_x, other_vertex_pos_x, 0);
                    double dy = pbcDistance(vertex_pos_y, other_vertex_pos_y, 1);
                    temp_vertex_overlap += (d_vertex_radius * 2) - sqrt(dx * dx + dy * dy);

                    double dist = sqrt(dx * dx + dy * dy);
                    double radsum = d_vertex_radius * 2.0;

                    double t_ij = - d_e_c / radsum * (1 - dist / radsum);
                    double c_ij = d_e_c / (radsum * radsum);

                    for (int a = 0; a < 3; a++) {
                        for (int b = 0; b < 3; b++) {
                            // off-diagonal block
                            double d_ia_dx = (a == 0) - R_i * std::sin(theta_i + phi_u) * (a == 2);
                            double d_ia_dy = (a == 1) + R_i * std::cos(theta_i + phi_u) * (a == 2);
                            double d_jb_dx = -1.0 * ((b == 0) - R_j * std::sin(theta_j + phi_v) * (b == 2));
                            double d_jb_dy = -1.0 * ((b == 1) + R_j * std::cos(theta_j + phi_v) * (b == 2));

                            double q_ia = dx * d_ia_dx + dy * d_ia_dy;
                            double q_jb = dx * d_jb_dx + dy * d_jb_dy;

                            double d_ia_q_jb = (d_ia_dx * d_jb_dx + d_ia_dy * d_jb_dy) / dist;  // there are no second derivatives here since i and j are different coordinates

                            hess_ij_ab[a * 3 + b] += c_ij * q_ia * q_jb / (dist * dist) + t_ij * (d_ia_q_jb - q_ia * q_jb / (dist * dist * dist));

                            // diagonal block
                            double d_ib_dx = (b == 0) - R_i * std::sin(theta_i + phi_u) * (b == 2);
                            double d_ib_dy = (b == 1) + R_i * std::cos(theta_i + phi_u) * (b == 2);

                            // there are now some second derivatives
                            double d_ia_ib_dx = - R_i * std::cos(theta_i + phi_u) * (a == 2) * (b == 2);
                            double d_ia_ib_dy = - R_i * std::sin(theta_i + phi_u) * (a == 2) * (b == 2);

                            double q_ib = dx * d_ib_dx + dy * d_ib_dy;

                            double d_ia_q_ib = (d_ia_dx * d_ib_dx + dx * d_ia_ib_dx + d_ia_dy * d_ib_dy + dy * d_ia_ib_dy) / dist;

                            hess_ii_ab[a * 3 + b] += c_ij * q_ia * q_ib / (dist * dist) + t_ij * (d_ia_q_ib - q_ia * q_ib / (dist * dist * dist));
                        }
                    }
                }
            }

            if (is_contact) {
                vertex_count_i++;  // this is correct
            }
        }

        long pair_id = particle_id * d_max_neighbors_allocated + n;
        potential_pairs[pair_id] = interaction_energy;
        force_pairs_x[pair_id] = force_x;
        force_pairs_y[pair_id] = force_y;
        double x_dist = pbcDistance(pos_x, other_pos_x, 0);
        double y_dist = pbcDistance(pos_y, other_pos_y, 1);
        distance_pairs_x[pair_id] = x_dist;
        distance_pairs_y[pair_id] = y_dist;
        
        this_pair_id[pair_id] = static_particle_id;
        other_pair_id[pair_id] = other_static_id;

        pair_vertex_overlaps[pair_id] = temp_vertex_overlap;
        
        // this_pair_id[pair_id] = particle_id;
        // other_pair_id[pair_id] = other_id;
        
        double dist = sqrt(x_dist * x_dist + y_dist * y_dist);
        overlap_pairs[pair_id] = dist - (rad + other_rad);
        radsum_pairs[pair_id] = rad + other_rad;
        pair_separation_angle[pair_id] = atan2(y_dist, x_dist);
        angle_pairs_i[pair_id] = angles[particle_id];
        angle_pairs_j[pair_id] = angles[other_id];
        this_vertex_contact_count[pair_id] = vertex_count_i;

        // calculate friction if the force is nonzero
        double friction_coefficient = 0.0;
        if (force_x != 0.0 || force_y != 0.0) {
            double normal_force_x = force_x * x_dist / dist;
            double normal_force_y = force_y * y_dist / dist;
            double normal_force = sqrt(normal_force_x * normal_force_x + normal_force_y * normal_force_y);
            double tangential_force_x = force_x - normal_force_x;
            double tangential_force_y = force_y - normal_force_y;
            double tangential_force = sqrt(tangential_force_x * tangential_force_x + tangential_force_y * tangential_force_y);
            friction_coefficient = tangential_force / normal_force;
        }
        pair_friction_coefficient[pair_id] = friction_coefficient;

        hessian_pairs_xx[pair_id] += hess_ij_ab[0];
        hessian_pairs_xy[pair_id] += hess_ij_ab[1];
        hessian_pairs_xt[pair_id] += hess_ij_ab[2];
        hessian_pairs_yx[pair_id] += hess_ij_ab[3];
        hessian_pairs_yy[pair_id] += hess_ij_ab[4];
        hessian_pairs_yt[pair_id] += hess_ij_ab[5];
        hessian_pairs_tx[pair_id] += hess_ij_ab[6];
        hessian_pairs_ty[pair_id] += hess_ij_ab[7];
        hessian_pairs_tt[pair_id] += hess_ij_ab[8];

        hessian_ii_xx[pair_id] += hess_ii_ab[0];
        hessian_ii_xy[pair_id] += hess_ii_ab[1];
        hessian_ii_xt[pair_id] += hess_ii_ab[2];
        hessian_ii_yx[pair_id] += hess_ii_ab[3];
        hessian_ii_yy[pair_id] += hess_ii_ab[4];
        hessian_ii_yt[pair_id] += hess_ii_ab[5];
        hessian_ii_tx[pair_id] += hess_ii_ab[6];
        hessian_ii_ty[pair_id] += hess_ii_ab[7];
        hessian_ii_tt[pair_id] += hess_ii_ab[8];
    }
}


__global__ void kernelCalcRigidBumpyStressTensor(
    const double* __restrict__ positions_x, const double* __restrict__ positions_y,
    const double* __restrict__ velocities_x, const double* __restrict__ velocities_y,
    const double* __restrict__ angular_velocities,
    const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y,
    const double* __restrict__ vertex_masses,
    double* __restrict__ stress_tensor_x_x, double* __restrict__ stress_tensor_x_y,
    double* __restrict__ stress_tensor_y_x, double* __restrict__ stress_tensor_y_y
) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;
    
    double stress_tensor_x_x_acc = 0.0;
    double stress_tensor_x_y_acc = 0.0;
    double stress_tensor_y_x_acc = 0.0;
    double stress_tensor_y_y_acc = 0.0;

    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];
    double vel_x = velocities_x[particle_id];
    double vel_y = velocities_y[particle_id];
    double ang_vel = angular_velocities[particle_id];

    double vertex_vel_x, vertex_vel_y;

    double box_area = d_box_size[0] * d_box_size[1];

    // loop over the vertices of the particle
    for (long v = 0; v < d_num_vertices_in_particle_ptr[particle_id]; v++) {
        long vertex_id = d_particle_start_index_ptr[particle_id] + v;
        double vertex_pos_x = vertex_positions_x[vertex_id];
        double vertex_pos_y = vertex_positions_y[vertex_id];
        double vertex_mass = vertex_masses[vertex_id];

        double rel_pos_x = vertex_pos_x - pos_x;
        double rel_pos_y = vertex_pos_y - pos_y;
        double p_rad = sqrt(rel_pos_x * rel_pos_x + rel_pos_y * rel_pos_y);

        vertex_vel_x = vel_x;
        vertex_vel_y = vel_y;
        if (ang_vel != 0.0) {
            double angle = atan2(rel_pos_y, rel_pos_x);
            vertex_vel_x -= sin(angle) * ang_vel * p_rad;
            vertex_vel_y += cos(angle) * ang_vel * p_rad;
        }

        // add the kinetic contribution of the vertex
        stress_tensor_x_x_acc += vertex_mass * vertex_vel_x * vertex_vel_x;
        stress_tensor_x_y_acc += vertex_mass * vertex_vel_x * vertex_vel_y;
        stress_tensor_y_x_acc += vertex_mass * vertex_vel_y * vertex_vel_x;
        stress_tensor_y_y_acc += vertex_mass * vertex_vel_y * vertex_vel_y;
        
        // loop over the neighbors of the vertex
        for (long n_v = 0; n_v < d_num_vertex_neighbors_ptr[vertex_id]; n_v++) {
            long other_vertex_id = d_vertex_neighbor_list_ptr[vertex_id * d_max_vertex_neighbors_allocated + n_v];
            long other_particle_id = d_vertex_particle_index_ptr[other_vertex_id];
            
            // calculate the interaction force between the vertex and the other vertex only if it belongs to the other particle
            if (other_vertex_id == -1 || other_vertex_id == vertex_id || other_particle_id == particle_id) continue;

            double other_vertex_pos_x = vertex_positions_x[other_vertex_id];
            double other_vertex_pos_y = vertex_positions_y[other_vertex_id];
            
            double x_dist = pbcDistance(vertex_pos_x, other_vertex_pos_x, 0);
            double y_dist = pbcDistance(vertex_pos_y, other_vertex_pos_y, 1);
            
            double force_x, force_y;
            double interaction_energy = calcPointPointInteraction(
                vertex_pos_x, vertex_pos_y, d_vertex_radius, 
                other_vertex_pos_x, other_vertex_pos_y, d_vertex_radius, 
                force_x, force_y
            );

            // divide by 2 to avoid double counting
            stress_tensor_x_x_acc += x_dist * force_x / 2;
            stress_tensor_x_y_acc += x_dist * force_y / 2;
            stress_tensor_y_x_acc += y_dist * force_x / 2;
            stress_tensor_y_y_acc += y_dist * force_y / 2;
        }
    }
    stress_tensor_x_x[particle_id] = stress_tensor_x_x_acc / box_area;
    stress_tensor_x_y[particle_id] = stress_tensor_x_y_acc / box_area;
    stress_tensor_y_x[particle_id] = stress_tensor_y_x_acc / box_area;
    stress_tensor_y_y[particle_id] = stress_tensor_y_y_acc / box_area;
}

// ----------------------------------------------------------------------
// --------------------- Contacts and Neighbors -------------------------
// ----------------------------------------------------------------------


__global__ void kernelReorderRigidBumpyParticleData(
    const long* __restrict__ particle_index,
    long* __restrict__ old_to_new_particle_index,
    const long* __restrict__ num_vertices_in_particle,
    long* __restrict__ num_vertices_in_particle_new,
    const double* __restrict__ positions_x, const double* __restrict__ positions_y,
    double* __restrict__ positions_x_new, double* __restrict__ positions_y_new,
    const double* __restrict__ forces_x, const double* __restrict__ forces_y,
    double* __restrict__ forces_x_new, double* __restrict__ forces_y_new,
    const double* __restrict__ velocities_x, const double* __restrict__ velocities_y,
    double* __restrict__ velocities_x_new, double* __restrict__ velocities_y_new,
    const double* __restrict__ angular_velocities, const double* __restrict__ torques,
    double* __restrict__ angular_velocities_new, double* __restrict__ torques_new,
    double* __restrict__ angles, double* __restrict__ angles_new,
    const double* __restrict__ masses, const double* __restrict__ radii,
    double* __restrict__ masses_new, double* __restrict__ radii_new,
    const double* __restrict__ moments_of_inertia,
    double* __restrict__ moments_of_inertia_new,
    double* __restrict__ last_cell_positions_x, double* __restrict__ last_cell_positions_y,
    double* __restrict__ cell_displacements_sq) {

    long new_particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (new_particle_id >= d_n_particles) return;

    long old_particle_id = particle_index[new_particle_id];
    
    // Create mapping from old particle indices to new particle indices
    old_to_new_particle_index[old_particle_id] = new_particle_id;

    // Reorder particle data
    positions_x_new[new_particle_id] = positions_x[old_particle_id];
    positions_y_new[new_particle_id] = positions_y[old_particle_id];
    forces_x_new[new_particle_id] = forces_x[old_particle_id];
    forces_y_new[new_particle_id] = forces_y[old_particle_id];
    velocities_x_new[new_particle_id] = velocities_x[old_particle_id];
    velocities_y_new[new_particle_id] = velocities_y[old_particle_id];
    angular_velocities_new[new_particle_id] = angular_velocities[old_particle_id];
    torques_new[new_particle_id] = torques[old_particle_id];
    angles_new[new_particle_id] = angles[old_particle_id];
    masses_new[new_particle_id] = masses[old_particle_id];
    radii_new[new_particle_id] = radii[old_particle_id];
    moments_of_inertia_new[new_particle_id] = moments_of_inertia[old_particle_id];
    num_vertices_in_particle_new[new_particle_id] = num_vertices_in_particle[old_particle_id];
    last_cell_positions_x[new_particle_id] = positions_x[old_particle_id];
    last_cell_positions_y[new_particle_id] = positions_y[old_particle_id];
    cell_displacements_sq[new_particle_id] = 0.0;
}

__global__ void kernelReorderRigidBumpyVertexData(
    const long* __restrict__ vertex_particle_index,
    long* __restrict__ vertex_particle_index_new,
    const long* __restrict__ old_to_new_particle_index,
    const long* __restrict__ particle_start_index,
    const long* __restrict__ particle_start_index_new,
    const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y,
    double* __restrict__ vertex_positions_x_new, double* __restrict__ vertex_positions_y_new,
    const double* __restrict__ vertex_masses, double* __restrict__ vertex_masses_new,
    const long* __restrict__ static_vertex_index,
    long* __restrict__ static_vertex_index_new) {

    long old_vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (old_vertex_id >= d_n_vertices) return;

    long old_particle_id = vertex_particle_index[old_vertex_id];
    long new_particle_id = old_to_new_particle_index[old_particle_id];
    long relative_vertex_id = old_vertex_id - particle_start_index[old_particle_id];
    long new_vertex_id = particle_start_index_new[new_particle_id] + relative_vertex_id;
    vertex_particle_index_new[new_vertex_id] = new_particle_id;
    vertex_positions_x_new[new_vertex_id] = vertex_positions_x[old_vertex_id];
    vertex_positions_y_new[new_vertex_id] = vertex_positions_y[old_vertex_id];
    vertex_masses_new[new_vertex_id] = vertex_masses[old_vertex_id];
    static_vertex_index_new[new_vertex_id] = static_vertex_index[old_vertex_id];
}

__global__ void kernelUpdateVertexNeighborList(
    const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y,
    const double* __restrict__ positions_x, const double* __restrict__ positions_y,
    const double cutoff,
    const double particle_cutoff
) {
    long vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_id >= d_n_vertices) return;

    long added_vertex_neighbors = 0;
    double pos_x = vertex_positions_x[vertex_id];
    double pos_y = vertex_positions_y[vertex_id];
    double cutoff_sq = cutoff * cutoff;
    double particle_cutoff_sq = particle_cutoff * particle_cutoff;

    long particle_id = d_vertex_particle_index_ptr[vertex_id];
    long num_particle_neighbors = d_num_neighbors_ptr[particle_id];

    // iterate over neighboring particles in the particle neighbor list
    // printf("vertex_id: %ld - %ld: %ld / %ld\n", vertex_id, num_particle_neighbors, added_vertex_neighbors, d_max_vertex_neighbors_allocated);
    // printf("d_vertex_particle_index_ptr[%ld]: %ld\n", vertex_id, d_vertex_particle_index_ptr[vertex_id]);
    // printf("d_vertex_neighbor_list_ptr[%ld]: %ld\n", vertex_id, d_vertex_neighbor_list_ptr[vertex_id * d_max_vertex_neighbors_allocated]);
    // printf("d_max_vertex_neighbors_allocated: %ld\n", d_max_vertex_neighbors_allocated);
    
    for (long n = 0; n < num_particle_neighbors; n++) {
        long other_particle_id = d_neighbor_list_ptr[particle_id * d_max_neighbors_allocated + n];
        if (other_particle_id == -1 || other_particle_id == particle_id) continue;

        // Load neighbor's data
        double other_particle_x = positions_x[other_particle_id];
        double other_particle_y = positions_y[other_particle_id];

        // If the particle center is within the particle cutoff radius of the vertex, then check the vertices of the particle
        if (isWithinCutoffSquared(pos_x, pos_y, other_particle_x, other_particle_y, particle_cutoff_sq)) {
            long first_vertex_index = d_particle_start_index_ptr[other_particle_id];
            long num_vertices = d_num_vertices_in_particle_ptr[other_particle_id];

            for (long i = first_vertex_index; i < first_vertex_index + num_vertices; i++) {
                double other_x = vertex_positions_x[i];
                double other_y = vertex_positions_y[i];

                if (isWithinCutoffSquared(pos_x, pos_y, other_x, other_y, cutoff_sq)) {
                    if (added_vertex_neighbors < d_max_vertex_neighbors_allocated) {
                        d_vertex_neighbor_list_ptr[vertex_id * d_max_vertex_neighbors_allocated + added_vertex_neighbors] = i;
                    }
                    added_vertex_neighbors++;
                }
            }
        }
    }
    d_num_vertex_neighbors_ptr[vertex_id] = added_vertex_neighbors;
    // printf("vertex_id: %ld - %ld\n", vertex_id, added_vertex_neighbors);
}

__global__ void kernelCountRigidBumpyContacts(
    const double* __restrict__ positions_x, const double* __restrict__ positions_y,
    const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y,
    const double* __restrict__ radii,
    long* __restrict__ contact_counts,
    long* __restrict__ vertex_contact_counts
) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    long num_neighbors = d_num_neighbors_ptr[particle_id];
    if (num_neighbors == 0) return;

    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];
    double rad = radii[particle_id];

    double vertex_radsum = 2 * d_vertex_radius;
    long num_vertex_contacts = 0;
    long num_contacts = 0;

    // loop over the particle neighbors
    for (long n = 0; n < num_neighbors; n++) {
        long other_id = d_neighbor_list_ptr[particle_id * d_max_neighbors_allocated + n];
        if (other_id == -1 || other_id == particle_id) continue;

        double other_pos_x = positions_x[other_id];
        double other_pos_y = positions_y[other_id];
        double other_rad = radii[other_id];
        
        double force_x = 0.0, force_y = 0.0;
        long vertex_count_i = 0;
        long vertex_count_j = 0;

        double interaction_energy = 0.0;
        double temp_energy;

        bool is_contact = false;

        // loop over the vertices of this particle
        for (long v = 0; v < d_num_vertices_in_particle_ptr[particle_id]; v++) {
            long vertex_id = d_particle_start_index_ptr[particle_id] + v;
            double vertex_pos_x = vertex_positions_x[vertex_id];
            double vertex_pos_y = vertex_positions_y[vertex_id];

            // loop over the neighbors of the vertex
            for (long n_v = 0; n_v < d_num_vertex_neighbors_ptr[vertex_id]; n_v++) {
                long other_vertex_id = d_vertex_neighbor_list_ptr[vertex_id * d_max_vertex_neighbors_allocated + n_v];
                long other_particle_id = d_vertex_particle_index_ptr[other_vertex_id];

                // calculate the interaction force between the vertex and the other vertex only if it belongs to the other particle
                if (other_vertex_id == -1 || other_vertex_id == vertex_id || other_particle_id != other_id) continue;
                double other_vertex_pos_x = vertex_positions_x[other_vertex_id];
                double other_vertex_pos_y = vertex_positions_y[other_vertex_id];

                double dx = pbcDistance(vertex_pos_x, other_vertex_pos_x, 0);
                double dy = pbcDistance(vertex_pos_y, other_vertex_pos_y, 1);
                double dist = sqrt(dx * dx + dy * dy);
                if (dist < vertex_radsum) {
                    if (!is_contact) {
                        // contact_counts[particle_id]++;
                        num_contacts++;
                    }
                    // vertex_contact_counts[vertex_id]++;
                    num_vertex_contacts++;
                    is_contact = true;
                }
            }
            if (is_contact) break;
        }
    }
    contact_counts[particle_id] = num_contacts;
    vertex_contact_counts[particle_id] = num_vertex_contacts;
}

// ----------------------------------------------------------------------
// --------------------------- Minimizers -------------------------------
// ----------------------------------------------------------------------


__global__ void kernelRigidBumpyAdamStep(
    double* __restrict__ last_positions_x, double* __restrict__ last_positions_y,
    double* __restrict__ first_moment_x, double* __restrict__ first_moment_y,
    double* __restrict__ first_moment_angle,
    double* __restrict__ second_moment_x, double* __restrict__ second_moment_y,
    double* __restrict__ second_moment_angle,
    double* __restrict__ positions_x, double* __restrict__ positions_y,
    double* __restrict__ angles,
    double* __restrict__ delta_x, double* __restrict__ delta_y,
    double* __restrict__ angle_delta,
    const double* __restrict__ forces_x, const double* __restrict__ forces_y,
    const double* __restrict__ torques,
    double alpha, double beta1, double beta2, double one_minus_beta1_pow_t, 
    double one_minus_beta2_pow_t, double epsilon, double* __restrict__ last_neigh_positions_x, double* __restrict__ last_neigh_positions_y,
    double* __restrict__ neigh_displacements_sq, double* __restrict__ last_cell_positions_x, double* __restrict__ last_cell_positions_y,
    double* __restrict__ cell_displacements_sq, bool rotation) {

    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    // Prefetch forces into registers
    double force_x = forces_x[particle_id];
    double force_y = forces_y[particle_id];
    double torque = torques[particle_id];

    // Load moments into registers
    double first_m_x = first_moment_x[particle_id];
    double first_m_y = first_moment_y[particle_id];
    double first_m_angle = first_moment_angle[particle_id];
    double second_m_x = second_moment_x[particle_id];
    double second_m_y = second_moment_y[particle_id];
    double second_m_angle = second_moment_angle[particle_id];

    // Update moments using fma for better performance
    first_m_x = fma(beta1, first_m_x, (beta1 - 1) * force_x);
    first_m_y = fma(beta1, first_m_y, (beta1 - 1) * force_y);
    first_m_angle = fma(beta1, first_m_angle, (beta1 - 1) * torque);

    second_m_x = fma(beta2, second_m_x, (1 - beta2) * force_x * force_x);
    second_m_y = fma(beta2, second_m_y, (1 - beta2) * force_y * force_y);
    second_m_angle = fma(beta2, second_m_angle, (1 - beta2) * torque * torque);

    // Compute bias-corrected moments
    double m_hat_x = first_m_x / one_minus_beta1_pow_t;
    double m_hat_y = first_m_y / one_minus_beta1_pow_t;
    double m_hat_angle = first_m_angle / one_minus_beta1_pow_t;

    double v_hat_x = second_m_x / one_minus_beta2_pow_t;
    double v_hat_y = second_m_y / one_minus_beta2_pow_t;
    double v_hat_angle = second_m_angle / one_minus_beta2_pow_t;

    // Compute position updates
    double update_x = -alpha * m_hat_x / (sqrt(v_hat_x) + epsilon);
    double update_y = -alpha * m_hat_y / (sqrt(v_hat_y) + epsilon);
    double update_angle = -alpha * m_hat_angle / (sqrt(v_hat_angle) + epsilon);

    // Store original positions
    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];
    last_positions_x[particle_id] = pos_x;
    last_positions_y[particle_id] = pos_y;

    // Update positions
    pos_x += update_x;
    pos_y += update_y;
    delta_x[particle_id] = update_x;
    delta_y[particle_id] = update_y;
    if (rotation) {
        angles[particle_id] += update_angle;
        angle_delta[particle_id] = update_angle;
    }
    positions_x[particle_id] = pos_x;
    positions_y[particle_id] = pos_y;

    double dx_neigh = pos_x - last_neigh_positions_x[particle_id];
    double dy_neigh = pos_y - last_neigh_positions_y[particle_id];
    neigh_displacements_sq[particle_id] = dx_neigh * dx_neigh + dy_neigh * dy_neigh;

    double dx_cell = pos_x - last_cell_positions_x[particle_id];
    double dy_cell = pos_y - last_cell_positions_y[particle_id];
    cell_displacements_sq[particle_id] = dx_cell * dx_cell + dy_cell * dy_cell;

    // Store updated moments back
    first_moment_x[particle_id] = first_m_x;
    first_moment_y[particle_id] = first_m_y;
    first_moment_angle[particle_id] = first_m_angle;
    second_moment_x[particle_id] = second_m_x;
    second_moment_y[particle_id] = second_m_y;
    second_moment_angle[particle_id] = second_m_angle;
}

// ----------------------------------------------------------------------
// ------------------------ Vertex Utilities ----------------------------
// ----------------------------------------------------------------------

__global__ void kernelSetVertexParticleIndex(
    const long* __restrict__ num_vertices_in_particle,
    const long* __restrict__ particle_start_index,
    long* __restrict__ vertex_particle_index
) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    long first_vertex_index = particle_start_index[particle_id];
    long num_vertices = num_vertices_in_particle[particle_id];

    for (long i = 0; i < num_vertices; i++) {
        vertex_particle_index[first_vertex_index + i] = particle_id;
    }
}

__global__ void kernelGetNumVerticesInParticles(
    const double* __restrict__ radii,
    const double min_particle_diam,
    const long num_vertices_in_small_particle,
    const double max_particle_diam,
    const long num_vertices_in_large_particle,
    long* __restrict__ num_vertices_in_particle) {
    
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    double radius = radii[particle_id];
    if (radius == min_particle_diam / 2.0) {
        num_vertices_in_particle[particle_id] = num_vertices_in_small_particle;
    } else if (radius == max_particle_diam / 2.0) {
        num_vertices_in_particle[particle_id] = num_vertices_in_large_particle;
    } else {
        printf("Error: particle radius %f is not equal to min or max particle diameter\n", radius);
    }
}


__global__ void kernelInitializeVerticesOnParticles(
    const double* __restrict__ positions_x, const double* __restrict__ positions_y,
    const double* __restrict__ radii, const double* __restrict__ angles,
    const long* __restrict__ particle_start_index,
    const long* __restrict__ num_vertices_in_particle,
    double* __restrict__ vertex_positions_x, double* __restrict__ vertex_positions_y) {
    
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    double angle = angles[particle_id];
    double radius = radii[particle_id];
    long num_vertices = num_vertices_in_particle[particle_id];
    long vertex_start_index = particle_start_index[particle_id];
    double vertex_angle_increment = 2 * M_PI / num_vertices;
    double inner_radius = radius - d_vertex_radius;


    for (long i = 0; i < num_vertices; i++) {
        double vertex_angle = i * vertex_angle_increment + angle;
        double vertex_x = positions_x[particle_id] + inner_radius * cos(vertex_angle);
        double vertex_y = positions_y[particle_id] + inner_radius * sin(vertex_angle);
        vertex_positions_x[vertex_start_index + i] = vertex_x;
        vertex_positions_y[vertex_start_index + i] = vertex_y;
    }
}

__global__ void kernelGetVertexMasses(const double* __restrict__ radii, double* __restrict__ vertex_masses, const double* __restrict__ particle_masses) {
    long vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_id >= d_n_vertices) return;
    long particle_id = d_vertex_particle_index_ptr[vertex_id];
    vertex_masses[vertex_id] = particle_masses[particle_id] / static_cast<double>(d_num_vertices_in_particle_ptr[particle_id]);
}

__global__ void kernelGetMomentsOfInertia(const double* __restrict__ positions_x, const double* __restrict__ positions_y, const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y, const double* __restrict__ vertex_masses, double* __restrict__ moments_of_inertia) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    long first_vertex_index = d_particle_start_index_ptr[particle_id];
    long num_vertices = d_num_vertices_in_particle_ptr[particle_id];
    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];
    double mass = vertex_masses[first_vertex_index];  // assuming uniform mass for vertices within a particle
    double rel_pos_x = 0.0;
    double rel_pos_y = 0.0;
    double moment_of_inertia = 0.0;
    for (long i = 0; i < num_vertices; i++) {
        rel_pos_x = vertex_positions_x[first_vertex_index + i] - pos_x;
        rel_pos_y = vertex_positions_y[first_vertex_index + i] - pos_y;
        moment_of_inertia += mass * (rel_pos_x * rel_pos_x + rel_pos_y * rel_pos_y);
    }
    moments_of_inertia[particle_id] = moment_of_inertia;
}

// ----------------------------------------------------------------------
// ---------------------------- Geometry --------------------------------
// ----------------------------------------------------------------------


__global__ void kernelCalculateParticlePolygonArea(
    const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y,
    double* __restrict__ particle_area
) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    double temp_area = 0.0;
    long first_vertex_index = d_particle_start_index_ptr[particle_id];
    long num_vertices = d_num_vertices_in_particle_ptr[particle_id];

    // Prefetch first vertex to avoid duplicate reads
    double pos_x = vertex_positions_x[first_vertex_index];
    double pos_y = vertex_positions_y[first_vertex_index];

    // Loop over vertices to compute the area
    for (long i = 1; i <= num_vertices; i++) {
        long next_index = first_vertex_index + (i % num_vertices);
        double next_pos_x = vertex_positions_x[next_index];
        double next_pos_y = vertex_positions_y[next_index];

        temp_area += pos_x * next_pos_y - next_pos_x * pos_y;

        // Move to the next vertex
        pos_x = next_pos_x;
        pos_y = next_pos_y;
    }
    particle_area[particle_id] = abs(temp_area) * 0.5;
}

__global__ void kernelCalculateBumpyParticleAreaFull(
    const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y,
    const double* __restrict__ vertex_radii,
    double* __restrict__ particle_area
) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    double temp_polygon_area = 0.0;
    double exposed_vertex_area = 0.0;
    double temp_exposed_vertex_area = 0.0;
    double vertex_area = M_PI * d_vertex_radius * d_vertex_radius;

    long first_vertex_index = d_particle_start_index_ptr[particle_id];
    long num_vertices = d_num_vertices_in_particle_ptr[particle_id];

    // Load initial positions
    double pos_x = vertex_positions_x[first_vertex_index];
    double pos_y = vertex_positions_y[first_vertex_index];
    double prev_pos_x = vertex_positions_x[first_vertex_index + num_vertices - 1];
    double prev_pos_y = vertex_positions_y[first_vertex_index + num_vertices - 1];

    // Loop over vertices to compute the area
    for (long i = 0; i < num_vertices; i++) {
        temp_exposed_vertex_area = 0.0;

        // Calculate next vertex index with wrapping
        long next_index = first_vertex_index + ((i + 1) % num_vertices);
        double next_pos_x = vertex_positions_x[next_index];
        double next_pos_y = vertex_positions_y[next_index];

        // Calculate polygon area
        temp_polygon_area += pos_x * next_pos_y - next_pos_x * pos_y;

        // Calculate angle area
        double angle = angleBetweenVectors(next_pos_x, next_pos_y, pos_x, pos_y, prev_pos_x, prev_pos_y);

        // Calculate next vertex overlap area
        double dx = next_pos_x - pos_x;
        double dy = next_pos_y - pos_y;
        double r_ij = sqrt(dx * dx + dy * dy);

        if (r_ij < (2 * d_vertex_radius - 1e-10)) {  // have to give some offset to prevent numerical errors - not good!
            // remove the exposed half of the overlap area from both vertices
            temp_exposed_vertex_area -= calcOverlapLenseArea(r_ij, d_vertex_radius, d_vertex_radius) / 2.0;
        }

        exposed_vertex_area += vertex_area * (M_PI - angle) / (2.0 * M_PI) - temp_exposed_vertex_area;

        // Update positions for next iteration
        prev_pos_x = pos_x;
        prev_pos_y = pos_y;
        pos_x = next_pos_x;
        pos_y = next_pos_y;
    }

    particle_area[particle_id] = abs(temp_polygon_area) * 0.5 + exposed_vertex_area;
}

__global__ void kernelCalculateParticlePositions(
    const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y,
    double* __restrict__ particle_positions_x, double* __restrict__ particle_positions_y
) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    double pos_x = 0.0;
    double pos_y = 0.0;

    long first_vertex_index = d_particle_start_index_ptr[particle_id];
    long num_vertices = d_num_vertices_in_particle_ptr[particle_id];

    // Loop over vertices to compute the area
    for (long i = first_vertex_index; i < first_vertex_index + num_vertices; i++) {
        pos_x += vertex_positions_x[i];
        pos_y += vertex_positions_y[i];
    }

    particle_positions_x[particle_id] = pos_x / num_vertices;
    particle_positions_y[particle_id] = pos_y / num_vertices;
}

__global__ void kernelScalePositions(
    double* __restrict__ positions_x, double* __restrict__ positions_y,
    double* __restrict__ vertex_positions_x, double* __restrict__ vertex_positions_y,
    const double scale_factor
) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;


    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];

    double new_pos_x = pos_x * scale_factor;
    double new_pos_y = pos_y * scale_factor;

    positions_x[particle_id] = new_pos_x;
    positions_y[particle_id] = new_pos_y;

    long first_vertex_index = d_particle_start_index_ptr[particle_id];
    long num_vertices = d_num_vertices_in_particle_ptr[particle_id];

    for (long i = first_vertex_index; i < first_vertex_index + num_vertices; i++) {
        vertex_positions_x[i] += (new_pos_x - pos_x);
        vertex_positions_y[i] += (new_pos_y - pos_y);
    }
}
