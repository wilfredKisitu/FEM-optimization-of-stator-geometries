#include "stator/geometry_builder.hpp"
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace stator {

static constexpr double PI = std::numbers::pi;

// ─── Constructor ──────────────────────────────────────────────────────────────

GeometryBuilder::GeometryBuilder(IGmshBackend* backend) : backend_(backend) {
    if (!backend_)
        throw std::invalid_argument("GeometryBuilder: backend must not be null");
}

// ─── Static helpers ───────────────────────────────────────────────────────────

std::pair<double,double> GeometryBuilder::rotate(double x, double y, double theta) noexcept {
    return { x * std::cos(theta) - y * std::sin(theta),
             x * std::sin(theta) + y * std::cos(theta) };
}

double GeometryBuilder::slot_angle(int k, int n_slots) noexcept {
    return 2.0 * PI * static_cast<double>(k) / static_cast<double>(n_slots);
}

int GeometryBuilder::add_rotated_point(double x, double y, double theta,
                                       double mesh_size) {
    auto [gx, gy] = rotate(x, y, theta);
    return backend_->add_point(gx, gy, 0.0, mesh_size);
}

// ─── build ────────────────────────────────────────────────────────────────────

GeometryBuildResult GeometryBuilder::build(const StatorParams& p) {
    GeometryBuildResult result;
    try {
        // 1. Outer and inner circles
        int c_outer = backend_->add_circle(0.0, 0.0, 0.0, p.R_outer);
        int c_inner = backend_->add_circle(0.0, 0.0, 0.0, p.R_inner);
        result.outer_curve = c_outer;
        result.bore_curve  = c_inner;

        // 2. Yoke annular surface (outer circle minus bore hole)
        int cl_outer = backend_->add_curve_loop({c_outer});
        int cl_inner = backend_->add_curve_loop({-c_inner});
        int yoke_sf  = backend_->add_plane_surface({cl_outer, cl_inner});
        result.yoke_surface = yoke_sf;

        // 3. Build all slot profiles
        result.slots.reserve(p.n_slots);
        std::vector<std::pair<int,int>> slot_surfaces;
        for (int k = 0; k < p.n_slots; ++k) {
            SlotProfile sp = build_single_slot(p, k);
            if (sp.slot_surface >= 0)
                slot_surfaces.push_back({2, sp.slot_surface});
            result.slots.push_back(sp);
        }

        // 4. Boolean cut: carve slots out of yoke
        if (!slot_surfaces.empty()) {
            backend_->boolean_cut({{2, yoke_sf}}, slot_surfaces, /*remove_tool=*/false);
        }

        // 5. Synchronize model
        backend_->synchronize();

        result.success = true;
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }
    return result;
}

// ─── build_single_slot ────────────────────────────────────────────────────────

SlotProfile GeometryBuilder::build_single_slot(const StatorParams& p, int k) {
    SlotProfile sp;
    sp.slot_idx = k;
    sp.angle    = slot_angle(k, p.n_slots);

    switch (p.slot_shape) {
        case SlotShape::RECTANGULAR:  sp = build_rectangular(p, k);  break;
        case SlotShape::TRAPEZOIDAL:  sp = build_trapezoidal(p, k);  break;
        case SlotShape::ROUND_BOTTOM: sp = build_round_bottom(p, k); break;
        case SlotShape::SEMI_CLOSED:  sp = build_semi_closed(p, k);  break;
    }

    build_coil_inside_slot(p, sp);
    build_insulation(p, sp);
    return sp;
}

// ─── RECTANGULAR slot ─────────────────────────────────────────────────────────
// Points (local frame, CCW):
//   p1: (R_inner, +hw)  p2: (R_inner, -hw)
//   p3: (R_inner+depth, -hw)  p4: (R_inner+depth, +hw)

SlotProfile GeometryBuilder::build_rectangular(const StatorParams& p, int k) {
    SlotProfile sp;
    sp.slot_idx = k;
    sp.angle    = slot_angle(k, p.n_slots);
    double th   = sp.angle;

    double hw = p.slot_width_inner * 0.5;
    double ri = p.R_inner;
    double ro = ri + p.slot_depth;

    int pt1 = add_rotated_point(ri, +hw, th);
    int pt2 = add_rotated_point(ri, -hw, th);
    int pt3 = add_rotated_point(ro, -hw, th);
    int pt4 = add_rotated_point(ro, +hw, th);

    int l1 = backend_->add_line(pt1, pt2); // bore-facing (mouth)
    int l2 = backend_->add_line(pt2, pt3);
    int l3 = backend_->add_line(pt3, pt4); // bottom (yoke side)
    int l4 = backend_->add_line(pt4, pt1);

    int cl = backend_->add_curve_loop({l1, l2, l3, l4});
    sp.slot_surface    = backend_->add_plane_surface({cl});
    sp.mouth_curve_bot = l1;
    return sp;
}

// ─── TRAPEZOIDAL slot ─────────────────────────────────────────────────────────

SlotProfile GeometryBuilder::build_trapezoidal(const StatorParams& p, int k) {
    SlotProfile sp;
    sp.slot_idx = k;
    sp.angle    = slot_angle(k, p.n_slots);
    double th   = sp.angle;

    double hwi = p.slot_width_inner * 0.5;
    double hwo = p.slot_width_outer * 0.5;
    double ri  = p.R_inner;
    double ro  = ri + p.slot_depth;

    int pt1 = add_rotated_point(ri, +hwi, th);
    int pt2 = add_rotated_point(ri, -hwi, th);
    int pt3 = add_rotated_point(ro, -hwo, th);
    int pt4 = add_rotated_point(ro, +hwo, th);

    int l1 = backend_->add_line(pt1, pt2); // bore-facing
    int l2 = backend_->add_line(pt2, pt3);
    int l3 = backend_->add_line(pt3, pt4); // bottom
    int l4 = backend_->add_line(pt4, pt1);

    int cl = backend_->add_curve_loop({l1, l2, l3, l4});
    sp.slot_surface    = backend_->add_plane_surface({cl});
    sp.mouth_curve_bot = l1;
    return sp;
}

// ─── ROUND_BOTTOM slot ────────────────────────────────────────────────────────

SlotProfile GeometryBuilder::build_round_bottom(const StatorParams& p, int k) {
    SlotProfile sp;
    sp.slot_idx = k;
    sp.angle    = slot_angle(k, p.n_slots);
    double th   = sp.angle;

    double hw  = p.slot_width_inner * 0.5;  // half-width at bore
    double ri  = p.R_inner;
    double hwo = p.slot_width_outer * 0.5;

    // Straight walls run to r_straight, then arc closes the bottom
    double r_straight = ri + p.slot_depth - hwo;

    // Top (bore-side) corners
    int pt_tl  = add_rotated_point(ri,         +hw,  th);
    int pt_tr  = add_rotated_point(ri,         -hw,  th);
    // Straight wall base corners
    int pt_rj  = add_rotated_point(r_straight, -hw,  th); // right join
    int pt_lj  = add_rotated_point(r_straight, +hw,  th); // left join

    // Arc centre and mid-point (TWO arcs < 180°)
    // Arc centre in local frame: (r_straight, 0)
    int pt_arc_c = add_rotated_point(r_straight, 0.0, th);
    int pt_bot   = add_rotated_point(r_straight + hwo, 0.0, th);

    // Lines: bore-facing, two straight walls
    int l_top  = backend_->add_line(pt_tl, pt_tr);   // bore
    int l_r    = backend_->add_line(pt_tr, pt_rj);   // right wall
    int arc1   = backend_->add_arc(pt_rj, pt_arc_c, pt_bot);  // right arc
    int arc2   = backend_->add_arc(pt_bot, pt_arc_c, pt_lj);  // left arc
    int l_l    = backend_->add_line(pt_lj, pt_tl);   // left wall

    int cl = backend_->add_curve_loop({l_top, l_r, arc1, arc2, l_l});
    sp.slot_surface    = backend_->add_plane_surface({cl});
    sp.mouth_curve_bot = l_top;
    return sp;
}

// ─── SEMI_CLOSED slot ─────────────────────────────────────────────────────────
// Radial levels:
//   r0 = R_inner
//   r1 = R_inner + slot_opening_depth
//   r2 = R_inner + slot_opening_depth + slot_depth

SlotProfile GeometryBuilder::build_semi_closed(const StatorParams& p, int k) {
    SlotProfile sp;
    sp.slot_idx = k;
    sp.angle    = slot_angle(k, p.n_slots);
    double th   = sp.angle;

    double r0 = p.R_inner;
    double r1 = p.R_inner + p.slot_opening_depth;
    double r2 = r1 + p.slot_depth;

    double hw_mouth    = p.slot_opening        * 0.5;
    double hw_shoulder = p.slot_width_inner    * 0.5;
    double hw_bottom   = p.slot_width_outer    * 0.5;

    // 6 points
    int pt1 = add_rotated_point(r0, +hw_mouth,    th); // p1 mouth left
    int pt2 = add_rotated_point(r0, -hw_mouth,    th); // p2 mouth right
    int pt3 = add_rotated_point(r1, +hw_shoulder, th); // p3 shoulder left
    int pt4 = add_rotated_point(r1, -hw_shoulder, th); // p4 shoulder right
    int pt5 = add_rotated_point(r2, +hw_bottom,   th); // p5 bottom left
    int pt6 = add_rotated_point(r2, -hw_bottom,   th); // p6 bottom right

    // CCW contour:
    int l_mouth_top = backend_->add_line(pt1, pt2);  // bore-facing mouth edge
    int l_mouth_rhs = backend_->add_line(pt2, pt4);
    int l_wall_r    = backend_->add_line(pt4, pt6);
    int l_bottom    = backend_->add_line(pt6, pt5);  // yoke-side bottom
    int l_wall_l    = backend_->add_line(pt5, pt3);
    int l_mouth_lhs = backend_->add_line(pt3, pt1);

    int cl = backend_->add_curve_loop(
        {l_mouth_top, l_mouth_rhs, l_wall_r, l_bottom, l_wall_l, l_mouth_lhs});
    sp.slot_surface    = backend_->add_plane_surface({cl});
    sp.mouth_curve_bot = l_mouth_top;
    sp.mouth_curve_top = l_bottom;
    return sp;
}

// ─── build_coil_inside_slot ──────────────────────────────────────────────────

void GeometryBuilder::build_coil_inside_slot(const StatorParams& p, SlotProfile& sp) {
    double th  = sp.angle;
    double ins = p.insulation_thickness;
    double ri  = p.R_inner + p.slot_opening_depth + ins;

    if (p.winding_type == WindingType::DOUBLE_LAYER) {
        double half_depth = (p.coil_depth - 2.0 * ins) / 2.0;

        // Upper coil
        double ru0 = ri;
        double ru1 = ru0 + half_depth;
        double hwi = p.coil_width_inner * 0.5;
        double hwo = p.coil_width_outer * 0.5;

        int u1 = add_rotated_point(ru0, +hwi, th);
        int u2 = add_rotated_point(ru0, -hwi, th);
        int u3 = add_rotated_point(ru1, -hwo, th);
        int u4 = add_rotated_point(ru1, +hwo, th);
        int ul1 = backend_->add_line(u1, u2);
        int ul2 = backend_->add_line(u2, u3);
        int ul3 = backend_->add_line(u3, u4);
        int ul4 = backend_->add_line(u4, u1);
        int ucl = backend_->add_curve_loop({ul1, ul2, ul3, ul4});
        sp.coil_upper_sf = backend_->add_plane_surface({ucl});

        // Lower coil
        double rl0 = ru1 + 2.0 * ins;
        double rl1 = rl0 + half_depth;
        int l1 = add_rotated_point(rl0, +hwi, th);
        int l2 = add_rotated_point(rl0, -hwi, th);
        int l3 = add_rotated_point(rl1, -hwo, th);
        int l4 = add_rotated_point(rl1, +hwo, th);
        int ll1 = backend_->add_line(l1, l2);
        int ll2 = backend_->add_line(l2, l3);
        int ll3 = backend_->add_line(l3, l4);
        int ll4 = backend_->add_line(l4, l1);
        int lcl = backend_->add_curve_loop({ll1, ll2, ll3, ll4});
        sp.coil_lower_sf = backend_->add_plane_surface({lcl});
    } else {
        // SINGLE_LAYER (and CONCENTRATED / DISTRIBUTED use same shape)
        double rc0 = ri;
        double rc1 = rc0 + p.coil_depth;
        double hwi = p.coil_width_inner * 0.5;
        double hwo = p.coil_width_outer * 0.5;

        int c1 = add_rotated_point(rc0, +hwi, th);
        int c2 = add_rotated_point(rc0, -hwi, th);
        int c3 = add_rotated_point(rc1, -hwo, th);
        int c4 = add_rotated_point(rc1, +hwo, th);
        int cl1 = backend_->add_line(c1, c2);
        int cl2 = backend_->add_line(c2, c3);
        int cl3 = backend_->add_line(c3, c4);
        int cl4 = backend_->add_line(c4, c1);
        int ccl = backend_->add_curve_loop({cl1, cl2, cl3, cl4});
        sp.coil_upper_sf = backend_->add_plane_surface({ccl});
        sp.coil_lower_sf = -1;
    }
}

// ─── build_insulation ─────────────────────────────────────────────────────────

void GeometryBuilder::build_insulation(const StatorParams& p, SlotProfile& sp) {
    double th  = sp.angle;
    double ins = p.insulation_thickness;
    double ri  = p.R_inner + p.slot_opening_depth;

    auto make_ins_surface = [&](double r0, double r1,
                                 double hwi, double hwo) -> int {
        double er0  = r0 - ins;
        double er1  = r1 + ins;
        double ehwi = hwi + ins;
        double ehwo = hwo + ins;
        int i1 = add_rotated_point(er0, +ehwi, th);
        int i2 = add_rotated_point(er0, -ehwi, th);
        int i3 = add_rotated_point(er1, -ehwo, th);
        int i4 = add_rotated_point(er1, +ehwo, th);
        int il1 = backend_->add_line(i1, i2);
        int il2 = backend_->add_line(i2, i3);
        int il3 = backend_->add_line(i3, i4);
        int il4 = backend_->add_line(i4, i1);
        int icl = backend_->add_curve_loop({il1, il2, il3, il4});
        return backend_->add_plane_surface({icl});
    };

    double hwi = p.coil_width_inner * 0.5;
    double hwo = p.coil_width_outer * 0.5;

    if (p.winding_type == WindingType::DOUBLE_LAYER) {
        double half_depth = (p.coil_depth - 2.0 * ins) / 2.0;
        double ru0 = ri + ins;
        double ru1 = ru0 + half_depth;
        sp.ins_upper_sf = make_ins_surface(ru0, ru1, hwi, hwo);

        double rl0 = ru1 + 2.0 * ins;
        double rl1 = rl0 + half_depth;
        sp.ins_lower_sf = make_ins_surface(rl0, rl1, hwi, hwo);
    } else {
        double rc0 = ri + ins;
        double rc1 = rc0 + p.coil_depth;
        sp.ins_upper_sf = make_ins_surface(rc0, rc1, hwi, hwo);
        sp.ins_lower_sf = -1;
    }
}

} // namespace stator
