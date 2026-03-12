#include "stator_c/geometry_builder.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

#define PI 3.14159265358979323846

/* ── Static helpers ─────────────────────────────────────────────────────── */

void stator_geom_rotate(double x, double y, double theta,
                         double* out_x, double* out_y) {
    double c = cos(theta), s = sin(theta);
    *out_x = x * c - y * s;
    *out_y = x * s + y * c;
}

double stator_geom_slot_angle(int k, int n_slots) {
    return 2.0 * PI * (double)k / (double)n_slots;
}

/* Add a point rotated by theta from local (x,y) to global */
static int add_rotated_point(GmshBackend* b, double x, double y,
                               double theta, double mesh_size) {
    double gx, gy;
    stator_geom_rotate(x, y, theta, &gx, &gy);
    return gmsh_add_point(b, gx, gy, 0.0, mesh_size);
}

/* ── stator_geom_builder_init ─────────────────────────────────────────── */

int stator_geom_builder_init(GeometryBuilder* gb, GmshBackend* backend,
                               char* err_buf, size_t err_len) {
    if (!backend) {
        STATOR_SET_ERR(err_buf, err_len, "GeometryBuilder: backend must not be null");
        return STATOR_ERR_INVAL;
    }
    gb->backend = backend;
    return STATOR_OK;
}

/* ── Slot shape builders ─────────────────────────────────────────────── */

static void build_rectangular(GeometryBuilder* gb, const StatorParams* p,
                                int k, SlotProfile* sp) {
    double th  = stator_geom_slot_angle(k, p->n_slots);
    double hw  = p->slot_width_inner * 0.5;
    double ri  = p->R_inner;
    double ro  = ri + p->slot_depth;

    int pt1 = add_rotated_point(gb->backend, ri, +hw, th, 0.0);
    int pt2 = add_rotated_point(gb->backend, ri, -hw, th, 0.0);
    int pt3 = add_rotated_point(gb->backend, ro, -hw, th, 0.0);
    int pt4 = add_rotated_point(gb->backend, ro, +hw, th, 0.0);

    int l1 = gmsh_add_line(gb->backend, pt1, pt2);
    int l2 = gmsh_add_line(gb->backend, pt2, pt3);
    int l3 = gmsh_add_line(gb->backend, pt3, pt4);
    int l4 = gmsh_add_line(gb->backend, pt4, pt1);

    int cl_tags[4] = {l1, l2, l3, l4};
    int cl = gmsh_add_curve_loop(gb->backend, cl_tags, 4);
    sp->slot_surface    = gmsh_add_plane_surface(gb->backend, &cl, 1);
    sp->mouth_curve_bot = l1;
    sp->slot_idx        = k;
    sp->angle           = th;
}

static void build_trapezoidal(GeometryBuilder* gb, const StatorParams* p,
                                int k, SlotProfile* sp) {
    double th  = stator_geom_slot_angle(k, p->n_slots);
    double hwi = p->slot_width_inner * 0.5;
    double hwo = p->slot_width_outer * 0.5;
    double ri  = p->R_inner;
    double ro  = ri + p->slot_depth;

    int pt1 = add_rotated_point(gb->backend, ri, +hwi, th, 0.0);
    int pt2 = add_rotated_point(gb->backend, ri, -hwi, th, 0.0);
    int pt3 = add_rotated_point(gb->backend, ro, -hwo, th, 0.0);
    int pt4 = add_rotated_point(gb->backend, ro, +hwo, th, 0.0);

    int l1 = gmsh_add_line(gb->backend, pt1, pt2);
    int l2 = gmsh_add_line(gb->backend, pt2, pt3);
    int l3 = gmsh_add_line(gb->backend, pt3, pt4);
    int l4 = gmsh_add_line(gb->backend, pt4, pt1);

    int cl_tags[4] = {l1, l2, l3, l4};
    int cl = gmsh_add_curve_loop(gb->backend, cl_tags, 4);
    sp->slot_surface    = gmsh_add_plane_surface(gb->backend, &cl, 1);
    sp->mouth_curve_bot = l1;
    sp->slot_idx        = k;
    sp->angle           = th;
}

static void build_round_bottom(GeometryBuilder* gb, const StatorParams* p,
                                 int k, SlotProfile* sp) {
    double th  = stator_geom_slot_angle(k, p->n_slots);
    double hw  = p->slot_width_inner * 0.5;
    double ri  = p->R_inner;
    double hwo = p->slot_width_outer * 0.5;
    double r_straight = ri + p->slot_depth - hwo;

    int pt_tl    = add_rotated_point(gb->backend, ri,             +hw,  th, 0.0);
    int pt_tr    = add_rotated_point(gb->backend, ri,             -hw,  th, 0.0);
    int pt_rj    = add_rotated_point(gb->backend, r_straight,     -hw,  th, 0.0);
    int pt_lj    = add_rotated_point(gb->backend, r_straight,     +hw,  th, 0.0);
    int pt_arc_c = add_rotated_point(gb->backend, r_straight,      0.0, th, 0.0);
    int pt_bot   = add_rotated_point(gb->backend, r_straight+hwo,  0.0, th, 0.0);

    int l_top = gmsh_add_line(gb->backend, pt_tl, pt_tr);
    int l_r   = gmsh_add_line(gb->backend, pt_tr, pt_rj);
    int arc1  = gb->backend->ops->add_arc(gb->backend->impl, pt_rj, pt_arc_c, pt_bot);
    int arc2  = gb->backend->ops->add_arc(gb->backend->impl, pt_bot, pt_arc_c, pt_lj);
    int l_l   = gmsh_add_line(gb->backend, pt_lj, pt_tl);

    int cl_tags[5] = {l_top, l_r, arc1, arc2, l_l};
    int cl = gmsh_add_curve_loop(gb->backend, cl_tags, 5);
    sp->slot_surface    = gmsh_add_plane_surface(gb->backend, &cl, 1);
    sp->mouth_curve_bot = l_top;
    sp->slot_idx        = k;
    sp->angle           = th;
}

static void build_semi_closed(GeometryBuilder* gb, const StatorParams* p,
                                int k, SlotProfile* sp) {
    double th = stator_geom_slot_angle(k, p->n_slots);
    double r0 = p->R_inner;
    double r1 = p->R_inner + p->slot_opening_depth;
    double r2 = r1 + p->slot_depth;

    double hw_mouth    = p->slot_opening     * 0.5;
    double hw_shoulder = p->slot_width_inner * 0.5;
    double hw_bottom   = p->slot_width_outer * 0.5;

    int pt1 = add_rotated_point(gb->backend, r0, +hw_mouth,    th, 0.0);
    int pt2 = add_rotated_point(gb->backend, r0, -hw_mouth,    th, 0.0);
    int pt3 = add_rotated_point(gb->backend, r1, +hw_shoulder, th, 0.0);
    int pt4 = add_rotated_point(gb->backend, r1, -hw_shoulder, th, 0.0);
    int pt5 = add_rotated_point(gb->backend, r2, +hw_bottom,   th, 0.0);
    int pt6 = add_rotated_point(gb->backend, r2, -hw_bottom,   th, 0.0);

    int l_mouth_top = gmsh_add_line(gb->backend, pt1, pt2);
    int l_mouth_rhs = gmsh_add_line(gb->backend, pt2, pt4);
    int l_wall_r    = gmsh_add_line(gb->backend, pt4, pt6);
    int l_bottom    = gmsh_add_line(gb->backend, pt6, pt5);
    int l_wall_l    = gmsh_add_line(gb->backend, pt5, pt3);
    int l_mouth_lhs = gmsh_add_line(gb->backend, pt3, pt1);

    int cl_tags[6] = {l_mouth_top, l_mouth_rhs, l_wall_r,
                       l_bottom, l_wall_l, l_mouth_lhs};
    int cl = gmsh_add_curve_loop(gb->backend, cl_tags, 6);
    sp->slot_surface    = gmsh_add_plane_surface(gb->backend, &cl, 1);
    sp->mouth_curve_bot = l_mouth_top;
    sp->mouth_curve_top = l_bottom;
    sp->slot_idx        = k;
    sp->angle           = th;
}

/* ── build_coil_inside_slot ─────────────────────────────────────────────── */

static void build_coil_inside_slot(GeometryBuilder* gb,
                                     const StatorParams* p, SlotProfile* sp) {
    double th  = sp->angle;
    double ins = p->insulation_thickness;
    double ri  = p->R_inner + p->slot_opening_depth + ins;

    if (p->winding_type == WINDING_DOUBLE_LAYER) {
        double half_depth = (p->coil_depth - 2.0 * ins) / 2.0;
        double ru0 = ri, ru1 = ru0 + half_depth;
        double hwi = p->coil_width_inner * 0.5;
        double hwo = p->coil_width_outer * 0.5;

        int u1 = add_rotated_point(gb->backend, ru0, +hwi, th, 0.0);
        int u2 = add_rotated_point(gb->backend, ru0, -hwi, th, 0.0);
        int u3 = add_rotated_point(gb->backend, ru1, -hwo, th, 0.0);
        int u4 = add_rotated_point(gb->backend, ru1, +hwo, th, 0.0);
        int ul1 = gmsh_add_line(gb->backend, u1, u2);
        int ul2 = gmsh_add_line(gb->backend, u2, u3);
        int ul3 = gmsh_add_line(gb->backend, u3, u4);
        int ul4 = gmsh_add_line(gb->backend, u4, u1);
        int ucl_tags[4] = {ul1, ul2, ul3, ul4};
        int ucl = gmsh_add_curve_loop(gb->backend, ucl_tags, 4);
        sp->coil_upper_sf = gmsh_add_plane_surface(gb->backend, &ucl, 1);

        double rl0 = ru1 + 2.0 * ins, rl1 = rl0 + half_depth;
        int l1 = add_rotated_point(gb->backend, rl0, +hwi, th, 0.0);
        int l2 = add_rotated_point(gb->backend, rl0, -hwi, th, 0.0);
        int l3 = add_rotated_point(gb->backend, rl1, -hwo, th, 0.0);
        int l4 = add_rotated_point(gb->backend, rl1, +hwo, th, 0.0);
        int ll1 = gmsh_add_line(gb->backend, l1, l2);
        int ll2 = gmsh_add_line(gb->backend, l2, l3);
        int ll3 = gmsh_add_line(gb->backend, l3, l4);
        int ll4 = gmsh_add_line(gb->backend, l4, l1);
        int lcl_tags[4] = {ll1, ll2, ll3, ll4};
        int lcl = gmsh_add_curve_loop(gb->backend, lcl_tags, 4);
        sp->coil_lower_sf = gmsh_add_plane_surface(gb->backend, &lcl, 1);
    } else {
        double rc0 = ri, rc1 = rc0 + p->coil_depth;
        double hwi = p->coil_width_inner * 0.5;
        double hwo = p->coil_width_outer * 0.5;

        int c1 = add_rotated_point(gb->backend, rc0, +hwi, th, 0.0);
        int c2 = add_rotated_point(gb->backend, rc0, -hwi, th, 0.0);
        int c3 = add_rotated_point(gb->backend, rc1, -hwo, th, 0.0);
        int c4 = add_rotated_point(gb->backend, rc1, +hwo, th, 0.0);
        int cl1 = gmsh_add_line(gb->backend, c1, c2);
        int cl2 = gmsh_add_line(gb->backend, c2, c3);
        int cl3 = gmsh_add_line(gb->backend, c3, c4);
        int cl4 = gmsh_add_line(gb->backend, c4, c1);
        int ccl_tags[4] = {cl1, cl2, cl3, cl4};
        int ccl = gmsh_add_curve_loop(gb->backend, ccl_tags, 4);
        sp->coil_upper_sf = gmsh_add_plane_surface(gb->backend, &ccl, 1);
        sp->coil_lower_sf = -1;
    }
}

/* ── build_insulation ────────────────────────────────────────────────────── */

static int make_ins_surface(GeometryBuilder* gb,
                              double r0, double r1,
                              double hwi, double hwo,
                              double ins, double th) {
    double er0  = r0 - ins, er1  = r1 + ins;
    double ehwi = hwi + ins, ehwo = hwo + ins;
    int i1 = add_rotated_point(gb->backend, er0, +ehwi, th, 0.0);
    int i2 = add_rotated_point(gb->backend, er0, -ehwi, th, 0.0);
    int i3 = add_rotated_point(gb->backend, er1, -ehwo, th, 0.0);
    int i4 = add_rotated_point(gb->backend, er1, +ehwo, th, 0.0);
    int il1 = gmsh_add_line(gb->backend, i1, i2);
    int il2 = gmsh_add_line(gb->backend, i2, i3);
    int il3 = gmsh_add_line(gb->backend, i3, i4);
    int il4 = gmsh_add_line(gb->backend, i4, i1);
    int icl_tags[4] = {il1, il2, il3, il4};
    int icl = gmsh_add_curve_loop(gb->backend, icl_tags, 4);
    return gmsh_add_plane_surface(gb->backend, &icl, 1);
}

static void build_insulation(GeometryBuilder* gb,
                               const StatorParams* p, SlotProfile* sp) {
    double th  = sp->angle;
    double ins = p->insulation_thickness;
    double ri  = p->R_inner + p->slot_opening_depth;
    double hwi = p->coil_width_inner * 0.5;
    double hwo = p->coil_width_outer * 0.5;

    if (p->winding_type == WINDING_DOUBLE_LAYER) {
        double half_depth = (p->coil_depth - 2.0 * ins) / 2.0;
        double ru0 = ri + ins, ru1 = ru0 + half_depth;
        sp->ins_upper_sf = make_ins_surface(gb, ru0, ru1, hwi, hwo, ins, th);
        double rl0 = ru1 + 2.0 * ins, rl1 = rl0 + half_depth;
        sp->ins_lower_sf = make_ins_surface(gb, rl0, rl1, hwi, hwo, ins, th);
    } else {
        double rc0 = ri + ins, rc1 = rc0 + p->coil_depth;
        sp->ins_upper_sf = make_ins_surface(gb, rc0, rc1, hwi, hwo, ins, th);
        sp->ins_lower_sf = -1;
    }
}

/* ── stator_geom_build ───────────────────────────────────────────────────── */

int stator_geom_build(GeometryBuilder* gb, const StatorParams* p,
                       GeometryBuildResult* result) {
    memset(result, 0, sizeof(*result));
    result->yoke_surface = -1;
    result->bore_curve   = -1;
    result->outer_curve  = -1;

    if (!gb->backend) {
        snprintf(result->error_message, sizeof(result->error_message),
                 "backend is NULL");
        return STATOR_ERR_INVAL;
    }

    /* Initialise all slots */
    for (int i = 0; i < p->n_slots && i < STATOR_MAX_SLOTS; i++) {
        result->slots[i].slot_idx      = i;
        result->slots[i].slot_surface  = -1;
        result->slots[i].coil_upper_sf = -1;
        result->slots[i].coil_lower_sf = -1;
        result->slots[i].ins_upper_sf  = -1;
        result->slots[i].ins_lower_sf  = -1;
        result->slots[i].mouth_curve_bot = -1;
        result->slots[i].mouth_curve_top = -1;
        result->slots[i].angle = stator_geom_slot_angle(i, p->n_slots);
    }
    result->n_slots = p->n_slots < STATOR_MAX_SLOTS ? p->n_slots : STATOR_MAX_SLOTS;

    /* 1. Outer and inner circles */
    int c_outer = gmsh_add_circle(gb->backend, 0.0, 0.0, 0.0, p->R_outer);
    int c_inner = gmsh_add_circle(gb->backend, 0.0, 0.0, 0.0, p->R_inner);
    result->outer_curve = c_outer;
    result->bore_curve  = c_inner;

    /* 2. Yoke annular surface */
    int cl_outer = gmsh_add_curve_loop(gb->backend, &c_outer, 1);
    int neg_c_inner = -c_inner;
    int cl_inner = gmsh_add_curve_loop(gb->backend, &neg_c_inner, 1);
    int loop_tags[2] = {cl_outer, cl_inner};
    int yoke_sf = gmsh_add_plane_surface(gb->backend, loop_tags, 2);
    result->yoke_surface = yoke_sf;

    /* 3. Build all slot profiles */
    IntPair slot_surfaces[STATOR_MAX_SLOTS];
    int n_slot_sf = 0;

    for (int k = 0; k < result->n_slots; k++) {
        SlotProfile* sp = &result->slots[k];
        /* reset */
        sp->slot_idx = k;
        sp->angle    = stator_geom_slot_angle(k, p->n_slots);
        sp->slot_surface = sp->coil_upper_sf = sp->coil_lower_sf = -1;
        sp->ins_upper_sf = sp->ins_lower_sf = -1;
        sp->mouth_curve_bot = sp->mouth_curve_top = -1;

        switch (p->slot_shape) {
            case SLOT_SHAPE_RECTANGULAR:  build_rectangular (gb, p, k, sp); break;
            case SLOT_SHAPE_TRAPEZOIDAL:  build_trapezoidal (gb, p, k, sp); break;
            case SLOT_SHAPE_ROUND_BOTTOM: build_round_bottom(gb, p, k, sp); break;
            case SLOT_SHAPE_SEMI_CLOSED:  build_semi_closed (gb, p, k, sp); break;
            default: build_rectangular(gb, p, k, sp); break;
        }

        build_coil_inside_slot(gb, p, sp);
        build_insulation(gb, p, sp);

        if (sp->slot_surface >= 0) {
            slot_surfaces[n_slot_sf].first  = 2;
            slot_surfaces[n_slot_sf].second = sp->slot_surface;
            n_slot_sf++;
        }
    }

    /* 4. Boolean cut: carve slots out of yoke */
    if (n_slot_sf > 0) {
        IntPair obj = {2, yoke_sf};
        IntPair out_pairs[STATOR_MAX_SLOTS + 1];
        int out_n = 0;
        gb->backend->ops->boolean_cut(gb->backend->impl,
            &obj, 1,
            slot_surfaces, n_slot_sf,
            0,  /* remove_tool = false */
            out_pairs, &out_n, STATOR_MAX_SLOTS + 1);
    }

    /* 5. Synchronize */
    gmsh_synchronize(gb->backend);

    result->success = true;
    return STATOR_OK;
}
