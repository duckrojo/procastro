import numpy as np
import astropy.units as u
import dataproc as dp
import matplotlib.patches as mp


__all__ = ['LogoPato',
           ]


def _line(xy, length, angle=0,
          other_boundary=None, to_shape=False, sort_x=True):

    x0, y0 = xy

    x1 = x0 + length*np.cos(angle)
    y1 = y0 + length*np.sin(angle)

    if other_boundary is not None:
        ret_xx = list(other_boundary[0]) + [x1, x0]
        ret_yy = list(other_boundary[1]) + [y1, y0]
    else:
        ret_xx = [x0, x1]
        ret_yy = [y0, y1]

    if to_shape:
        return np.column_stack([ret_xx, ret_yy])
    else:
        if sort_x:
            idx = np.argmin(ret_xx)
        else:
            idx = np.argmin(ret_yy)
        ret_xx = [ret_xx[idx], ret_xx[(idx+1) % 2]]
        ret_yy = [ret_yy[idx], ret_yy[(idx+1) % 2]]
        return ret_xx, ret_yy


def _circle(radius, resolution=80,
            other_boundary=None, close=False, to_shape=False):
    angle = np.linspace(0, 360, resolution)*u.deg
    xx = radius * np.cos(angle)
    yy = radius * np.sin(angle)

    if other_boundary is not None:
        ret_xx = list(other_boundary[0]) + list(xx[::-1])
        ret_yy = list(other_boundary[1]) + list(yy[::-1])
        if close:
            ret_xx += [xx[-1], other_boundary[0][0]]
            ret_yy += [yy[-1], other_boundary[1][0]]
        ret_xx = np.array(ret_xx)
        ret_yy = np.array(ret_yy)

    else:
        ret_xx = xx
        ret_yy = yy

    if to_shape:
        return np.column_stack([ret_xx, ret_yy])
    else:
        return ret_xx, ret_yy


def _intersect_line_line(xy1, angle1, xy2, angle2):

    if angle1.to(u.deg).value == angle2.to(u.deg).value:
        raise ValueError("Lines do not intersect")

    if angle1.to(u.deg).value == 90:
        m2 = np.tan(angle2)
        n2 = xy2[1] - m2 * xy2[0]
        x = xy1[0]
        y = xy1[0]*m2+n2
    elif angle2.to(u.deg).value == 90:
        m1 = np.tan(angle1)
        n1 = xy1[1] - m1 * xy1[0]
        x = xy2[0]
        y = xy2[0]*m1+n1
    else:
        m2 = np.tan(angle2)
        n2 = xy2[1] - m2 * xy2[0]
        m1 = np.tan(angle1)
        n1 = xy1[1] - m1 * xy1[0]
        x = (n2-n1)/(m1-m2)
        y = m1*x + n1

    return x, y


def _intersect_line_circle(radius, xy, angle):

    if angle.to(u.deg).value == 90:
        return [xy[0]]*2, [radius*radius - xy[0]*xy[0]]*2

    m = np.tan(angle).to(u.dimensionless_unscaled).value
    n = xy[1] - m*xy[0]
    x = (-2*m*n + np.array([-1, 1]) * np.sqrt((2*m*n)**2 - 4*(1+m*m)*(n*n-radius*radius))) / 2 / (1+m*m)
    y = m*x + n
    return x, y


def _start_length(x, y):
    dx = x[1]-x[0]
    dy = y[1]-y[0]

    return (x[0], y[0]), np.sqrt(dx*dx + dy*dy), np.arctan2(dy, dx)


class LogoPato:
    def __init__(self,
                 ax=None,
                 width=0.1, o_diameter=1,
                 t_top_fraction=0.5,
                 a_offset_frc=-0.1, a_angle_deg=65,
                 a_bottom_fraction=-0.2,
                 p_extent_fraction=0.45, p_height_fraction=0.4):
        color = 'black'
        a_angle = a_angle_deg*u.deg

        hw = width/2
        cr = o_diameter/2

        f, ax = dp.figaxes(ax)

        # O
        circle_out = _circle(cr + hw)
        circle_border = _circle(cr - hw, close=True, to_shape=True,
                                other_boundary=circle_out)
        ax.add_patch(mp.Polygon(circle_border, color=color))

        # T
        horz_low_line = _intersect_line_circle(cr, (0, t_top_fraction * cr - hw), 0 * u.deg)
        horz_top_line = _intersect_line_circle(cr, (0, t_top_fraction * cr + hw), 0 * u.deg)
        line_border = _line(*_start_length(*horz_top_line),
                            to_shape=True, other_boundary=horz_low_line)
        ax.add_patch(mp.Polygon(line_border, color=color))

        vert = np.sqrt(cr*cr - hw*hw)
        vert_left = _line((-hw, -vert), vert + t_top_fraction * cr - hw, 90 * u.deg)
        ax.add_patch(mp.Polygon(_line((hw, -vert), vert + t_top_fraction * cr - hw, 90 * u.deg,
                                      to_shape=True, other_boundary=vert_left), color=color))

        # A
        delta = hw/np.sin(a_angle).to(u.dimensionless_unscaled).value
        xy = (a_offset_frc*cr - delta, t_top_fraction*cr)
        diag_low_line = _intersect_line_circle(cr, xy, a_angle)
        diag_low_line[0][1], diag_low_line[1][1] = xy
        xy = (a_offset_frc*cr + delta, t_top_fraction*cr)
        diag_top_line = _intersect_line_circle(cr, xy, a_angle)
        diag_top_line[0][1], diag_top_line[1][1] = xy

        line_border = _line(*_start_length(*diag_top_line),
                            to_shape=True, other_boundary=diag_low_line)
        ax.add_patch(mp.Polygon(line_border, color=color))

        xyh = (-hw, a_bottom_fraction*cr-hw)
        horz_bottom_line = np.column_stack([xyh, _intersect_line_line(xyh, 0 * u.deg, xy, a_angle)])
        xyh = (-hw, a_bottom_fraction*cr+hw)
        horz_top_line = np.column_stack([xyh, _intersect_line_line(xyh, 0 * u.deg, xy, a_angle)])

        line_border = _line(*_start_length(*horz_top_line),
                            to_shape=True, other_boundary=horz_bottom_line)
        ax.add_patch(mp.Polygon(line_border, color=color))

        # P
        deg = np.linspace(-90, 90, 20)*u.deg
        xy = (hw, (p_height_fraction+a_bottom_fraction)*cr)
        xx = xy[0] + (cr*p_extent_fraction-hw)*np.cos(deg).to(u.dimensionless_unscaled).value
        yy = xy[1] + (cr*p_height_fraction-hw)*np.sin(deg).to(u.dimensionless_unscaled).value

        out_p_x = xy[0] + (cr*p_extent_fraction+hw)*np.cos(deg).to(u.dimensionless_unscaled).value
        out_p_y = xy[1] + (cr*p_height_fraction+hw)*np.sin(deg).to(u.dimensionless_unscaled).value
        bool = out_p_y < t_top_fraction*cr - hw
        bool[np.argmax(bool == False)] = True

        xx = list(xx) + list(out_p_x[bool])[::-1]
        yy = list(yy) + list(out_p_y[bool])[::-1]
        ax.add_patch(mp.Polygon(np.column_stack([xx, yy]), color=color))

        f.show()
