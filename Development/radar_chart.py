"""
Example of creating a radar chart (a.k.a. a spider or star chart) [1]_.

Although this example allows a frame of either 'circle' or 'polygon', polygon
frames don't have proper gridlines (the lines are circles instead of polygons).
It's possible to get a polygon grid by setting GRIDLINE_INTERPOLATION_STEPS in
matplotlib.axis to the desired number of vertices, but the orientation of the
polygon is not aligned with the radial axes.

.. [1] http://en.wikipedia.org/wiki/Radar_chart
"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = 2*np.pi * np.linspace(0, 1-1./num_vars, num_vars)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(theta * 180/np.pi, labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.
    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts


def plot(data, spoke_labels, sensor_labels,saveto=None,frame_type='polygon'):
    theta = radar_factory(len(spoke_labels), frame=frame_type)
    fig = plt.figure(figsize=(9, 9))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    ax = fig.add_subplot(111, projection="radar")
    plt.rgrids([np.round(0.1 + i / 10.0,2) for i in range(10)])
    for d in data:
        ax.plot(theta, d)
        ax.fill(theta, d, alpha=0.2)
    
    ax.set_varlabels(spoke_labels)
    
    legend = plt.legend(sensor_labels, loc=(0.0, 0.9), labelspacing=0.1)
    plt.setp(legend.get_texts(), fontsize='small')
    
    if saveto != None:
        plt.savefig(saveto)
        
        
        
def subplot(data, spoke_labels, sensor_labels,saveto=None,frame_type='polygon'):
    #def subplot(data, spoke_labels, sensor_labels,saveto=None,frame_type='polygon'):
    num_of_picks=9
    theta = radar_factory(len(spoke_labels), frame='circle')
    fig = plt.figure(figsize=(num_of_picks, num_of_picks))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    num_col=np.floor(np.sqrt(len(data)))
    num_row=np.ceil(num_of_picks/num_col)
    for k,(data_col,sensor_label_) in enumerate(zip(data,sensor_labels)):
        #subplot(num_col,num_row,i+1)
        ax = fig.add_subplot(num_col,num_row,k+1, projection="radar")
        ax.plot(theta, data_col)
        ax.fill(theta, data_col, alpha=0.2)
        ax.set_varlabels(spoke_labels)
        #plt.title(sensor_label_,fontsize='small')
        legend = plt.legend([sensor_label_], loc=(-0.2, 1.1), labelspacing=0.01)
        plt.setp(legend.get_texts(), fontsize='small')
        radar_bnd=max(max(data))
        #import pdb;pdb.set_trace()
        rgrid_spacing=np.round(list(np.arange(0.1,radar_bnd,float(radar_bnd)/5)),2)
        plt.rgrids(rgrid_spacing)
        #plt.rgrids([0.1 + 2*i / 10.0 for i in range(radar_bnd)])
        ##radar_chart.plot(data_col, spoke_labels, sensor_label_, saveto="time_radar.png",frame_type='circle')    
    if saveto != None:
        plt.savefig(saveto)        
    
  


