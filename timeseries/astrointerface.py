import PIL.Image as PILimg
import ImageTk
import numpy as np
import tkMessageBox as box

from Tkinter import *
from scipy.ndimage.interpolation import zoom
from img_scale import linear, sqrt, log, power, asinh, histeq, range_from_zscale


class AstroInterface(object):

    """This class implements a GUI for pixel coordinates selection in an astronomical image.
    """

    def __init__(self, data, maxsize=500):
        """AstroInterface object constructor.

        :param data: image
        :type data: array
        :param maxsize: pixel size in the GUI of the maximum dimension of the image (default value 500)
        :type maxsize: int
        :rtype: AstroInterface
        """

        # MAIN WINDOW AND EXIT BUTTON
        root = Tk()
        self.root = root
        self.root.protocol('WM_DELETE_WINDOW', self._exit_button)

        # INSTANCE VARIABLES
        maxdim = np.max(data.shape)
        self.scale_factor = float(maxsize) / maxdim
        self.data = zoom(data, self.scale_factor)
        self.data = self.data[::-1]  # CHANGE OF ORIGIN FOR IMAGE DISPLAY
        self.ver, self.hor = self.data.shape
        self.zscale = None
        self.photo = self._get_photo_image()

        # DYNAMIC PIXEL COORDINATES FROM POINTS IN THE IMAGE (INT VALUES)
        self.dynamic_pixcoo = []
        # DYNAMIC POINTS COORDINATES CONVERTED FROM DYNAMIC PIXEL COORDINATES
        # (FLOAT VALUES)
        self.dynamic_points = []
        # DINAMIC FRAME THAT CONTAINS THE LIST OF SELECTED BUTTONS IN THE RIGHT
        # SIDE OF THE GUI
        self.dynamic_frame = Frame()

        # MAIN FRAMES: F1 (LEFT SIDE OF THE GUI) AND F2 (RIGHT SIDE)
        F1 = Frame(root)
        F1.pack(side=LEFT)
        F2 = Frame(root)
        F2.pack(side=LEFT)

        # SCALE BUTTONS
        Fbuttons = Frame(F1)
        self.B1 = Button(Fbuttons, text='zscale')
        self.B1.grid(row=1, column=1)
        self.B2 = Button(Fbuttons, text='linear')
        self.B2.grid(row=1, column=2)
        self.B3 = Button(
            Fbuttons,
            text='sqrt')
        self.B3.grid(
            row=1,
            column=3)
        self.B4 = Button(
            Fbuttons,
            text='log')
        self.B4.grid(
            row=1,
            column=4)
        self.B5 = Button(
            Fbuttons,
            text='power')
        self.B5.grid(
            row=1,
            column=5)
        self.B6 = Button(
            Fbuttons,
            text='asinh')
        self.B6.grid(
            row=1,
            column=6)
        self.B7 = Button(
            Fbuttons,
            text='histogram')
        self.B7.grid(
            row=1,
            column=7)

        self.prev_pressed_button = self.B1
        self.prev_pressed_button.configure(bg='#a2a2a2')

        self.B1.bind(
            "<Button-1>",
            lambda event: self._scale_button_action(
                event,
                self.B1,
                self.prev_pressed_button))
        self.B2.bind(
            "<Button-1>",
            lambda event: self._scale_button_action(
                event,
                self.B2,
                self.prev_pressed_button))
        self.B3.bind(
            "<Button-1>",
            lambda event: self._scale_button_action(
                event,
                self.B3,
                self.prev_pressed_button))
        self.B4.bind(
            "<Button-1>",
            lambda event: self._scale_button_action(
                event,
                self.B4,
                self.prev_pressed_button))
        self.B5.bind(
            "<Button-1>",
            lambda event: self._scale_button_action(
                event,
                self.B5,
                self.prev_pressed_button))
        self.B6.bind(
            "<Button-1>",
            lambda event: self._scale_button_action(
                event,
                self.B6,
                self.prev_pressed_button))
        self.B7.bind(
            "<Button-1>",
            lambda event: self._scale_button_action(
                event,
                self.B7,
                self.prev_pressed_button))
        Fbuttons.pack()

        # LABELS FOR PIXEL COORDINATES DISPLAY
        Fcoord = Frame(F1)
        Fcoord.pack()
        self.posx = StringVar(value='')
        self.posy = StringVar(value='')
        Label(Fcoord, text='X     = ', width=8).pack(side=LEFT)
        L1 = Label(Fcoord, textvariable=self.posx, width=8)
        L1.pack(side=LEFT)
        Label(Fcoord, text='\t\t').pack(side=LEFT)
        Label(Fcoord, text='Y     = ', width=8).pack(side=LEFT)
        L2 = Label(Fcoord, textvariable=self.posy, width=8)
        L2.pack(side=LEFT)

        # CANVAS
        self.canvas = Canvas(
            F1,
            background='white',
            width=self.hor,
            height=self.ver)
        self.canvas.create_image(0, 0, anchor=NW, image=self.photo, tags='img')
        self.canvas.pack()
        self.canvas.bind('<Button-1>', self._canvas_on_click_add_point)
        self.canvas.bind('<Motion>', self._canvas_on_motion)

        # SELECTED POINTS LIST
        self.cont = Button(
            F2,
            text='CONTINUE',
            state=DISABLED,
            command=self.root.destroy)
        self.cont.pack()
        self.label = Label(
            F2,
            text='\n            No points selected            \n')
        self.label.pack()
        self.Fpoints = Frame(F2)
        self.Fpoints.pack()

    def execute(self):
        """Executes the mainloop in the AstroInterface object.

        :rtype: None
        """

        self.root.mainloop()
        return self.dynamic_points

    def _exit_button(self):
        """Exit button function. A warning is triggered when the exit button is pressed.

        :rtype: None
        """

        answer = box.askyesno("Warning", "Are you sure you want to exit?")
        if answer:
            self.root.destroy()
            exit('Program finished from interface\n')

    def _get_photo_image(self, mode='zscale'):
        """Returns the scaled version of the image given in the constructor as a PhotoImage instance.

        :param mode: image scale mode (options: 'zscale' (default), 'linear', 'sqrt', 'log', 'power', 'asinh', 'histogram')
        :type mode: str
        :rtype: PhotoImage
        """

        if mode == 'zscale':
            if self.zscale is None:
                self.zscale = range_from_zscale(self.data)
            z1 = self.zscale[0]
            z2 = self.zscale[1]
            scaled_data = self.data.copy()
            scaled_data[scaled_data < z1] = z1
            scaled_data[scaled_data > z2] = z2
        elif mode == 'linear':
            scaled_data = linear(self.data)
        elif mode == 'sqrt':
            scaled_data = sqrt(self.data)
        elif mode == 'log':
            min_val = self.data.min()
            delta = 0.0
            if min_val < 1.0:
                delta = 1.0 - min_val
            scaled_data = log(self.data + delta)
        elif mode == 'power':
            scaled_data = power(self.data)
        elif mode == 'asinh':
            scaled_data = asinh(self.data)
        elif mode == 'histogram':
            ver, hor = self.data.shape
            num_bins = (ver * hor) / 10
            scaled_data = histeq(self.data, num_bins=num_bins)
        else:
            raise ValueError("Unknown mode '%s' for image scaling" % mode)

        print "img_scale: %s" % mode
        scaled_data = (255.0 / scaled_data.max() *
                       (scaled_data - scaled_data.min())).astype(np.uint8)
        return ImageTk.PhotoImage(PILimg.fromarray(scaled_data))

    def _scale_button_action(self, event, pressed_button, prev_pressed_button):
        """Scale buttons function. Updates the scale in the displayed image.

        :param event: event
        :type event: event
        :param pressed_button: scale pressed button
        :type pressed_button: button
        :param prev_pressed_button: scale button pressed previously
        :type prev_pressed_button: button
        :rtype: None
        """

        prev_pressed_button.configure(bg='#d9d9d9')
        pressed_button.configure(bg='#a2a2a2')
        mode = pressed_button['text']
        self.photo = self._get_photo_image(mode=mode)
        self.canvas.itemconfigure('img', image=self.photo)
        self.canvas.update()
        self.prev_pressed_button = pressed_button

    def _update_point_list(self):
        """Update the list of selected points in the right side of the GUI.

        :rtype: None
        """

        self.dynamic_frame.pack_forget()
        self.dynamic_frame.destroy()
        self.dynamic_frame = Frame(self.Fpoints)
        self.dynamic_frame.pack()

        for coo, i in zip(self.dynamic_points, range(len(self.dynamic_points))):
            L = Label(self.dynamic_frame, text=str([coo[0], coo[1]]))
            B = Button(
                self.dynamic_frame,
                text='\xe2\x98\x92'.decode('utf8'),
                command=lambda i=i: self._button_remove_point(i))
            L.grid(row=i + 1, column=1)
            B.grid(row=i + 1, column=2)

        if len(self.dynamic_points) > 0:
            self.cont.configure(state=ACTIVE)
            self.label.configure(
                text='\n              Selected points              \n')
        else:
            self.cont.configure(state=DISABLED)
            self.label.configure(
                text='\n            No points selected            \n')

    def _canvas_draw_circle(self, x, y, rad=4):
        """Draws a red circle at (x,y) coordinates in the canvas.

        :param x: horizontal center coordinate of the circle
        :type x: int
        :param y: vertical center coordinate of the circle
        :type y: int
        :param rad: radius of the circle (default 4)
        :type rad: int
        :rtype: None
        """

        self.canvas.create_oval(
            x - rad,
            y - rad,
            x + rad,
            y + rad,
            width=0,
            fill='red')

    def _update_canvas(self):
        """Updates the canvas content. Displays the image and currently selected points on it.

        :rtype: None
        """

        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor=NW, image=self.photo, tags='img')
        for x, y in self.dynamic_pixcoo:
            self._canvas_draw_circle(x, y)
        self.canvas.update()

    def _button_remove_point(self, index):
        """Removes a point from the the list of selected points.

        :param index: index of the point to be removed in the list of selected points.
        :type index: int
        :rtype: None
        """

        self.dynamic_points.pop(index)
        self.dynamic_pixcoo.pop(index)
        self._update_point_list()
        self._update_canvas()

    def _canvas_on_click_add_point(self, event):
        """Adds a new point to the list of selected points in the right side of the GUI.

        :param event: event
        :type event: event
        :rtype: None
        """

        self.dynamic_pixcoo.append((event.x, event.y))
        posx = round(event.x / self.scale_factor, 2)
        posy = round((self.ver - event.y) / self.scale_factor, 2)
        self.posx.set('%s' % str(posx))
        self.posy.set('%s' % str(posy))
        self.dynamic_points.append([posx, posy])
        self._update_point_list()
        self._update_canvas()

    def _canvas_on_motion(self, event):
        """Updates the pixel coordinates while the cursor is over the image.

        :param event: event
        :type event: event
        :rtype: None
        """

        posx = round(event.x / self.scale_factor, 2)
        posy = round((self.ver - event.y) / self.scale_factor, 2)
        self.posx.set('%s' % str(posx))
        self.posy.set('%s' % str(posy))
