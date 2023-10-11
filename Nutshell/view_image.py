import matplotlib.pyplot


class ViewImage:
    def __init__(self):
        matplotlib.pyplot.ion()
        self.fig, self.axis = matplotlib.pyplot.subplots()
        self.fig.canvas.manager.set_window_title('Image')
        self.axis.set_axis_off()
        self.image = None

    def apply(self, image):
        if self.image is None:
            self.image = self.axis.imshow(image)
        else:
            self.image.set_array(image)
        self.fig.canvas.flush_events()
