import matplotlib.pyplot


class View2D:
    def __init__(self, title, label_x, max_x, margin_x, label_y, max_y, margin_y):
        self.label_x = label_x
        self.max_x = max_x
        self.margin_x = margin_x
        self.label_y = label_y
        self.max_y = max_y
        self.margin_y = margin_y

        matplotlib.pyplot.ion()
        self.fig, _ = matplotlib.pyplot.subplots()
        self.fig.canvas.manager.set_window_title(title)

    def reset(self, goal_x, goal_y):
        self.xs = []
        self.ys = []
        self.steps = 0

        self.fig.clear()

        self.axis_input = self.fig.add_subplot()
        self.axis_input.set_xlabel(self.label_x)
        self.axis_input.set_xlim([-self.max_x, self.max_x])
        self.axis_input.set_ylabel(self.label_y)
        self.axis_input.set_ylim([-self.max_y, self.max_y])
        self.axis_input.plot(goal_x, goal_y, color='tab:green', marker='o', linestyle='dashed', label='Optimal input setting')
        self.axis_input.add_patch(matplotlib.patches.Rectangle((goal_x - self.margin_x, goal_y - self.margin_y), 2 * self.margin_x, 2 * self.margin_y, color='lightgreen'))

    def apply(self, x, y):
        self.xs.append(x)
        self.ys.append(y)
        self.steps += 1

        self.axis_input.plot(self.xs, self.ys, color='tab:blue', marker='o', label='Current input setting')
        if self.steps == 1:
            self.fig.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
