import matplotlib.pyplot


class ViewRegression:
    def __init__(self, title, max_input, margin_input):
        self.max_input = max_input
        self.margin_input = margin_input

        matplotlib.pyplot.ion()
        self.fig, _ = matplotlib.pyplot.subplots()
        self.fig.canvas.manager.set_window_title(title)

    def reset(self, goal_input):
        self.goal_input = goal_input
        self.xs = []
        self.ys1 = []
        self.ys2 = []
        self.ys3 = []
        self.steps = 0

        self.fig.clear()

        self.axis_input = self.fig.add_subplot()
        self.axis_input.set_xlabel('Time steps')
        self.axis_input.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, min_n_ticks=1))
        self.axis_input.set_ylabel('Input setting')
        self.axis_input.set_ylim([-self.max_input, self.max_input])
        self.axis_input.axhline(goal_input, color='tab:green', linestyle='dashed', label='Optimal input setting')
        self.axis_input.axhspan(goal_input - self.margin_input, goal_input + self.margin_input, color='lightgreen')

        self.axis_output = self.axis_input.twinx()
        self.axis_output.set_ylabel('Output level')
        self.axis_output.set_ylim([0, 2 * self.max_input])

    def apply(self, actual_input, actual_output):
        self.xs.append(self.steps)
        self.ys1.append(actual_input)
        self.ys2.append(actual_output)
        self.ys3.append(abs(actual_input - self.goal_input))
        self.steps += 1

        self.axis_input.plot(self.xs, self.ys1, color='tab:blue', marker='o', label='Current input setting')
        self.axis_output.plot(self.xs, self.ys2, color='tab:red', marker='o', linestyle='dotted', label='Predicted output level for current input setting')
        self.axis_output.plot(self.xs, self.ys3, color='tab:orange', marker='o', linestyle='dotted', label='Real output level for current input setting')
        if self.steps == 1:
            self.fig.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
