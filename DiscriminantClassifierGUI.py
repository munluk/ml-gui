import Tkinter
import src.utils.utilities as utilities
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from src.utils.ConfigProvider import ConfigProvider as config_provider
from src.utils.utilities import DatasetException


class App:
    def __init__(self, master):
        # Create variables to build the data set
        self.c_red = np.array([], dtype=np.float64).reshape(0, 3)
        self.c_green = np.array([], dtype=np.float64).reshape(0, 3)
        self.c_blue = np.array([], dtype=np.float64).reshape(0, 3)
        self.selected_class_label = Tkinter.IntVar()
        self.selected_sample_option = Tkinter.IntVar()

        # set the main frames
        frame = Tkinter.Frame(master)
        self.left_frame = Tkinter.Frame(master)
        self.left_frame.pack(side='left')
        self.right_bottom_frame = Tkinter.Frame(frame)
        self.right_bottom_frame.pack(side='bottom')
        # Tkinter.Label(self.right_bottom_frame, text='right_bottom_frame').pack(side='left')

        # create the first three radio buttons
        self.class_labels = ['red', 'green', 'blue']
        for i in range(0, len(self.class_labels)):
            # print class_labels[i]
            self.button_class_labels = Tkinter.Radiobutton(self.left_frame, text=self.class_labels[i],
                                                           variable=self.selected_class_label, value=i, anchor='w')
            self.button_class_labels.pack(side='top', fill='both')

        # draw black line
        Tkinter.Frame(self.left_frame, height=1, width=70, bg='black').pack(side='top')

        # create the second radio buttons
        self.sample_option = ['single', 'gaussian']
        for i in range(0, len(self.sample_option)):
            print self.class_labels[i]
            self.button_sample_option = Tkinter.Radiobutton(self.left_frame, text=self.sample_option[i],
                                                            variable=self.selected_sample_option, value=i, anchor='w')
            self.button_sample_option.pack(side='top', fill='both')

        # create the entry objects to chose the variance
        variance_frame = Tkinter.Frame(self.left_frame)
        variance_frame.pack(side='top')
        Tkinter.Label(variance_frame, text=u'\u03c3\u2081\u00B2', anchor='w').grid(row=0, sticky='w')
        Tkinter.Label(variance_frame, text=u'\u03c3\u2082\u00B2', anchor='w').grid(row=1, sticky='w')
        Tkinter.Label(variance_frame, text='N', anchor='w').grid(row=2, sticky='w')
        self.var1 = Tkinter.Entry(variance_frame, width=5)
        self.var1.insert(Tkinter.END, '1.0')
        self.var1.grid(row=0, column=1, sticky='w')
        self.var2 = Tkinter.Entry(variance_frame, width=5)
        self.var2.insert(Tkinter.END, '1.0')
        self.var2.grid(row=1, column=1, sticky='w')
        self.N = Tkinter.Entry(variance_frame, width=5)
        self.N.grid(row=2, column=1, sticky='w')
        self.N.insert(Tkinter.END, '20')

        # draw black line
        Tkinter.Frame(self.left_frame, height=1, width=70, bg='black').pack(side='top')

        # create radio buttons to select the classifier
        self.selected_classifier = Tkinter.IntVar()
        self.classifier_option = config_provider.get_classifiers()
        keys = self.classifier_option.keys()
        for i in range(len(keys)):
            self.classifier_radiobutton = Tkinter.Radiobutton(self.left_frame,
                                                              text=keys[i],
                                                              variable=self.selected_classifier,
                                                              value=i, anchor='w')
            self.classifier_radiobutton.pack(side='top', fill='both')

        # create the button to activate classification
        self.train_button = Tkinter.Button(self.left_frame, text='Train', command=self.train_button_callback)
        self.train_button.pack(side='top')

        # create the plot on the right frame
        fig = Figure()
        self.plot_axes = fig.add_subplot(111, xlim=[-10, 10], ylim=[-10, 10])
        self.plot_axes.get_yaxis().set_visible(False)
        self.plot_axes.get_xaxis().set_visible(False)
        self.canvas = FigureCanvasTkAgg(fig, master=frame)
        self.canvas.callbacks.connect('button_press_event', self.mouse_callback)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        frame.pack()

        # create button to clear everything except the samples
        self.clear_button = Tkinter.Button(self.right_bottom_frame, text='Clear', command=self.clear_button_callback)
        self.clear_button.pack(side='left')

        # create button to clear the whole plot
        self.clear_all_button = Tkinter.Button(self.right_bottom_frame, text='Clear all',
                                               command=self.clear_all_button_callback)
        self.clear_all_button.pack(side='left')

        # create save button to save the current data set
        self.save_button = Tkinter.Button(self.right_bottom_frame, text='Save', command=self.save_data_set)
        self.save_button.pack(side='left')

    def clear_all_button_callback(self):
        """
        This function is the callback for the clear button. Clicking on the button clears all samples from the plot and
        empties the class containers
        """
        self.initialize_class_containers()
        # clear plot
        self.plot_axes.cla()
        self.plot_axes.set_xlim([-10, 10])
        self.plot_axes.set_ylim([-10, 10])
        self.canvas.draw()
        self.show_n_samples()

    def clear_button_callback(self):
        self.plot_axes.cla()
        self.plot_axes.set_xlim([-10, 10])
        self.plot_axes.set_ylim([-10, 10])
        for x, y in self.c_red[:, 0:2]:
            self.plot_axes.plot(x, y, marker='o', color='r', markersize=5)
        for x, y in self.c_green[:, 0:2]:
            self.plot_axes.plot(x, y, marker='o', color='g', markersize=5)
        for x, y in self.c_blue[:, 0:2]:
            self.plot_axes.plot(x, y, marker='o', color='b', markersize=5)
        self.canvas.draw()

    def mouse_callback(self, event):
        """
        This function is the callback for a mouse click event. When the user clicks onto the canvas/plot then samples of
        the selected class are created and stored into the corresponding container
        """
        if event.inaxes is not None:
            samples = self.create_samples(event)
            color = ''
            if self.class_labels[self.selected_class_label.get()] == 'red':
                self.c_red = np.concatenate((self.c_red, samples), axis=0)
                color = 'r'

            if self.class_labels[self.selected_class_label.get()] == 'green':
                self.c_green = np.concatenate((self.c_green, samples), axis=0)
                color = 'g'

            if self.class_labels[self.selected_class_label.get()] == 'blue':
                self.c_blue = np.concatenate((self.c_blue, samples), axis=0)
                color = 'b'
            # draw the samples
            for x, y in samples[:, 0:2]:
                self.plot_axes.plot(x, y, marker='o', color=color, markersize=5)
            self.canvas.draw()
            self.show_n_samples()

    def create_samples(self, event):
        """
        This functions creates samples by using the the location of the mouse on the canvas/plot
        """
        if self.sample_option[self.selected_sample_option.get()] == 'single':
            return np.array([[event.xdata, event.ydata, 1.0]])
        if self.sample_option[self.selected_sample_option.get()] == 'gaussian':
            samples = np.random.multivariate_normal([event.xdata, event.ydata],
                                                    np.diag([float(self.var1.get()), float(self.var2.get())]),
                                                    int(self.N.get()))
            return np.array(np.hstack((samples, np.ones((samples.shape[0], 1)))))

    def initialize_class_containers(self):
        """
        This function clears the storage containers for all classes red, green and blue
        """
        self.c_red = np.array([], dtype=np.float64).reshape(0, 3)
        self.c_green = np.array([], dtype=np.float64).reshape(0, 3)
        self.c_blue = np.array([], dtype=np.float64).reshape(0, 3)

    def show_n_samples(self):
        """
        This function prints the number of samples to the console output
        """
        print 'number of samples: ' + str(self.c_red.shape[0] + self.c_green.shape[0] + self.c_blue.shape[0]) \
              + '(red: %i, green: %i, blue: %i)' % \
                (self.c_red.shape[0], self.c_green.shape[0], self.c_blue.shape[0])

    def train_button_callback(self):
        """
        This function is the callback function for the classify button. When the button is pressed a classifier function
        is called to train a classifier on the samples in the containers. Which is classifier is selected with the
        variable self.selected_classifier which is the state variable for the radio buttons classifier_radiobutton
        """
        data, labels, valid_dims = self.get_data_set()

        # Here the selected classifier object is dynamically instantiated
        classifier_keys = self.classifier_option.keys()[self.selected_classifier.get()]
        func_name = self.classifier_option[classifier_keys]
        cls = __import__('src.classifiers.' + func_name, fromlist=func_name.encode(encoding='utf-8'))
        classifier_obj = getattr(cls, func_name)()

        # train the classifier and predict labels
        classifier_obj.train(data, labels)
        test_samples, meshx, meshy = utilities.generate_test_set()
        predicted_labels = classifier_obj.predict(test_samples)

        # draw decision boundaries
        utilities.draw_decision_boundaries(self.plot_axes, meshx, meshy, predicted_labels, valid_dims)
        self.canvas.draw()

    def get_data_set(self):
        """
        This function packs the samples into a data matrix (homogenous [x,y,1]) and the labels in to a one hot matrix
        and returns them
        """
        one_hot_red = np.zeros((self.c_red.shape[0], 3))
        one_hot_green = np.zeros((self.c_green.shape[0], 3))
        one_hot_blue = np.zeros((self.c_blue.shape[0], 3))
        one_hot_red[:, 0] = 1
        one_hot_green[:, 1] = 1
        one_hot_blue[:, 2] = 1
        data = np.vstack((self.c_red, self.c_green, self.c_blue))
        labels = np.vstack((one_hot_red, one_hot_green, one_hot_blue))

        dims_valid = np.any(labels, axis=0)
        if sum(dims_valid) < 2:
            raise DatasetException("Not enough classes in data set")
        dims_to_remove = [idx for idx, val in enumerate(dims_valid) if not val]
        labels = np.delete(labels, dims_to_remove, 1)
        return data, labels, dims_valid

    def save_data_set(self):
        """
        This function saves the current data set to the working folder
        :return:
        """
        samples, labels, valid_dims = self.get_data_set()
        try:
            np.save('samples.npy', samples)
            np.save('labels.npy', labels)
            print utilities.bcolors.OKGREEN + 'Succesfully saved the data set' + utilities.bcolors.ENDC
        except Exception as e:
            raise e.message


if __name__ == "__main__":
    # run GUI
    root = Tkinter.Tk()
    root.resizable(width=False, height=False)
    root.wm_title("ml-gui")
    app = App(root)
    root.mainloop()
