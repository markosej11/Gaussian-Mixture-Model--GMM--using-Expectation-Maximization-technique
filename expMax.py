import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy import stats
import warnings
warnings.filterwarnings('error')


class ExpMax:
    def __init__(self, dx, dy, means, colors=None, indices=None, bounds=(0, 0, 255, 255), max_iterations=30,
                 point_alpha=1.0, covariance_weight=1.0, axis_titles=("x", "y")):
        self.data_x = dx
        self.data_y = dy
        self.n = self.data_x.shape[0]
        self.colors = colors if colors is not None else ["#000000"] * means
        self.original = indices
        self.clusters = indices if indices is not None else (np.zeros((self.n,), dtype=np.int32) - 1)
        self.k = means
        self.x_means = []
        self.x_std = []
        self.y_means = []
        self.y_std = []
        self.weights = np.zeros((self.k, self.n), dtype=np.float64)
        self.cov = []
        self.plot_frames = []
        self.plot_bounds = bounds
        self.max_iterations = max_iterations
        self.plot_point_alpha = point_alpha
        self.covariance_weight = covariance_weight
        self.x_title, self.y_title = axis_titles
        success = False
        while not success:
            self.plot_frames = []
            success = self.maximize()

    def initialize_means(self):
        avg_xmean = np.mean(self.data_x)
        avg_ymean = np.mean(self.data_y)
        avg_xstd = np.std(self.data_x)
        avg_ystd = np.std(self.data_y)
        min_distance = -1.0
        for i in range(5):
            theta = i / self.k / 5
            x_means = [avg_xmean + avg_xstd * np.cos((theta + k / self.k) * 2.0 * np.pi) for k in range(self.k)]
            y_means = [avg_ymean + avg_ystd * np.sin((theta + k / self.k) * 2.0 * np.pi) for k in range(self.k)]
            new_distance = np.sum(np.min(np.array([(self.data_x - x_means[k]) ** 2 + (self.data_y - y_means[k]) ** 2
                                                   for k in range(self.k)]), axis=0))
            if new_distance < min_distance or min_distance < 0.0:
                min_distance = new_distance
                self.x_means = x_means
                self.y_means = y_means
        self.x_std = [avg_xstd / self.k] * self.k
        self.y_std = [avg_ystd / self.k] * self.k
        self.cov = [np.diagflat([avg_xstd ** 2, avg_ystd ** 2]).astype(np.float64) + 1.0] * self.k

    def maximize(self, tolerance=0.0005):
        """
        Expectation Maximization Implementation

        Parameters
        ----------
        tolerance: float
            The ratio of the total change in the location of the means to the total of the standard deviations.
            As the plots converge, this ratio decreases, and when it gets below tolerance, the EM has converged.

        Returns
        -------
        bool
            True if convergence was successful, False otherwise
        """
        # Evenly distribute means and standard deviations for initial guesses.
        self.initialize_means()
        priors = np.array([[1.0/self.k] * self.n] * self.k)
        p_gauss = np.zeros((self.k, self.n), dtype=np.float64)
        delta = tolerance * 2.0

        # E-step and M-step implementations
        self.plot(0, with_gaussian=False)
        iteration = 1
        while delta >= tolerance and iteration < self.max_iterations:
            # E-step:  Estimate responsibilities
            try:
                for j in range(self.k):
                    p_gauss[j] = np.exp(-(((self.data_x - self.x_means[j]) / (np.sqrt(2) * self.x_std[j])) ** 2) -
                                        ((self.data_y - self.y_means[j]) / (np.sqrt(2) * self.y_std[j])) ** 2) / \
                                 np.sqrt(2.0 * np.pi * np.abs(self.cov[j][0, 1]) ** self.covariance_weight)
            except Warning:  # In case a Gaussian vanishes, where standard deviation = 0.0
                return False
            self.weights = p_gauss * priors
            total_weights = np.sum(self.weights, axis=0)
            self.weights /= np.repeat(np.expand_dims(total_weights, axis=0), self.k, axis=0)

            self.clusters = np.argmax(self.weights, axis=0)
            self.plot(iteration)

            # M-step: Estimate parameters
            priors = np.repeat(np.expand_dims(np.sum(self.weights, axis=1) / self.n, axis=1), self.n, axis=1)
            contribution = self.weights / (self.n * priors)
            x_prev = self.x_means
            y_prev = self.y_means
            self.x_means = [np.sum(self.data_x * contribution[j]) for j in range(self.k)]
            self.y_means = [np.sum(self.data_y * contribution[j]) for j in range(self.k)]
            self.x_std = [np.sqrt(np.sum((self.data_x - self.x_means[j]) ** 2 * contribution[j]))
                          for j in range(self.k)]
            self.y_std = [np.sqrt(np.sum((self.data_y - self.y_means[j]) ** 2 * contribution[j]))
                          for j in range(self.k)]
            for j in range(self.k):
                cov = np.sum((self.data_x - self.x_means[j]) * (self.data_y - self.y_means[j]) * contribution[j])
                self.cov[j] = np.array([[self.x_std[j] ** 2, cov], [cov, self.y_std[j] ** 2]], dtype=np.float64)

            # Check for convergence using the total distance that the Gaussians traveled
            x_dist = [x_prev[i] - self.x_means[i] for i in range(self.k)]
            y_dist = [y_prev[i] - self.y_means[i] for i in range(self.k)]
            total_dist = sum([np.sqrt(x_dist[i] ** 2 + y_dist[i] ** 2) for i in range(self.k)])
            total_std = sum([np.sqrt(self.x_std[i] ** 2 + self.y_std[i] ** 2) for i in range(self.k)])
            delta = total_dist / total_std
            iteration += 1
        self.plot(iteration)
        return True

    def plot(self, index, with_gaussian=True):
        """
        Parameters
        ----------
        index: int
            The iteration of the expectation maximization, to include in the title of the plot.

        with_gaussian: bool
            Whether to plot the Gaussian distributions along with the data points (True) or not (False).

        References
        ----------
        https://matplotlib.org/gallery/units/ellipse_with_units.html#sphx-glr-gallery-units-ellipse-with-units-py
        https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
        """
        fig = plt.figure(figsize=(8.0, 6.4))
        width = int((fig.get_size_inches() * fig.get_dpi())[0])
        height = int((fig.get_size_inches() * fig.get_dpi())[1])
        canvas = FigureCanvas(fig)
        plt.title("Data (Iteration: %d)" % index)
        plt.xlabel(self.x_title)
        plt.ylabel(self.y_title)
        plt.xlim(self.plot_bounds[0] - 10.0, self.plot_bounds[2] + 10.0)
        plt.ylim(self.plot_bounds[1] - 10.0, self.plot_bounds[3] + 10.0)
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.85, top=0.85)

        # Plot data points
        colors = np.array([self.colors[c] if c >= 0 else "#000000" for c in self.clusters])
        plt.scatter(self.data_x, self.data_y, s=1.0, c=colors, alpha=self.plot_point_alpha)

        # Plot Gaussian distributions for 90.0 %, 97.5 %, and 99.5 % confidence levels
        if with_gaussian:
            theta = np.arange(0, 360, 2).astype(np.float64) * np.pi / 180.0
            confidence = stats.chi2.ppf([0.9, 0.975, 0.995], 2)
            for i in range(self.k):
                xc = self.x_means[i]
                yc = self.y_means[i]
                try:
                    eigenvalues, eigenvectors = np.linalg.eig(self.cov[i])
                except np.linalg.LinAlgError:
                    return
                rotation = -np.arctan2(eigenvectors[0][1], eigenvectors[0][0])
                r_mat = np.array([[np.cos(rotation), -np.sin(rotation)],
                                  [np.sin(rotation), np.cos(rotation)]], dtype=np.float64)
                for j, c in enumerate(confidence):
                    x_points = self.x_std[i] * np.cos(theta) * np.sqrt(c)
                    y_points = self.y_std[i] * np.sin(theta) * np.sqrt(c)
                    x_points, y_points = np.dot(r_mat, np.array([x_points, y_points]))
                    x_points += xc
                    y_points += yc
                    plt.fill(x_points, y_points, alpha=0.125/(1+j), facecolor=self.colors[i])
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
        self.plot_frames.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        plt.close(fig)

    def get_parameters(self):
        """
        Retrieve resulting means and covariance matrices from the expectation maximization.

        Returns
        -------
        tuple
            x_means, y_means, cov: parameters to use for data categorization
        """
        return self.x_means, self.y_means, self.cov

    def play_animation(self):
        for i in range(len(self.plot_frames)):
            cv2.imshow("Plot", self.plot_frames[i])
            cv2.waitKey(0 if (i <= 1 or i == len(self.plot_frames) - 1) else 200)
        cv2.destroyWindow("Plot")


def main():
    params = [(35.0, 7.0, 30.0, 4.0, 200, "#FF0000"),
              (40.0, 6.0, 70.0, 9.0, 300, "#00C000"),
              (70.0, 5.0, 50.0, 12.0, 400, "#0000FF")]
    sample_x = np.zeros((0,), dtype=np.float64)
    sample_y = np.zeros((0,), dtype=np.float64)
    sample_i = np.zeros((0,), dtype=np.int32)
    sample_c = []
    for i in range(len(params)):
        sxm, sxs, sym, sys, sn, color = params[i]
        sample_x = np.concatenate((sample_x, np.random.normal(sxm, sxs, sn)), axis=0)
        sample_y = np.concatenate((sample_y, np.random.normal(sym, sys, sn)), axis=0)
        sample_i = np.concatenate((sample_i, np.array([i] * sn)), axis=0)
        sample_c.append(color)
    sample_y[:200] -= (sample_x[:200] * 0.5)
    em = ExpMax(sample_x, sample_y, len(params), colors=sample_c, indices=None, bounds=(0, 0, 100, 100))
    em.play_animation()


if __name__ == '__main__':
    main()
