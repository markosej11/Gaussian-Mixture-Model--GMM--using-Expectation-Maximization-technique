from expMax import ExpMax
import numpy as np
from scipy import stats
import cv2
import sys
import os
import argparse


def main(args):
    # Parse arguments
    parser = argparse.ArgumentParser(description="ENPM673 - Robot Perception - Project 3:  Buoy segmentation")
    parser.add_argument('--plot', action="store_true", help="Plot expectation maximization convergence.")
    v = argparse.Namespace()
    args = parser.parse_args(args=args, namespace=v)
    plot = args.plot

    # Open training data
    count = 0
    frames = []
    frames_g = []
    frames_o = []
    frames_y = []
    folder = "images/"
    file_wrappers = [("frames/trainingData/frame", ".jpg"),
                     ("green/g", ".png"),
                     ("orange/o", ".png"),
                     ("yellow/y", ".png")]
    while all((os.path.exists("%s%s%d%s" % (folder, middle, count, tail))) for middle, tail in file_wrappers):
        frames.append(cv2.imread("%s%s%d%s" % (folder, file_wrappers[0][0], count, file_wrappers[0][1])))
        frames_g.append(cv2.imread("%s%s%d%s" % (folder, file_wrappers[1][0], count, file_wrappers[1][1])))
        frames_o.append(cv2.imread("%s%s%d%s" % (folder, file_wrappers[2][0], count, file_wrappers[2][1])))
        frames_y.append(cv2.imread("%s%s%d%s" % (folder, file_wrappers[3][0], count, file_wrappers[3][1])))
        count += 1
    sys.stdout.write("\nRead %d frames for training." % count)

    # Read pixels from masks for training data.
    pixel_colors = np.zeros((0, 3), dtype=np.float64)
    for i, frame in enumerate(frames):
        for j, mask_list in enumerate([frames_g, frames_o, frames_y]):
            mask = cv2.cvtColor(mask_list[i], cv2.COLOR_RGB2GRAY)
            if np.count_nonzero(mask) > 0:
                average_color = np.mean(frame[np.nonzero(mask)], axis=0).reshape((1, -1))
                pixel_colors = np.append(pixel_colors, average_color, axis=0)
    sys.stdout.write("\nGenerating model from %d marked buoys." % pixel_colors.shape[0])
    em = ExpMax(pixel_colors[:, 2], pixel_colors[:, 1], 3, colors=["#D0B000", "#00C000", "#FF8000"], max_iterations=50,
                point_alpha=1.0, covariance_weight=10.0, axis_titles=("Red", "Green"))
    x_means, y_means, cov = em.get_parameters()

    # Play animation of expectation maximization convergence.
    sys.stdout.write("\n\nModel generated.")
    if plot:
        sys.stdout.write("\nDisplaying expectation maximization convergence."
                         "\nPress any key to proceed when animation is paused.")
        em.play_animation()

    # Read original frames for testing data (including frames used for training)
    test_path = "images/frames/original/frame{0:d}.jpg"
    test_count = 0
    test_frames = []
    while os.path.exists(test_path.format(test_count)):
        test_frames.append(cv2.imread(test_path.format(test_count)))
        test_count += 1
    sys.stdout.write("\n\nProcessing %d frames for testing (including training frames)." % test_count)

    # Search for buoy shapes
    final_images = []
    for i, frame in enumerate(test_frames):
        blur_image = cv2.GaussianBlur(frame, (5, 5), 0)
        canny_image = cv2.Canny(blur_image, 40, 175, 3)

        # Search for circles within the image.
        circles = cv2.HoughCircles(canny_image, cv2.HOUGH_GRADIENT, 2, 20,
                                   param1=150, param2=42, minRadius=8, maxRadius=45)
        circle_mask = np.zeros_like(canny_image)
        if circles is not None:
            for x, y, r in circles[0]:
                cv2.circle(circle_mask, (int(x), int(y)), int(r) + 7, 255, thickness=cv2.FILLED)
        kernel = np.ones((5, 5), dtype=np.uint8)
        masked_canny = cv2.dilate(np.bitwise_and(circle_mask, canny_image), kernel=kernel)
        contours, hierarchy = cv2.findContours(masked_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= 300]
        buoy_image = np.copy(frame)
        outline_colors = [(0, 160, 216), (128, 128, 0), (0, 0, 192)]
        for c in contours:
            # Filter for contours with lower "inertia", i.e. closer to a circular shape.
            moments = cv2.moments(c)
            contour_cx = (moments["m10"] / moments["m00"])
            contour_cy = (moments["m01"] / moments["m00"])
            contour_area = cv2.contourArea(c)
            contour_image = cv2.drawContours(np.zeros_like(canny_image), [c], -1, 255, thickness=cv2.FILLED)
            contour_image = cv2.erode(contour_image, kernel=kernel)
            y_ind, x_ind = np.nonzero(contour_image)
            contour_dist = np.mean((x_ind - contour_cx) ** 2 + (y_ind - contour_cy) ** 2)
            area_distribution = contour_dist / contour_area
            new_contours, hierarchy = cv2.findContours(contour_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            masked_contour_image = np.bitwise_and(frame, cv2.cvtColor(contour_image, cv2.COLOR_GRAY2RGB))
            if np.count_nonzero(masked_contour_image) > 0 and area_distribution < 0.17:
                # Categorize the color of the confirmed buoy shape.
                average_color = np.mean(frame[np.nonzero(masked_contour_image)[:2]], axis=0).reshape((1, -1))
                bv, gv, rv = average_color[0]
                for j in range(3):
                    eigenvalues, eigenvectors = np.linalg.eig(cov[j])
                    rotation = -np.arctan2(eigenvectors[0][1], eigenvectors[0][0])
                    theta = np.arctan2(gv - y_means[j], rv - x_means[j]) + rotation
                    radius = np.sqrt((rv - x_means[j]) ** 2 + (gv - y_means[j]) ** 2)
                    c_level = 0.995  # Confidence level
                    xs = np.sqrt(cov[j][0, 0]) * np.sqrt(stats.chi2.ppf(c_level, 2))
                    ys = np.sqrt(cov[j][1, 1]) * np.sqrt(stats.chi2.ppf(c_level, 2))
                    if radius < xs * ys / np.sqrt((xs * np.cos(theta)) ** 2 + (ys * np.sin(theta)) ** 2):
                        # Draw original edges around buoy.
                        buoy_image = cv2.drawContours(buoy_image, new_contours, -1, outline_colors[j], thickness=2)
        final_images.append(buoy_image)
    cv2.destroyAllWindows()

    # Generate video of result.
    video_output_filename = "final_animation.mp4"
    sys.stdout.write("\nGenerating video of result with filename \"%s\"." % video_output_filename)
    writer = cv2.VideoWriter(video_output_filename, cv2.VideoWriter_fourcc('H', '2', '6', '4'), 15,
                             (test_frames[0].shape[1], test_frames[0].shape[0]))
    for frame in final_images:
        cv2.imshow("Buoy", frame)
        cv2.waitKey(75)
        writer.write(frame)
    sys.stdout.write("\n\nProgram completed!  (Ignore the following error if you get one...)\n%s\n" % ("-" * 80))
    writer.release()


if __name__ == '__main__':
    main(sys.argv[1:])
