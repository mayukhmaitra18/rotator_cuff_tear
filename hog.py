# Importing the libraries
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os

path = './images'
for images in os.listdir(path):

    image = cv2.imread("./images/" + str(images))
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #applying hough for curve detection
    inputImageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smoothImage = cv2.GaussianBlur(inputImageGray, (5, 5), 0)

    edges = cv2.Canny(smoothImage, 150, 200, apertureSize=3)
    minLineLength = 1000
    maxLineGap = 5
    lines = cv2.HoughLinesP(edges, cv2.HOUGH_PROBABILISTIC, np.pi / 180, 30, minLineLength, maxLineGap)
    for m in range(0, len(lines)):
        for m1, n1, m2, n2 in lines[m]:
            pts = np.array([[m1, n1], [m2, n2]], np.int32)
            cv2.polylines(image, [pts], True, (0, 255, 0))
    
    
    # Setting HOG Parameters
    cell_size = (10, 10)
    cells_per_block = (2, 2)
    horizontal_stride = 1
    vertical_stride = 1
    hist_bars = 15

    block_size = (cells_per_block[0] * cell_size[0],
                  cells_per_block[1] * cell_size[1])

    # Total number of cells in image
    x_cell = image.shape[1] // cell_size[0]
    y_cell = image.shape[0] // cell_size[1]

    # Block stride
    block_stride = (cell_size[0] * horizontal_stride,
                    cell_size[1] * vertical_stride)

    # Window size
    win_size = (x_cell * cell_size[0],
                y_cell * cell_size[1])


    HOG = cv2.HOGDescriptor(win_size,                       # Detection window pixels
                            block_size,                     # Cells per block.
                            block_stride,                   # distance between blocks
                            cell_size,                      # cell size
                            hist_bars)                      # num of histogram's bars.

    HOG_descriptor = HOG.compute(image)


    # blocks inside detection window (x axis)
    total_blocks_x = np.uint32(((x_cell - cells_per_block[0]) / horizontal_stride) + 1)

    # blocks inside detection window (y axis)
    total_blocks_y = np.uint32(((y_cell - cells_per_block[1]) / vertical_stride) + 1)

    # feature vector elements
    total_elements = (total_blocks_x) * (total_blocks_y) * cells_per_block[0] * cells_per_block[1] * hist_bars

    # feature vector reshaped
    hog_descriptor_reshaped = HOG_descriptor.reshape(total_blocks_x,
                                                     total_blocks_y,
                                                     cells_per_block[0],
                                                     cells_per_block[1],
                                                     hist_bars)


    # Transposing the blocks
    hog_descriptor_reshaped = hog_descriptor_reshaped.transpose((1, 0, 2, 3, 4))

    # Initializing "Average Gradient per Cell" with zeros
    ave_grad = np.zeros((y_cell, x_cell, hist_bars))

    # Initializing an array of zeros for counting the number of histograms per cell
    hist_counter = np.zeros((y_cell, x_cell, 1))

    # Adding up all the histograms for each cell
    for i in range(cells_per_block[0]):
        for j in range(cells_per_block[1]):
            ave_grad[i:total_blocks_y + i,
            j:total_blocks_x + j] += hog_descriptor_reshaped[:, :, i, j, :]
            hist_counter[i:total_blocks_y + i,
            j:total_blocks_x + j] += 1

    # Average gradient for each cell
    ave_grad /= hist_counter

    # Total number of vectors in all cells
    len_vecs = ave_grad.shape[0] * ave_grad.shape[1] * ave_grad.shape[2]

    # Equally spacing degrees between 0-180 degrees radians to the hist_bars
    deg = np.linspace(0, np.pi, hist_bars, endpoint = False)

    # Creating arrays for holding components of vector
    U = np.zeros((len_vecs))
    V = np.zeros((len_vecs))

    # Creating arrays for holding positions of vector
    X = np.zeros((len_vecs))
    Y = np.zeros((len_vecs))

    # Set the counter to zero
    counter = 0

    # Calculating the positions and components

    # Iterating thorogh x axis of average_gradients
    for i in range(ave_grad.shape[0]):

        # Iterating through y axis of average_gradients
        for j in range(ave_grad.shape[1]):

            # Iterating through z axis of average_gradients
            for k in range(ave_grad.shape[2]):
                # Computing Components
                U[counter] = ave_grad[i, j, k] * np.cos(deg[k])
                V[counter] = ave_grad[i, j, k] * np.sin(deg[k])

                # Computing positions
                X[counter] = (cell_size[0] / 2) + (cell_size[0] * i)
                Y[counter] = (cell_size[1] / 2) + (cell_size[1] * j)

                # Incrementing the counter
                counter = counter + 1

    # equally spacing the histogram between 0-180 degrees
    angle_axis = np.linspace(0, 180, hist_bars, endpoint = False)

    # Adding (half of two degrees = 10) degrees to each
    angle_axis += ((angle_axis[1] - angle_axis[0]) / 2)

    # subplot figure size
    plt.rcParams['figure.figsize'] = [15, 5]

    # final figure with 3 subplots
    fig, (HOG_on_img, zoom, hist) = plt.subplots(1, 3)


    """Visualizing the HOG"""
    # Onclick event


    def onpress(event):
        i = 0
        if event.inaxes in [HOG_on_img]:

            # mouse click co-ordinates
            x, y = event.xdata, event.ydata

            # selecting the cell that is near to the place where click is made
            cell_num_x = np.uint32(x / cell_size[0])
            cell_num_y = np.uint32(y / cell_size[1])

            # create a rectangle to show the selected cell above
            edgex = x - (x % cell_size[0])
            edgey = y - (y % cell_size[1])

            rect = patches.Rectangle((edgex, edgey),
                                     cell_size[0], cell_size[1],
                                     linewidth=1.5,
                                     edgecolor='#f000c8',
                                     facecolor='none')

            # creating copies to use in multiple subplots
            rect4 = copy.copy(rect)
            rect5 = copy.copy(rect)

            # visualizing the HOG descriptor on top of the MRI image
            HOG_on_img.clear()
            HOG_on_img.set(
                title='HOG Descriptor')
            HOG_on_img.quiver(Y, X, U, V, headwidth=0, headlength=0, scale_units='inches', scale=5)
            HOG_on_img.invert_yaxis()
            HOG_on_img.set_aspect(aspect=1)
            HOG_on_img.set_facecolor('white')
            HOG_on_img.imshow(image, cmap='gray')
            HOG_on_img.add_patch(rect4)
            HOG_on_img.xaxis.set_visible(False)
            HOG_on_img.yaxis.set_visible(False)
            HOG_on_img.imshow(image, cmap='gray')

            # zoomed in view of the selected cell on the image
            zoom.clear()
            zoom.set(title='HOG Descriptor (Zoom Window)')
            zoom.quiver(Y, X, U, V, headwidth=0, headlength=0, scale_units='inches', scale=1)
            zoom.set_xlim(edgex - cell_size[0], edgex + (2 * cell_size[0]))
            zoom.set_ylim(edgey - cell_size[1], edgey + (2 * cell_size[1]))
            zoom.invert_yaxis()
            zoom.set_aspect(aspect=1)
            zoom.set_facecolor('white')
            zoom.add_patch(rect5)
            zoom.xaxis.set_visible(False)
            zoom.yaxis.set_visible(False)
            zoom.imshow(image, cmap='gray')

            # visualizing the histogram for the selected cell
            hist.clear()
            hist.set(title='Histogram of Gradients')
            hist.grid()
            hist.set_xlim(0, 180)
            hist.set_xticklabels(angle_axis)
            hist.set_xlabel('Angle')
            hist.set_ylabel("Orientation")
            hist.bar(angle_axis,
                  ave_grad[cell_num_y, cell_num_x, :],
                  180 // hist_bars,
                  align='center',
                  alpha=0.5,
                  linewidth=1.2,
                  edgecolor='k',
                  color='#f000c8')

            fig.canvas.draw()
            plot_name = './hog_res/hog_'+str(images)
            fig.savefig(plot_name)


    # Create a connection between the figure and the mouse click
    fig.canvas.mpl_connect('button_press_event', onpress)
    plt.show()
