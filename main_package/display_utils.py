def display_numpy_image(img_numpy):
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot
    pyplot.imshow(img_numpy, interpolation="nearest")
    pyplot.show()