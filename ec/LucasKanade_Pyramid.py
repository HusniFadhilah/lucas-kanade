def LucasKanadePyramid(It, It1, rect, levels=3, p0=np.zeros(2)):
    """
    Lucas-Kanade tracker with image pyramid.
    """
    # Create image pyramid
    pyramid_It = [It]
    pyramid_It1 = [It1]
    for i in range(1, levels + 1):
        pyramid_It.append(gaussian_filter(pyramid_It[-1], sigma=2)[::2, ::2])
        pyramid_It1.append(gaussian_filter(pyramid_It1[-1], sigma=2)[::2, ::2])

    # Adjust rectangle for the top level of the pyramid
    rect_pyramid = rect / (2 ** levels)

    # Start tracking from the coarsest level
    for level in range(levels, -1, -1):
        It_level = pyramid_It[level]
        It1_level = pyramid_It1[level]

        # Use the Lucas-Kanade tracker at this level
        p = LucasKanade(It_level, It1_level, rect_pyramid, p0)

        # Adjust rectangle and p0 for the next finer level
        rect_pyramid = rect_pyramid * 2
        p0 = p * 2

    return p0