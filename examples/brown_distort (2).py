def compute_vectors(star_centroids,hfov,k1,k2,p1,p2,width,height):
    """Get unit vectors from star centroids (pinhole camera)."""
    # compute list of (i,j,k) vectors given list of (y,x) star centroids and
    # an estimate of the image's field-of-view in the x dimension
    # by applying the pinhole camera equations
    #k1: radial distortion term. Should be small. Start off at 0.
    #k2: radial distortion term. Should be small. Start off at 0. Should be smaller than k1.
    #p1: tangential distortion term. Should be small. Start off at 0.
    #p2: tangential distortion term. Should be small. Start off at 0.
    center_x = width / 2.
    center_y = height / 2.
    scale_factor = np.tan(hfov / 2) / center_x  #this is the plate scale

    star_vectors = []
    for (star_y, star_x) in star_centroids:
        x_d=center_x-star_x
        y_d=center_y-star_y
        r_d=np.sqrt(x_d**2+y_d**2)
        star_x=star_x+(x_d*(k1*r_d**2+k2*r_d**4))+(p1*(r_d**2+2*(x_d)**2))+(2*p2*x_d*y_d)
        star_y=star_y+(y_d*(k1*r_d**2+k2*r_d**4))+(2*p1*x_d*y_d)+(p2*(r_d**2+2*(y_d)**2))
        j_over_i = (center_x - star_x) * scale_factor
        k_over_i = (center_y - star_y) * scale_factor
        i = 1. / np.sqrt(1 + j_over_i**2 + k_over_i**2)
        j = j_over_i * i
        k = k_over_i * i
        vec = np.array([i, j, k])
        print(f'compute_vectors2a: {vec}')
        star_vectors.append(vec)

        #estimated paremeters
        focal_length = 1/scale_factor
        vfov=2*np.arctan(scale_factor*center_y)
        plate_scale = hfov/width    #radian/pixel
        #also want to record best fits for k1,k2,p1,p2
        #####################
    return star_vectors