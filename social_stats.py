import sys
# use capturejs
# https://github.com/superbrothers/capturejs
if __name__ == '__main__':
    img1 = cv2.imread('sample/twitter/sample.png', 0)
    img2 = cv2.imread('sample/screenshot.png', 0)
    detector, matcher = init_feature(feature_name)
    if detector != None:
        print 'using', feature_name
    else:
        print 'unknown feature:', feature_name
        sys.exit(1)


    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)
    print 'img1 - %d features, img2 - %d features' % (len(kp1), len(kp2))

    def match_and_draw(win):
        print 'matching...'
        raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
        if len(p1) >= 4:
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            print '%d / %d  inliers/matched' % (np.sum(status), len(status))
        else:
            H, status = None, None
            print '%d matches found, not enough for homography estimation' % len(p1)

        vis = explore_match(win, img1, img2, kp_pairs, status, H)

    match_and_draw('find_obj')
    cv2.waitKey()
    cv2.destroyAllWindows()