#!/usr/bin/env python3
import torch
from demo_superpoint import SuperPointFrontend, PointTracker
from demo_superpoint import myjet
import argparse
import rosbag
import numpy as np
import cv2
import os


topic_colour = "/camera/rgb/image_rect_color/compressed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('bag_path', type=str)
    parser.add_argument('--components', type=str,
        help="path to PCA component matrix (n_components, n_features) for projecting features to 3D RGB space")
    parser.add_argument('--decimate', type=int,
        help="reduce image size by only keeping certain rows and columns")
    parser.add_argument('--export', type=str,
        help='export images')
    args = parser.parse_args()

    if args.components:
        components = np.loadtxt(args.components)

    if args.export is not None and not os.path.isdir(args.export):
        os.makedirs(args.export)

    bag = rosbag.Bag(args.bag_path, 'r')

    min_length = 2
    max_length = 1000
    nn_thresh = 0.7

    spfe = SuperPointFrontend(weights_path='superpoint_v1.pth', nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=torch.cuda.is_available())

    tracker = PointTracker(max_length, nn_thresh=spfe.nn_thresh)

    win_tracks = "tracks"
    cv2.namedWindow(win_tracks, cv2.WINDOW_NORMAL)
    win_matches = "matches"
    cv2.namedWindow(win_matches, cv2.WINDOW_NORMAL)
    if args.components:
        win_feature = "feature"
        cv2.namedWindow(win_feature, cv2.WINDOW_NORMAL)

    img0 = None
    pts0 = None
    desc0 = None

    for topic, msg, t in bag.read_messages(topics=[topic_colour]):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
        # 8bit RGB colour image
        cimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if args.decimate is not None:
            img = img[::args.decimate, ::args.decimate]

        # float [0,1] for prediction
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255

        if img0 is None:
            img0 = img

        pts, desc, coarse_desc, heatmap = spfe.run(img)

        if pts0 is None:
            pts0 = pts

        if desc0 is None:
            desc0 = desc

        # match current and initial descriptors
        matches = tracker.nn_match_two_way(desc, desc0, nn_thresh=nn_thresh)

        pts_i = pts.T[:, :2].astype(np.int).tolist()
        pts_j = pts0.T[:, :2].astype(np.int).tolist()

        img0_points = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        img1_points = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        marker_size = 3
        for i, (id_i, id_j, score) in enumerate(matches.T):
            c = myjet[int(i%myjet.shape[0])]*255
            cv2.circle(img0_points, tuple(pts_j[int(id_j)]), marker_size, c)
            cv2.circle(img1_points, tuple(pts_i[int(id_i)]), marker_size, c)
        img_matches = cv2.hconcat((img1_points, img0_points), 2)

        pts_j = (pts0.T[:, :2] + [img1_points.shape[1], 0]).astype(np.int).tolist()
        for i, (id_i, id_j, score) in enumerate(matches.T):
            c = myjet[int(i%myjet.shape[0])]*255
            cv2.line(img_matches, tuple(pts_i[int(id_i)]), tuple(pts_j[int(id_j)]), c)

        cv2.imshow(win_matches, img_matches)

        if args.export:
            cv2.imwrite(os.path.join(args.export, "matches_{}.png").format(t), img_matches*255)

        tracker.update(pts, desc)
        tracks = tracker.get_tracks(min_length)

        out1 = (np.dstack((img, img, img)) * 255.).astype('uint8')
        tracks[:, 1] /= float(nn_thresh) # Normalize track scores to [0,1].

        tracker.draw_tracks(out1, tracks)

        cv2.resizeWindow(win_tracks, 2*cimg.shape[1], 2*cimg.shape[0])
        cv2.imshow(win_tracks, out1)

        if args.export:
            cv2.imwrite(os.path.join(args.export, "tracks_{}.png").format(t), out1)

        if args.components:
            coarse_desc = np.moveaxis(coarse_desc[0], 0, -1) @ components.T
            coarse_desc += 0.5
            coarse_desc = coarse_desc / np.linalg.norm(coarse_desc, axis=-1, keepdims=True)
            coarse_desc = cv2.resize(coarse_desc, tuple(np.roll(img.shape[:2], 1)))

            cv2.resizeWindow(win_feature, 2*cimg.shape[1], 2*cimg.shape[0])

            out2 = np.dstack((img, img, img))
            out2 = out2*0.1 + coarse_desc*0.9
            cv2.imshow(win_feature, out2)

            if args.export:
                cv2.imwrite(os.path.join(args.export, "features_{}.png").format(t), out2*255)

        cv2.waitKey(1)

    bag.close()