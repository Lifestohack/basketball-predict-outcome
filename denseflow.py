#!/usr/bin/env python

import numpy as np
import cv2 as cv
import os

def __draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.9)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def __draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*8, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


def __warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res

def flow_to_color(flow, hsv):
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

def get_optical_flow(images, visual=False):
    if images is None:
        raise RuntimeError('No images found.')
    #if images[0].shape[0] == 3:
    #    images = [np.moveaxis(image, 0, 2) for image in images]
    images = iter(images)

    prev = next(images, None)
    prev = np.array(prev)
    
    #prev = cv.imread(firstframepath) #Beware that cv.imread() returns a numpy array in BGR not RGB

    #_ret, prev = cam.read()
    prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    show_hsv = True
    show_glitch = False
    show_vector = False
    cur_glitch = prevgray.copy()
    i = 0
    video = []
    hsv = np.zeros_like(prev)
    hsv[..., 1] = 255
    for img in images:
        #img = cv.imread(imagepath)
        img = np.array(img)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray
        #savepath = imagepath.replace(processfrom, processto)
        #savedir = os.path.dirname(savepath)
        #if not os.path.isdir(savedir):
        #    os.makedirs(savedir)
        new_img = flow_to_color(flow, hsv)
        #cv.imwrite(savepath, new_img)
        video.append(new_img)
        i+=1
        if visual:
            if show_vector:
                cv.imshow('flow', __draw_flow(gray, flow))
            if show_hsv:
                cv.imshow('flow HSV', __draw_hsv(flow))
            if show_glitch:
                cur_glitch = __warp_flow(cur_glitch, flow)
                cv.imshow('glitch', cur_glitch)

            ch = cv.waitKey(5)
            if ch == 27:
                break
            if ch == ord('1'):
                show_hsv = not show_hsv
                print('HSV flow visualization is', ['off', 'on'][show_hsv])
            if ch == ord('2'):
                show_glitch = not show_glitch
                if show_glitch:
                    cur_glitch = img.copy()
                print('glitch is', ['off', 'on'][show_glitch])
    return video
        