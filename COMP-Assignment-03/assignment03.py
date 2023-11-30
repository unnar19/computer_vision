import cv2
import time
import os
import numpy as np

cap = cv2.VideoCapture(0)

#os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp;buffer_size;2'
#cap = cv2.VideoCapture('rtsp://10.1.19.46/video', cv2.CAP_FFMPEG)

(w,h) = (int(cap.get(3)), int(cap.get(4)))

prev_time = time.time()

keep_delta = []
print_delta = 0
print_fps = 0

while(True):

    ret, original = cap.read()

    frame = np.copy(original)

    edge = cv2.Canny(frame,170,200)
    edge_bgr = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    cv2.addWeighted(frame, 1, edge_bgr, 1,0)

    line = cv2.HoughLinesP(edge,rho=5*np.pi/8, theta=2*np.pi/180,threshold=90, minLineLength=120, maxLineGap=70)

    save_mb = []
    save_mb2 = []
    save_line = []
    if type(line) != type(None):
        if len(line) >= 4:

            # Save all y = mx + b
            for l in line:
                x1, y1, x2, y2 = l[0]
                if x1 == x2:
                    m = 0
                else:
                    m = (np.double(y2)-y1)/(x2-x1)

                b = y1 - m * x1
                save_mb.append((m,b))

            # Remove duplicates
            for i, mb in enumerate(save_mb):
                m = mb[0]
                b = mb[1]

                unique = True
                for existing in save_mb2:

                    if abs(m - existing[0]) <= 5*np.pi/180 and abs(b - existing[1]) <= 50+15*m:
                        unique = False
                        break

                if unique:
                    save_mb2.append((m,b))
                    save_line.append(line[i])

        for l in save_line:
            x1, y1, x2, y2 = l[0]
            cv2.line(frame,(x1, y1),(x2, y2),(255,0,0),4)

        if len(save_line) >= 4:

            # Find intersects
            corners = []
            for line1 in save_mb2:
                cv2.line(frame,(0,int(line1[1])), (w,int(line1[0]*w+line1[1])),(0,255,0),1)
                
                for line2 in save_mb2:
                    x = -(line1[1]-line2[1])/(line1[0]-line2[0])
                    y = line1[0]*x + line1[1]
                    if x >= 20 and x <= w-20 and y >= 20 and y <= h - 20:
                        corners.append((x,y))

            corners = np.array(list(set(corners)), dtype="float32")

            if len(corners) >= 4 and len(corners) <= 10:
                hull = cv2.convexHull(corners)
                epsilon = 0.02 * cv2.arcLength(hull, closed=True)
                approx = cv2.approxPolyDP(hull, epsilon, closed=True)

                if len(approx) == 4:
                    corners = approx.reshape(-1,2)

                    # top left, top right, bottom right, bottom left
                    window_shape = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")

                    # reorder corners to the same as warp shape
                    reordered_corners = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype="float32")

                    x_values = []
                    y_values = []

                    for k in range(4):
                        cv2.circle(frame, (int(corners[k][0]),int(corners[k][1])), 5, (0,0,255), 3)

                        best_dist = w*2
                        for i, corner in enumerate(corners):
                            dist = np.linalg.norm([corner[0]-window_shape[k][0],corner[1]-window_shape[k][1]])

                            if dist < best_dist:
                                best_dist = dist
                                best_i = i

                        x_values.append(corners[k][0])
                        y_values.append(corners[k][1])

                        reordered_corners[k] = corners[best_i]
                        
                    w_rect = 350
                    h_rect = 350

                    warp_shape = np.array([[0, 0], [w_rect, 0], [w_rect, h_rect], [0, h_rect]], dtype="float32")
                    perspective = cv2.getPerspectiveTransform(reordered_corners, warp_shape)
                    rectified = cv2.warpPerspective(original, perspective, (w_rect, h_rect))
                    cv2.imshow('Last detected rectangle frame', frame)
                    cv2.imshow('Rectified rectangle', rectified)

    time_delta = time.time() - prev_time
    prev_time = time.time()

    keep_delta.append(time_delta)

    if len(keep_delta) == 30:
        print_delta = np.mean(keep_delta)*1000
        print_fps = 1/np.mean(keep_delta)

        keep_delta, keep_score = [], []

    cv2.putText(frame, "FPS:",(8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0),1, cv2.LINE_AA)
    cv2.putText(frame, f"{print_fps:.2f}",(60,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0),1, cv2.LINE_AA)
    cv2.putText(frame, "Delay:",(8,36), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0),1, cv2.LINE_AA)
    cv2.putText(frame, f"{print_delta:.0f} ms",(60,36), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0),1, cv2.LINE_AA)
    
    #cv2.imshow('RGB',frame)
    

    cv2.imshow('RGB',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()